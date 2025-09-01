import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from torch import amp as _amp  # 新：使用新的 AMP 接口

from src.dataset import KvasirSegDataset
from src.model import UNet
from src.loss import DiceLoss, BCEDiceLoss
from src.engine import train_one_epoch, validate
from src.utils import set_seed, ensure_dir, save_checkpoint

def build_loss(name: str):
    name = name.lower()
    if name == "dice":
        return DiceLoss()
    elif name == "bce":
        return torch.nn.BCEWithLogitsLoss()
    elif name == "bce_dice":
        return BCEDiceLoss(0.5)
    else:
        raise ValueError(f"Unknown loss: {name}")

def build_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def build_scheduler(name: str, optimizer, epochs: int):
    name = name.lower()
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(5, epochs//3), gamma=0.5)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unknown scheduler: {name}")

def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = cfg["dataset_dir"]
    img_size = int(cfg.get("img_size", 256))
    batch_size = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 4))
    pin_memory = bool(cfg.get("pin_memory", True))
    epochs = int(cfg.get("epochs", 50))
    lr = float(cfg.get("learning_rate", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    optimizer_name = cfg.get("optimizer", "adamw")
    scheduler_name = cfg.get("scheduler", "cosine")
    loss_name = cfg.get("loss", "bce_dice")
    mixed_precision = bool(cfg.get("mixed_precision", True))
    val_every = int(cfg.get("val_every", 1))
    early_patience = int(cfg.get("early_stop_patience", 15))
    experiment_name = cfg.get("experiment_name", "exp1")
    save_root = Path(cfg.get("save_dir", "runs"))
    out_dir = save_root / experiment_name
    ensure_dir(out_dir)

    train_list = "data/splits/train.txt"
    val_list = "data/splits/val.txt"
    assert os.path.exists(train_list) and os.path.exists(val_list), \
        "未找到数据划分文件，请先运行 scripts/prepare_data.py 生成 train/val/test 列表"

    train_ds = KvasirSegDataset(dataset_dir, train_list, img_size=img_size, is_train=True)
    val_ds = KvasirSegDataset(dataset_dir, val_list, img_size=img_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    model = UNet(in_channels=3, num_classes=1, base_c=64, bilinear=True).to(device)

    criterion = build_loss(loss_name)
    optimizer = build_optimizer(optimizer_name, model.parameters(), lr, weight_decay)
    scheduler = build_scheduler(scheduler_name, optimizer, epochs)

    # 新：使用 torch.amp 的 GradScaler 接口
    scaler = _amp.GradScaler('cuda', enabled=mixed_precision and str(device).startswith("cuda"))

    best_dice = -1.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        if scheduler is not None:
            scheduler.step()

        if epoch % val_every == 0:
            val_metrics = validate(model, val_loader, criterion, device)
            print(f"Val: loss={val_metrics['loss']:.4f} dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f}")

            is_best = val_metrics["dice"] > best_dice
            if is_best:
                best_dice = val_metrics["dice"]
                no_improve = 0
            else:
                no_improve += 1

            ckpt_path = save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_dice": best_dice,
                "config": cfg,
            }, is_best=is_best, out_dir=str(out_dir))

            print(f"Checkpoint saved to {ckpt_path}")
            if no_improve >= early_patience:
                print("Early stopping.")
                break

    print(f"Best Val Dice: {best_dice:.4f}  (weights at {out_dir/'best.ckpt'})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
