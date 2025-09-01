import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import amp as _amp  # 新：使用新的 AMP 接口
from .metrics import dice_coef, iou_score

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            # 新：torch.amp.autocast
            with _amp.autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dices, ious = [], []
    pbar = tqdm(loader, desc="Val  ", leave=False)
    for batch in pbar:
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, masks)
        running_loss += loss.item() * imgs.size(0)

        dices.append(dice_coef(logits, masks))
        ious.append(iou_score(logits, masks))

    avg_loss = running_loss / len(loader.dataset)
    avg_dice = sum(dices) / len(dices) if dices else 0.0
    avg_iou = sum(ious) / len(ious) if ious else 0.0
    return {"loss": avg_loss, "dice": avg_dice, "iou": avg_iou}
