import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from src.model import UNet
from src.transforms import get_val_augs

def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    return state_dict, ckpt.get("config", None)

def infer_on_image(model, img_path, size, device, threshold=0.5):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    aug = get_val_augs(size=size)
    auged = aug(image=image_rgb, mask=np.zeros(image.shape[:2], dtype=np.uint8))
    img = auged["image"]
    img = np.transpose(img, (2,0,1)).astype(np.float32)
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0,0].cpu().numpy()
    pred = (prob > threshold).astype(np.uint8) * 255

    return pred

def main(args):
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    state_dict, cfg = load_checkpoint(args.checkpoint, device)
    size = args.size if args.size is not None else (cfg.get("img_size", 256) if cfg else 256)

    model = UNet(in_channels=3, num_classes=1, base_c=64, bilinear=True).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = []
    if os.path.isdir(args.input):
        for name in os.listdir(args.input):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                inputs.append(os.path.join(args.input, name))
    else:
        inputs.append(args.input)

    for p in inputs:
        pred = infer_on_image(model, p, size, device, threshold=args.threshold)
        base = os.path.splitext(os.path.basename(p))[0]
        out_path = out_dir / f"{base}_pred.png"
        cv2.imwrite(str(out_path), pred)
        print("Saved:", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="图片路径或目录")
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
