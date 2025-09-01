import argparse
import os
import random
from pathlib import Path

def main(args):
    data_dir = Path(args.data_dir)
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    assert img_dir.exists() and mask_dir.exists(), "请确认 data_dir 下存在 images/ 与 masks/ 目录"

    # 允许 jpg/png 混合
    names = []
    for p in img_dir.iterdir():
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            names.append(p.name)

    names.sort()
    random.seed(args.seed)
    random.shuffle(names)

    n = len(names)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    test = names[:n_test]
    val = names[n_test:n_test+n_val]
    train = names[n_test+n_val:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(lst, fname):
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            for x in lst:
                f.write(x + "\n")

    dump(train, "train.txt")
    dump(val, "val.txt")
    dump(test, "test.txt")

    print(f"Total: {n}, Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"Splits saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Kvasir-SEG 根目录（含 images/ masks/）")
    parser.add_argument("--out_dir", type=str, default="data/splits", help="划分文件保存目录")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
