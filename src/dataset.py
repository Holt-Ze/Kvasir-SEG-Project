import os
from typing import Tuple, Dict
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .transforms import get_train_augs, get_val_augs

class KvasirSegDataset(Dataset):
    def __init__(self, root: str, split_list: str, img_size: int = 256, is_train: bool = True):
        """
        root: Kvasir-SEG 根目录（包含 images/ 与 masks/）
        split_list: 文本文件路径，每行是文件名（无扩展名或带扩展名都可）
        img_size: 统一缩放尺寸
        is_train: 训练或验证/测试，决定数据增强
        """
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.items = []

        with open(split_list, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                # 允许既支持带扩展名也支持不带扩展名的列表
                if os.path.splitext(name)[1] == "":
                    # 默认 jpg/png 都尝试
                    if os.path.exists(os.path.join(self.img_dir, name + ".jpg")):
                        name = name + ".jpg"
                    elif os.path.exists(os.path.join(self.img_dir, name + ".png")):
                        name = name + ".png"
                self.items.append(name)

        self.img_size = img_size
        self.is_train = is_train
        self.train_tfms = get_train_augs(img_size)
        self.val_tfms = get_val_augs(img_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_name = self.items[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # mask 名与 image 名一致（扩展名可能不同，尝试替换）
        base = os.path.splitext(img_name)[0]
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            cand = os.path.join(self.mask_dir, base + ext)
            if os.path.exists(cand):
                mask_path = cand
                break
        else:
            raise FileNotFoundError(f"Mask not found for {img_name}")

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 将 mask 归一化到 {0,1}
        mask = (mask > 0).astype(np.uint8)

        if self.is_train:
            aug = self.train_tfms(image=image, mask=mask)
        else:
            aug = self.val_tfms(image=image, mask=mask)

        image = aug["image"]
        mask = aug["mask"].astype(np.float32)

        # CHW
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "name": os.path.splitext(img_name)[0]
        }
