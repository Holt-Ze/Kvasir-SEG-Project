import albumentations as A
from albumentations.pytorch import ToTensorV2  # 可留可删，这里未直接使用

def get_train_augs(size: int):
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        # 去掉 value / mask_value，避免版本不兼容警告
        A.PadIfNeeded(min_height=size, min_width=size),
        A.RandomCrop(height=size, width=size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.GaussNoise(p=0.15),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def get_val_augs(size: int):
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        # 同样去掉 value / mask_value
        A.PadIfNeeded(min_height=size, min_width=size),
        A.CenterCrop(height=size, width=size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
