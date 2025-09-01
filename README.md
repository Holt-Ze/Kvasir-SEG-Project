
# U-Net 分割 (Kvasir-SEG)

一个用 PyTorch 复现 U-Net 的小而全项目，面向 Kvasir-SEG 息肉分割任务。

## 快速开始
```bash
# 1) 安装依赖 (建议 Python ≥ 3.9，CUDA 环境可选)
pip install -r requirements.txt

# 2) 生成数据集划分文件 (按比例随机切分)
python scripts/prepare_data.py --data_dir /path/to/Kvasir-SEG --val_ratio 0.1 --test_ratio 0.1

# 3) 训练
python -m scripts.train --config config.yaml    

# 4) 预测（对单张图片或文件夹）
python -m scripts.predict --checkpoint runs/exp1/best.ckpt --input D:\Kvasir-SEG-Project\data\kvasir-seg\images --output outputs
```
> 说明：Kvasir-SEG 官方解压后包含 `images/` 与 `masks/` 两个文件夹（以及一个 JSON 元数据文件，可忽略）。

## 目录结构
```
unet_kvasir_seg
├─ config.yaml
├─ requirements.txt
├─ README.md
├─ data/
│  └─ splits/                 # 自动生成的 train/val/test 列表
├─ runs/                      # 训练日志与权重
├─ outputs/                   # 推理与可视化输出
├─ src/
│  ├─ __init__.py
│  ├─ dataset.py              # 数据集与 DataLoader
│  ├─ transforms.py           # Albumentations 数据增强
│  ├─ model.py                # U-Net 模型
│  ├─ loss.py                 # Dice、BCE+Dice 等
│  ├─ metrics.py              # Dice/IoU 指标
│  ├─ utils.py                # 通用工具（保存/日志/种子等）
│  └─ engine.py               # 训练/验证循环
└─ scripts/
   ├─ prepare_data.py         # 划分训练/验证/测试
   ├─ train.py                # 启动训练
   └─ predict.py              # 推理与批量可视化
```
