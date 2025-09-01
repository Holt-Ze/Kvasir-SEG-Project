import torch

def dice_coef(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = (probs > 0.5).float()
    num = 2 * (probs * targets).sum(dim=(2,3)) + eps
    den = (probs.sum(dim=(2,3)) + targets.sum(dim=(2,3))) + eps
    return (num / den).mean().item()

def iou_score(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = (probs > 0.5).float()
    inter = (probs * targets).sum(dim=(2,3)) + eps
    union = (probs + targets - probs * targets).sum(dim=(2,3)) + eps
    return (inter / union).mean().item()
