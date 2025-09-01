import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        # logits: (N,1,H,W), targets: (N,1,H,W)
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2,3)) + self.eps
        den = (probs.pow(2) + targets.pow(2)).sum(dim=(2,3)) + self.eps
        loss = 1 - (num / den)
        return loss.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.w = bce_weight

    def forward(self, logits, targets):
        return self.w * self.bce(logits, targets) + (1 - self.w) * self.dice(logits, targets)
