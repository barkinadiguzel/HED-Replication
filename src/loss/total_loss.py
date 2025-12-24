import torch
import torch.nn as nn
from .side_loss import SideLoss        
from .fusion_loss import FusionLoss   

class TotalLoss(nn.Module):
    def __init__(self, alpha=None):
        super().__init__()
        self.side_loss = SideLoss()
        self.fusion_loss = FusionLoss()
        self.alpha = alpha  

    def forward(self, side_outputs, fused, gt):
        if self.alpha is None:
            self.alpha = [1.0] * len(side_outputs)
        side_losses = []
        for i, side in enumerate(side_outputs):
            l = self.side_loss(side, gt)
            side_losses.append(self.alpha[i] * l)
        total_side_loss = sum(side_losses)
        fusion_loss_val = self.fusion_loss(fused, gt)
        total_loss = total_side_loss + fusion_loss_val
        return total_loss, total_side_loss, fusion_loss_val
