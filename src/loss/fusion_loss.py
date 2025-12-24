import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionLoss(nn.Module):
    def forward(self, fused, gt):
        gt = gt.float()
        pos = gt.sum()
        neg = gt.numel() - pos
        beta = neg / (pos + neg + 1e-6)

        weight = torch.where(gt==1, beta, 1-beta)
        loss = F.binary_cross_entropy(fused, gt, weight=weight, reduction='mean')
        return loss
