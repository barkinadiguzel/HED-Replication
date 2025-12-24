import torch
import torch.nn as nn

from src.backbone.vgg_blocks import VGGBlock
from src.side_outputs.side_head import SideHead
from src.fusion.fusion_head import FusionHead

class HEDNet(nn.Module):
    def __init__(self, num_sides=5):
        super().__init__()

        self.stage1 = VGGBlock(3, 64, 2)
        self.stage2 = VGGBlock(64, 128, 2)
        self.stage3 = VGGBlock(128, 256, 3)
        self.stage4 = VGGBlock(256, 512, 3)
        self.stage5 = VGGBlock(512, 512, 3)

        self.side_heads = nn.ModuleList([
            SideHead(64),
            SideHead(128),
            SideHead(256),
            SideHead(512),
            SideHead(512),
        ])

        self.fusion = FusionHead(num_sides)

    def forward(self, x):
        side_outputs = []

        f1, x = self.stage1(x)
        side_outputs.append(self.side_heads[0](f1))

        f2, x = self.stage2(x)
        side_outputs.append(self.side_heads[1](f2))

        f3, x = self.stage3(x)
        side_outputs.append(self.side_heads[2](f3))

        f4, x = self.stage4(x)
        side_outputs.append(self.side_heads[3](f4))

        f5, _ = self.stage5(x)
        side_outputs.append(self.side_heads[4](f5))

        fused = self.fusion(side_outputs)

        return side_outputs, fused
