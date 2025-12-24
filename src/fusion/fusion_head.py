import torch
import torch.nn as nn

class FusionHead(nn.Module):
    def __init__(self, num_sides):
        super().__init__()
        self.fuse = nn.Conv2d(
            in_channels=num_sides,
            out_channels=1,
            kernel_size=1,
            bias=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, side_outputs):
        x = torch.cat(side_outputs, dim=1)  # [B, M, H, W]
        x = self.fuse(x)                    # weighted sum (h_m)
        return self.sigmoid(x)
