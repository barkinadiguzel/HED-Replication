import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_convs):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(
                in_ch if i == 0 else out_ch,
                out_ch,
                kernel_size=3,
                padding=1
            ))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.block(x)
        p = self.pool(x)
        return x, p
