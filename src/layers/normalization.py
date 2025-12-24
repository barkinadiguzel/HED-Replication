import torch.nn as nn

def get_norm(channels, use_bn=False):
    if use_bn:
        return nn.BatchNorm2d(channels)
    return nn.Identity()
