import torch.nn.functional as F

def upsample_to_input(x, target):
    return F.interpolate(
        x,
        size=target.shape[2:],
        mode="bilinear",
        align_corners=False
    )
