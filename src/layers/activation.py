import torch.nn as nn

def get_activation(name: str):
    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    if name.lower() == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")
