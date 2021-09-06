import torch

def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor((0.5, 0.5, 0.5)) + torch.Tensor((0.5, 0.5, 0.5))
    x = x.transpose(1, 3)
    return x