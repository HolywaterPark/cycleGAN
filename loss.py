import torch.nn as nn
import torch

def real_target_loss(x):
    target = torch.cuda.FloatTensor(x.shape[0], 1).fill_(1.0)
    return nn.MSELoss(x, target)

def fake_target_loss(x):
    target = torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.0)
    return nn.MSELoss(x, target)

def identity_loss(x, y):
    return nn.L1Loss(x, y)

def cycle_loss(x, y):
    return nn.L1Loss(x, y)

