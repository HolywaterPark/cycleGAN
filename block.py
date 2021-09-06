import torch
import torch.nn as nn

class Residual_block(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=0):
        super(Residual_block, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=True),
                      nn.InstanceNorm2d(num_features=channels),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=True),
                      nn.InstanceNorm2d(num_features=channels) ]
        self.residual = nn.Sequential(*conv_block)
    def forward(self, x):
        return x + self.residual(x)
