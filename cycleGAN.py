import torch.nn as nn
from block import Residual_block
import torch.nn.functional as F

class Gen(nn.Module):
    def __init__(self, channels=3, n_residual_blocks=9):
        super(Gen, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(channels, 64, kernel_size=7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2

        for _ in range(n_residual_blocks):
            model += [Residual_block(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, channels, kernel_size=7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Dis(nn.Module):
    def __init__(self, channels=3):
        super(Dis, self).__init__()
        model = [nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(512, 1, kernel_size=4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)