import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


class WaveletDownsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')

    def forward(self, x):
        yl, yh = self.dwt(x)
        yh0 = yh[0].reshape(yh[0].shape[0], yh[0].shape[1] * yh[0].shape[2], yh[0].shape[3], yh[0].shape[4])
        return torch.cat([yl, yh0], dim=1)


class DWBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.act1 = nn.ReLU(inplace=True)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pw(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

# Squeeze-Excite (tiny)
class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc1 = nn.Linear(c, c//r)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(c//r, c)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b,c,h,w = x.shape
        y = x.mean((2,3))            # [B, C]
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sig(y).view(b,c,1,1)
        return x * y


def count_params(model):
    return sum(p.numel() for p in model.parameters())

def build_model(num_classes, base_c=16, use_se=True, wavelets=2):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, base_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(base_c),
                nn.ReLU(inplace=True)
            )

            c = base_c

            self.wave1 = WaveletDownsample()
            c *= 4
            self.block1 = DWBlock(c)
            self.se1 = SE(c) if use_se else nn.Identity()

            if wavelets == 2:
                self.wave2 = WaveletDownsample()
                c *= 4
                self.block2 = DWBlock(c)
                self.se2 = SE(c) if use_se else nn.Identity()
            else:
                self.wave2 = None

            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(c, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.wave1(x)
            x = self.block1(x)
            x = self.se1(x)

            if self.wave2:
                x = self.wave2(x)
                x = self.block2(x)
                x = self.se2(x)

            x = self.pool(x).flatten(1)
            return self.fc(x)

    return Net()