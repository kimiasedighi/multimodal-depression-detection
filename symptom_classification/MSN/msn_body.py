# msn_body.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(pad, 0), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class MSNBody(nn.Module):
    def __init__(self, in_channels=3, num_joints=11, num_classes=2, scales=[3,5,7]):
        super().__init__()
        self.branches = nn.ModuleList([
            MultiScaleBlock(in_channels, 32, k) for k in scales
        ])
        total = 32 * len(scales)
        self.conv_fuse = nn.Conv2d(total, 64, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: [B, C, T, J]
        outs = [branch(x) for branch in self.branches]
        x = torch.cat(outs, dim=1)
        x = F.relu(self.conv_fuse(x))
        x = self.pool(x)  # output [B, 64,1,1]
        x = x.view(x.size(0), -1)
        return self.fc(x)
