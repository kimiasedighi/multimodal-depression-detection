# msn_body.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MSNBody(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, scales=[3, 5, 7]):
        super().__init__()

        self.branches = nn.ModuleList([
            MultiScaleBlock(in_channels, 32, k) for k in scales
        ])

        self.conv_fuse = nn.Conv2d(32 * len(scales), 64, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)

        self.head_n = nn.Linear(64, num_classes)
        self.head_p = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        x = torch.cat(outs, dim=1)
        x = F.relu(self.conv_fuse(x))
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)

        return self.head_n(x), self.head_p(x)
