# resbranch_with_spatial.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNNSpatial(nn.Module):
    """
    Torch-only backbone producing:
      - spatial feature map: [B, 256, H, W]
      - pooled feature: [B, 256]
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 112
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 56
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 28
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 14
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        spatial = self.conv4(x)        # [B,256,14,14]
        feat = self.pool(spatial).view(x.size(0), -1)  # [B,256]
        return feat, spatial


class ResBranchSpatial(nn.Module):
    def __init__(self, pretrained=False, num_classes=3, dropout=0.2):
        super().__init__()
        self.backbone = SmallCNNSpatial()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(256, num_classes)

    def forward_spatial_and_logits(self, x):
        feat, spatial = self.backbone(x)
        feat = self.dropout(feat)
        logits = self.head(feat)
        return logits, spatial
        