# resbranch_with_spatial.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class ResBranchSpatial(nn.Module):
    def __init__(self, pretrained=False, num_classes=2, dropout=0.2):
        super().__init__()

        self.backbone = tvm.resnet18(pretrained=pretrained)

        self.feat_channels = 256  # layer3 output
        feat_dim = self.backbone.fc.in_features

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward_spatial_and_logits(self, x):
        b = self.backbone

        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)

        x = b.layer1(x)
        x = b.layer2(x)
        spatial = b.layer3(x)       # [B,256,H,W]

        x = b.layer4(spatial)
        x = b.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.head(x)

        return logits, spatial
