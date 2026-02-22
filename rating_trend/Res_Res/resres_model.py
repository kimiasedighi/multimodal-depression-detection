# resres_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SmallCNN(nn.Module):
    """
    Torch-only CNN backbone (no torchvision).
    Input: [B, 3, 224, 224]
    Output: feature vector
    """
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.out_dim = 256

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)


class ResBranch(nn.Module):
    def __init__(self, num_classes=3, dropout=0.2):
        super().__init__()
        self.backbone = SmallCNN(out_dim=256)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.dropout(feats)
        logits = self.head(feats)
        return logits, feats


class ResRes18DecisionFusion(nn.Module):
    def __init__(
        self,
        num_classes=3,
        input_type="B",
        fusion="avg_logit",
        branch_loss_weight=1.0,
        fusion_loss_weight=1.0,
    ):
        super().__init__()

        self.input_type = input_type.upper()
        self.fusion = fusion
        self.branch_loss_weight = branch_loss_weight
        self.fusion_loss_weight = fusion_loss_weight

        self.has_body = "B" in self.input_type
        self.has_face = "F" in self.input_type

        self.body = ResBranch(num_classes) if self.has_body else None
        self.face = ResBranch(num_classes) if self.has_face else None

    def fuse_logits(self, logits_list):
        if len(logits_list) == 1:
            return logits_list[0]

        if self.fusion == "avg_prob":
            probs = [F.softmax(l, dim=1) for l in logits_list]
            mean_prob = torch.stack(probs).mean(0)
            return torch.log(mean_prob + 1e-8)
        else:
            return torch.stack(logits_list).mean(0)

    def forward(self, body_img=None, face_img=None, labels=None):
        outputs = {}
        logits_list = []

        if self.body is not None and body_img is not None:
            lb, _ = self.body(body_img)
            outputs["logits_body"] = lb
            logits_list.append(lb)

        if self.face is not None and face_img is not None:
            lf, _ = self.face(face_img)
            outputs["logits_face"] = lf
            logits_list.append(lf)

        outputs["logits_fused"] = self.fuse_logits(logits_list)

        if labels is None:
            return outputs

        loss = 0.0
        if "logits_body" in outputs:
            loss += self.branch_loss_weight * F.cross_entropy(outputs["logits_body"], labels)
        if "logits_face" in outputs:
            loss += self.branch_loss_weight * F.cross_entropy(outputs["logits_face"], labels)

        if self.fusion_loss_weight > 0:
            loss += self.fusion_loss_weight * F.cross_entropy(outputs["logits_fused"], labels)

        return loss, outputs
