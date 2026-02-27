# tsffm_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from resbranch_with_spatial import ResBranchSpatial
from tsffm_modules import FEBlock


class TSFFM(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, fle_dim=256):
        super().__init__()

        self.face_branch = ResBranchSpatial(pretrained=pretrained, num_classes=num_classes)
        self.body_branch = ResBranchSpatial(pretrained=pretrained, num_classes=num_classes)

        self.fe_block = FEBlock(in_channels=256, fle_dim=fle_dim)

        self.fused_head = nn.Linear(fle_dim, num_classes)

        self.decision_head = nn.Linear(num_classes * 3, num_classes)

    def forward(self, body_img, face_img, labels=None):
        logits_f, spatial_f = self.face_branch.forward_spatial_and_logits(face_img)
        logits_b, spatial_b = self.body_branch.forward_spatial_and_logits(body_img)

        fused_spat, fused_temp = self.fe_block(spatial_f, spatial_b)

        logits_fused = self.fused_head(fused_temp)

        concat_logits = torch.cat([logits_f, logits_b, logits_fused], dim=1)
        logits_final = self.decision_head(concat_logits)

        out = dict(
            logits_face=logits_f,
            logits_body=logits_b,
            logits_fused=logits_fused,
            logits_final=logits_final,
        )

        if labels is None:
            return out

        loss = (
            F.cross_entropy(logits_f, labels) +
            F.cross_entropy(logits_b, labels) +
            F.cross_entropy(logits_fused, labels) +
            F.cross_entropy(logits_final, labels)
        )

        return loss, out
