# resres_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    import torchvision.models as tvm
except Exception:
    tvm = None

try:
    from torchvision.models import ResNet18_Weights
except Exception:
    ResNet18_Weights = None


def _build_resnet18(pretrained: bool = False) -> nn.Module:
    """
    Return a ResNet18 backbone with the final FC removed (Identity).
    Handles both old torchvision API (pretrained=True/False) and new (weights=...).
    """
    if tvm is None:
        raise ImportError("torchvision is required for ResBranch (import torchvision.models failed).")

    if ResNet18_Weights is not None:
        # Newer API: weights argument
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.resnet18(weights=weights)
    else:
        # Older API (your case with torchvision==0.11.2)
        backbone = tvm.resnet18(pretrained=pretrained)

    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    backbone._out_features = feat_dim  # convenience
    return backbone


class ResBranch(nn.Module):
    """
    One ResNet branch (e.g., face OR body) with its own classifier head.
    Input:  [B, 3, H, W]
    Output: logits: [B, num_classes], feats: [B, feat_dim]
    """
    def __init__(self, arch: str = "resnet18", pretrained: bool = False,
                 num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        if arch.lower() != "resnet18":
            raise ValueError(f"Unsupported arch: {arch}. Only 'resnet18' is implemented.")

        self.backbone = _build_resnet18(pretrained=pretrained)
        feat_dim = getattr(self.backbone, "_out_features", 512)
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)       # [B, feat_dim]
        feats = self.dropout(feats)
        logits = self.head(feats)      # [B, num_classes]
        return logits, feats


class ResRes18DecisionFusion(nn.Module):
    """
    Two-branch ResNet18 with decision-level fusion (late fusion).

    input_type:
      - "B"   -> body only (single branch)
      - "F"   -> face only  (single branch)
      - "F+B" -> two branches; fuse decisions

    fusion:
      - "avg_logit" (default): average branch logits then apply CE
      - "avg_prob":  average softmax probabilities; returns log-probs for fused output (use NLL)

    Training loss = (branch CE losses) * branch_loss_weight
                    + (fused loss)     * fusion_loss_weight
    """
    def __init__(self,
                 num_classes: int = 2,
                 input_type: str = "B",
                 fusion: str = "avg_logit",
                 pretrained: bool = False,
                 dropout: float = 0.2,
                 branch_loss_weight: float = 1.0,
                 fusion_loss_weight: float = 1.0):
        super().__init__()

        self.input_type = input_type.upper()
        self.fusion = fusion
        self.branch_loss_weight = float(branch_loss_weight)
        self.fusion_loss_weight = float(fusion_loss_weight)

        self.has_face = "F" in self.input_type
        self.has_body = "B" in self.input_type
        if not (self.has_face or self.has_body):
            raise ValueError("input_type must include 'F', 'B', or both (e.g., 'F+B').")

        # Build branches as requested
        self.face = ResBranch("resnet18", pretrained=pretrained, num_classes=num_classes, dropout=dropout) if self.has_face else None
        self.body = ResBranch("resnet18", pretrained=pretrained, num_classes=num_classes, dropout=dropout) if self.has_body else None

        
        # ----------------------------------------------------
        # FREEZE BACKBONE LAYERS (Step A: Warm-up training)
        # ----------------------------------------------------
        if self.has_body:
            for p in self.body.backbone.parameters():
                p.requires_grad = False

        if self.has_face:
            for p in self.face.backbone.parameters():
                p.requires_grad = False

                
    def fuse_logits(self, logits_list):
        """
        Fuse a list of [B, C] logits tensors into a single [B, C] tensor.
        If fusion == 'avg_prob', returns log-probabilities (for NLLLoss).
        """
        if len(logits_list) == 1:
            # Single-stream case; no fusion needed
            return logits_list[0]

        if self.fusion == "avg_prob":
            # Average probabilities then return log-probs for stability with NLLLoss
            probs = [F.softmax(l, dim=1) for l in logits_list]
            mean_prob = torch.stack(probs, dim=0).mean(dim=0)  # [B, C]
            return torch.log(mean_prob + 1e-8)                 # log-prob
        else:
            # Default: average logits (works naturally with CrossEntropyLoss)
            return torch.stack(logits_list, dim=0).mean(dim=0)

    def forward(self,
                body_img: Optional[torch.Tensor] = None,
                face_img: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        """
        If labels is None:
            returns dict of logits: {'logits_body', 'logits_face', 'logits_fused'}
        Else:
            returns (loss, same_dict)
        """
        outputs = {}
        logits_list = []

        if self.has_body and body_img is not None:
            logits_b, _ = self.body(body_img)
            outputs["logits_body"] = logits_b
            logits_list.append(logits_b)

        if self.has_face and face_img is not None:
            logits_f, _ = self.face(face_img)
            outputs["logits_face"] = logits_f
            logits_list.append(logits_f)

        if not logits_list:
            raise ValueError("No inputs provided to forward (got None for required streams).")

        outputs["logits_fused"] = self.fuse_logits(logits_list)

        # Inference-only path
        if labels is None:
            return outputs

        # Training loss: sum branch losses + fused loss
        total_loss = 0.0
        if "logits_body" in outputs:
            total_loss = total_loss + self.branch_loss_weight * F.cross_entropy(outputs["logits_body"], labels)
        if "logits_face" in outputs:
            total_loss = total_loss + self.branch_loss_weight * F.cross_entropy(outputs["logits_face"], labels)

        if self.fusion_loss_weight > 0:
            if self.fusion == "avg_prob":
                # outputs["logits_fused"] is log-probabilities
                total_loss = total_loss + self.fusion_loss_weight * F.nll_loss(outputs["logits_fused"], labels)
            else:
                total_loss = total_loss + self.fusion_loss_weight * F.cross_entropy(outputs["logits_fused"], labels)

        return total_loss, outputs
