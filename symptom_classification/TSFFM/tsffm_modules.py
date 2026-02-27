# tsffm_modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Channel Attention (CBAM)
# -------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // reduction, in_planes, bias=False),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        avg_out = x.mean(dim=(2, 3))              # [B, C]
        max_out = torch.amax(x, dim=(2, 3))       # [B, C]
        out = self.mlp(avg_out) + self.mlp(max_out)
        scale = torch.sigmoid(out).view(B, C, 1, 1)
        return x * scale


# -------------------------
# Spatial Attention (CBAM)
# -------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)      # [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(x_cat))
        return x * attn


# -------------------------
# CBAM block
# -------------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))


# -------------------------
# Temporal Feature Extractor (FLE)
# -------------------------
class FLE(nn.Module):
    """
    FLE on single-frame input (T=1). Later easily extended to real sequences.
    x: [B, 1, C, H, W]
    """
    def __init__(self, in_channels, hidden_dim=256, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [B, 1, C, H, W]
        B, T, C, H, W = x.shape
        x = x.mean(dim=(3, 4))     # [B, T, C]
        x = x.mean(dim=1)          # [B, C]
        return self.mlp(x)         # [B, out_dim]


# -------------------------
# FEBlock for TSFFM
# -------------------------
class FEBlock(nn.Module):
    def __init__(self, in_channels, fle_dim=256):
        super().__init__()
        self.cbam_face = CBAM(in_channels)
        self.cbam_body = CBAM(in_channels)

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.keep_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.fle = FLE(in_channels, hidden_dim=fle_dim, out_dim=fle_dim)

        self.temporal_merge = nn.Linear(fle_dim * 2, fle_dim)

    def forward(self, face_feat, body_feat, prev_spatial=None, prev_temporal=None):
        f = self.cbam_face(face_feat)
        b = self.cbam_body(body_feat)

        fused = self.fuse_conv(f + b)

        if prev_spatial is not None:
            fused = fused + self.keep_conv(prev_spatial)

        fle_input = fused.unsqueeze(1)  # [B,1,C,H,W]
        t_feat = self.fle(fle_input)

        if prev_temporal is not None:
            t_feat = torch.relu(self.temporal_merge(torch.cat([prev_temporal, t_feat], dim=1)))

        return fused, t_feat
