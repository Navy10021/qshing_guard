# src/train/modeling_fusion.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class FusionNet(nn.Module):
    """Concat fusion: [img_emb, url_emb, lex_emb, ctx_emb] -> head."""

    def __init__(self, url_dim: int, lex_dim: int, ctx_dim: int = 7, num_classes: int = 2):
        super().__init__()
        import torchvision.models as models

        backbone = models.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # [B,512,1,1]
        self.img_proj = nn.Sequential(nn.Flatten(), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2))

        self.url_mlp = nn.Sequential(nn.Linear(url_dim, 256), nn.ReLU(), nn.Dropout(0.2))
        self.lex_mlp = nn.Sequential(nn.Linear(lex_dim, 64), nn.ReLU(), nn.Dropout(0.1))
        self.ctx_mlp = nn.Sequential(nn.Linear(ctx_dim, 64), nn.ReLU(), nn.Dropout(0.1))

        self.head = nn.Sequential(
            nn.Linear(256 + 256 + 64 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_img, x_url, x_lex, x_ctx: Optional[torch.Tensor] = None):
        z_img = self.img_proj(self.cnn(x_img))
        z_url = self.url_mlp(x_url)
        z_lex = self.lex_mlp(x_lex)
        if x_ctx is None:
            x_ctx = torch.zeros((x_img.shape[0], 7), device=x_img.device, dtype=z_img.dtype)
        z_ctx = self.ctx_mlp(x_ctx)
        z = torch.cat([z_img, z_url, z_lex, z_ctx], dim=1)
        return self.head(z)


class FusionGatedNet(nn.Module):
    """Gated fusion (MoE style) over 4 modalities: img/url/lex/ctx."""

    def __init__(self, url_dim: int, lex_dim: int, ctx_dim: int = 7, num_classes: int = 2):
        super().__init__()
        import torchvision.models as models

        backbone = models.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # [B,512,1,1]
        self.img_proj = nn.Sequential(nn.Flatten(), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2))
        self.url_proj = nn.Sequential(nn.Linear(url_dim, 256), nn.ReLU(), nn.Dropout(0.2))
        self.lex_proj = nn.Sequential(nn.Linear(lex_dim, 256), nn.ReLU(), nn.Dropout(0.2))
        self.ctx_proj = nn.Sequential(nn.Linear(ctx_dim, 256), nn.ReLU(), nn.Dropout(0.2))

        self.gate = nn.Sequential(
            nn.Linear(256 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4),
        )

        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_img, x_url, x_lex, x_ctx: Optional[torch.Tensor] = None):
        z_img = self.img_proj(self.cnn(x_img))
        z_url = self.url_proj(x_url)
        z_lex = self.lex_proj(x_lex)
        if x_ctx is None:
            x_ctx = torch.zeros((x_img.shape[0], 7), device=x_img.device, dtype=z_img.dtype)
        z_ctx = self.ctx_proj(x_ctx)

        g = torch.softmax(self.gate(torch.cat([z_img, z_url, z_lex, z_ctx], dim=1)), dim=1)  # [B,4]
        z = g[:, 0:1] * z_img + g[:, 1:2] * z_url + g[:, 2:3] * z_lex + g[:, 3:4] * z_ctx
        return self.head(z)


def build_fusion_model(
    fusion_mode: str,
    url_dim: int,
    lex_dim: int,
    ctx_dim: int = 7,
    num_classes: int = 2,
) -> nn.Module:
    fusion_mode = (fusion_mode or "gated").lower()
    if fusion_mode == "concat":
        return FusionNet(url_dim=url_dim, lex_dim=lex_dim, ctx_dim=ctx_dim, num_classes=num_classes)
    return FusionGatedNet(url_dim=url_dim, lex_dim=lex_dim, ctx_dim=ctx_dim, num_classes=num_classes)
