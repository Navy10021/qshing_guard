# src/train/modeling_qr.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class QRNet(nn.Module):
    """Plain QR image classifier (ResNet18)."""

    def __init__(self, num_classes: int = 2, weights: Optional[str] = None):
        super().__init__()
        import torchvision.models as models

        if weights is None:
            m = models.resnet18(weights=None)
        else:
            m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.backbone = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class QRContextNet(nn.Module):
    """QR image + context feature classifier.

    forward(x_img, x_ctx=None) where x_ctx shape [B,7].
    If x_ctx is None, uses zeros -> inference compatibility.
    """

    def __init__(self, ctx_dim: int = 7, num_classes: int = 2, weights: Optional[str] = None):
        super().__init__()
        import torchvision.models as models

        if weights is None:
            backbone = models.resnet18(weights=None)
        else:
            backbone = models.resnet18(weights=None)

        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # [B,512,1,1]
        self.img_proj = nn.Sequential(nn.Flatten(), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2))
        self.ctx_mlp = nn.Sequential(nn.Linear(ctx_dim, 64), nn.ReLU(), nn.Dropout(0.1))

        self.head = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_img: torch.Tensor, x_ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_img = self.img_proj(self.cnn(x_img))
        if x_ctx is None:
            x_ctx = torch.zeros((x_img.shape[0], 7), device=x_img.device, dtype=z_img.dtype)
        z_ctx = self.ctx_mlp(x_ctx)
        z = torch.cat([z_img, z_ctx], dim=1)
        return self.head(z)


def build_qr_model(
    num_classes: int = 2,
    weights: Optional[str] = None,
    use_context: bool = True,
    ctx_dim: int = 7,
) -> nn.Module:
    """
    QR 이미지 분류 모델 빌더.
    - use_context=True: QRContextNet (image + context features)
    - use_context=False: plain ResNet18 (QRNet)
    """
    if use_context:
        return QRContextNet(ctx_dim=ctx_dim, num_classes=num_classes, weights=weights)
    return QRNet(num_classes=num_classes, weights=weights)


def load_model_state(model: nn.Module, ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    """
    체크포인트 포맷 다양성 흡수:
    - train_*.py: {"model": state_dict, "epoch": ...}
    - 다른 스크립트/실험: {"state_dict": ...} 혹은 state_dict 단독
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            model.load_state_dict(ckpt["model"])
            return ckpt
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            model.load_state_dict(ckpt["state_dict"])
            return ckpt

        # ckpt 자체가 state_dict인 케이스
        try:
            model.load_state_dict(ckpt)  # type: ignore[arg-type]
            return {"raw": ckpt}
        except Exception as e:
            raise RuntimeError(
                f"Unrecognized checkpoint format: {ckpt_path}, keys={list(ckpt.keys())}"
            ) from e

    raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")
