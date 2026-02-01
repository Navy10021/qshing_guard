from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..train.modeling_qr import build_qr_model
from ..train.modeling_fusion import build_fusion_model


@dataclass
class DetectorConfig:
    mode: str = "qr"          # "qr" or "fusion"
    fusion_mode: str = "gated"
    url_dim: int = 0
    lex_dim: int = 0
    use_context: bool = True
    ctx_dim: int = 7


class QuishingDetector(nn.Module):
    """Signature-normalized detector wrapper for QR-only and Fusion models.

    Matches your modeling files:
      - QRContextNet.forward(x_img, x_ctx=None)
      - Fusion*.forward(x_img, x_url, x_lex, x_ctx=None)

    This wrapper guarantees that in fusion-mode, x_ctx is always a tensor
    (zeros if None), so call-sites don't need to branch.
    """

    def __init__(self, cfg: DetectorConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.mode == "qr":
            self.net = build_qr_model(use_context=cfg.use_context, ctx_dim=cfg.ctx_dim)
        elif cfg.mode == "fusion":
            assert cfg.url_dim > 0 and cfg.lex_dim > 0
            self.net = build_fusion_model(
                fusion_mode=cfg.fusion_mode,
                url_dim=cfg.url_dim,
                lex_dim=cfg.lex_dim,
                ctx_dim=cfg.ctx_dim,
            )
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")

    def forward(
        self,
        img: torch.Tensor,
        url_vec: Optional[torch.Tensor] = None,
        lex: Optional[torch.Tensor] = None,
        ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.cfg.mode == "qr":
            if self.cfg.use_context:
                # QRContextNet accepts None, but we normalize to zeros for stability
                if ctx is None:
                    ctx = torch.zeros((img.size(0), self.cfg.ctx_dim), device=img.device, dtype=torch.float32)
                if ctx.dim() == 1:
                    ctx = ctx.unsqueeze(0).expand(img.size(0), -1)
                return self.net(img, ctx)
            return self.net(img)

        # fusion
        assert url_vec is not None and lex is not None, "fusion requires url_vec and lex"
        if ctx is None:
            ctx = torch.zeros((img.size(0), self.cfg.ctx_dim), device=img.device, dtype=torch.float32)
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0).expand(img.size(0), -1)
        return self.net(img, url_vec, lex, ctx)
