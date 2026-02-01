from __future__ import annotations
import torch
import torch.nn as nn
from ..utils.context_ops import ContextAttackParams


class ContextAttacker(nn.Module):
    def __init__(self, style_dim: int = 16):
        super().__init__()
        import torchvision.models as models
        backbone = models.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.img_proj = nn.Sequential(nn.Flatten(), nn.Linear(512, 256), nn.ReLU())
        self.style = nn.Sequential(nn.Linear(style_dim, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(256+64, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(self, clean_img: torch.Tensor, style_code: torch.Tensor) -> ContextAttackParams:
        z = self.img_proj(self.cnn(clean_img))
        s = self.style(style_code)
        h = torch.sigmoid(self.head(torch.cat([z, s], dim=1)))
        return ContextAttackParams(
            alpha=h[:,0:1],
            brightness=h[:,1:2],
            blur=h[:,2:3],
            noise=h[:,3:4],
            occ_x=h[:,4:5],
            occ_y=h[:,5:6],
            occ_w=h[:,6:7],
            occ_h=h[:,7:8],
            occ_val=h[:,8:9],
            persp=h[:,9:10],
        )
