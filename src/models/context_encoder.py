from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[0, 1, 0],
                          [1,-4, 1],
                          [0, 1, 0]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('lap_k', k)
        sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sx', sx)
        self.register_buffer('sy', sy)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        gray = (0.299*img[:,0:1] + 0.587*img[:,1:2] + 0.114*img[:,2:3]).clamp(0, 1)

        dark = (gray < 0.45).float()
        area_ratio = dark.mean(dim=(2,3))

        ys = torch.linspace(0, 1, H, device=img.device).view(1,1,H,1)
        xs = torch.linspace(0, 1, W, device=img.device).view(1,1,1,W)
        m = dark.sum(dim=(2,3), keepdim=True).clamp(min=1.0)
        cx = (dark * xs).sum(dim=(2,3), keepdim=True) / m
        cy = (dark * ys).sum(dim=(2,3), keepdim=True) / m
        cx = cx.view(B,1); cy = cy.view(B,1)

        lap = F.conv2d(gray, self.lap_k, padding=1)
        var_lap = lap.var(dim=(2,3))
        blur_score = torch.log1p(var_lap).clamp(0, 10) / 10.0

        contrast = gray.std(dim=(2,3)).clamp(0, 1) / 0.5

        gx = F.conv2d(gray, self.sx, padding=1)
        gy = F.conv2d(gray, self.sy, padding=1)
        g = torch.sqrt(gx*gx + gy*gy)
        bg_complex = (g > 0.25).float().mean(dim=(2,3))

        mid = ((gray >= 0.20) & (gray <= 0.85)).float().mean(dim=(2,3))

        feat = torch.cat([area_ratio, cx, cy, blur_score, contrast, bg_complex, mid], dim=1).clamp(0, 1)
        return feat
