from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class ContextAttackParams:
    alpha: torch.Tensor
    brightness: torch.Tensor
    blur: torch.Tensor
    noise: torch.Tensor
    occ_x: torch.Tensor
    occ_y: torch.Tensor
    occ_w: torch.Tensor
    occ_h: torch.Tensor
    occ_val: torch.Tensor
    persp: Optional[torch.Tensor] = None


def _gaussian_kernel1d(sigma: torch.Tensor, ksize: int, device) -> torch.Tensor:
    half = ksize // 2
    x = torch.arange(-half, half + 1, device=device).float()[None, :]
    sig = sigma.clamp(min=1e-4)
    g = torch.exp(-0.5 * (x / sig) ** 2)
    g = g / g.sum(dim=1, keepdim=True)
    return g


def gaussian_blur(img: torch.Tensor, sigma01: torch.Tensor, max_sigma: float = 2.5) -> torch.Tensor:
    B, C, H, W = img.shape
    device = img.device
    sigma = sigma01.clamp(0, 1) * max_sigma + 1e-4
    ksize = 9
    g1 = _gaussian_kernel1d(sigma, ksize, device=device)
    g1x = g1[:, None, None, :]
    g1y = g1[:, None, :, None]

    img_ = img.view(1, B * C, H, W)
    kx = g1x.repeat_interleave(C, dim=1).view(B * C, 1, 1, ksize)
    ky = g1y.repeat_interleave(C, dim=1).view(B * C, 1, ksize, 1)

    out = F.conv2d(img_, kx, padding=(0, ksize // 2), groups=B * C)
    out = F.conv2d(out, ky, padding=(ksize // 2, 0), groups=B * C)
    return out.view(B, C, H, W)


def apply_occlusion(img: torch.Tensor, p: ContextAttackParams) -> torch.Tensor:
    B, C, H, W = img.shape
    x0 = (p.occ_x.clamp(0, 1) * (W - 1)).long().view(B)
    y0 = (p.occ_y.clamp(0, 1) * (H - 1)).long().view(B)
    ww = (p.occ_w.clamp(0, 1) * W).long().clamp(min=8).view(B)
    hh = (p.occ_h.clamp(0, 1) * H).long().clamp(min=8).view(B)
    val = p.occ_val.clamp(0, 1).view(B, 1, 1, 1)

    out = img.clone()
    for i in range(B):
        xs = int(x0[i].item())
        ys = int(y0[i].item())
        xe = min(W, xs + int(ww[i].item()))
        ye = min(H, ys + int(hh[i].item()))
        out[i, :, ys:ye, xs:xe] = val[i]
    return out


def composite_on_background(qr_img: torch.Tensor, bg_img: Optional[torch.Tensor], alpha01: torch.Tensor) -> torch.Tensor:
    if bg_img is None:
        return qr_img
    a = alpha01.clamp(0, 1).view(-1, 1, 1, 1)
    return (1 - a) * qr_img + a * bg_img


def _rand_perspective_grid(B: int, H: int, W: int, strength01: torch.Tensor, device) -> torch.Tensor:
    max_shift = 0.12 * strength01.clamp(0, 1).view(B, 1)
    def jitter():
        return (torch.rand((B, 1), device=device) * 2 - 1) * max_shift

    tl = torch.cat([(-1 + jitter()), (-1 + jitter())], dim=1)
    tr = torch.cat([( 1 + jitter()), (-1 + jitter())], dim=1)
    br = torch.cat([( 1 + jitter()), ( 1 + jitter())], dim=1)
    bl = torch.cat([(-1 + jitter()), ( 1 + jitter())], dim=1)

    ys = torch.linspace(0, 1, H, device=device).view(1, H, 1, 1)
    xs = torch.linspace(0, 1, W, device=device).view(1, 1, W, 1)

    top = tl.view(B, 1, 1, 2) * (1 - xs) + tr.view(B, 1, 1, 2) * xs
    bot = bl.view(B, 1, 1, 2) * (1 - xs) + br.view(B, 1, 1, 2) * xs
    grid = top * (1 - ys) + bot * ys
    return grid


def mild_perspective(img: torch.Tensor, strength01: torch.Tensor) -> torch.Tensor:
    B, C, H, W = img.shape
    grid = _rand_perspective_grid(B, H, W, strength01, device=img.device)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def context_attack(clean_img: torch.Tensor, bg_img: Optional[torch.Tensor], params: ContextAttackParams, difficulty: float = 1.0) -> torch.Tensor:
    d = float(difficulty)
    x = composite_on_background(clean_img, bg_img, (params.alpha * d).clamp(0, 1))

    gain = 0.7 + 0.6 * params.brightness.clamp(0, 1)
    x = (x * gain.view(-1, 1, 1, 1)).clamp(0, 1)

    if params.persp is not None:
        x = mild_perspective(x, (params.persp * d).clamp(0, 1))

    x = gaussian_blur(x, (params.blur * d).clamp(0, 1))

    sigma = (params.noise.clamp(0, 1) * d * 0.08).view(-1, 1, 1, 1)
    x = (x + torch.randn_like(x) * sigma).clamp(0, 1)

    x = apply_occlusion(x, params)
    return x
