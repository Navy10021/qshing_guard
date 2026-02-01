# src/qr/augmentations.py
from __future__ import annotations

"""
Real-world QR augmentation utilities.

This module supports:
1) effect-only: perspective/blur/noise/jpeg/occlusion on the QR image
2) context-aware: composite QR onto an *unrelated* background, then apply camera effects

It also exposes helper APIs expected by src/qr/augment_qr.py:
- ContextConfig
- scan_backgrounds()
- augment_qr_contextual()

NOTE:
- augment_qr_realistic() already supports on-the-fly background compositing when
  cfg.background_dir has images and cfg.context_mode allows it.
- augment_qr_contextual() is used by offline augmentation script to explicitly composite
  with a chosen background image and optionally return context features + bbox.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# -------------------------
# Configs
# -------------------------
@dataclass
class AugmentConfig:
    # probabilistic switches
    p_perspective: float = 0.85
    p_blur: float = 0.70
    p_brightness: float = 0.55
    p_noise: float = 0.65
    p_jpeg: float = 0.55
    p_occlusion: float = 0.60

    # effect strengths (min,max)
    perspective_strength: Tuple[float, float] = (0.05, 0.14)
    blur_sigma: Tuple[float, float] = (0.6, 2.0)
    brightness_gain: Tuple[float, float] = (0.75, 1.25)
    noise_sigma: Tuple[float, float] = (2.0, 18.0)
    jpeg_q: Tuple[int, int] = (18, 92)
    occlusion_frac: Tuple[float, float] = (0.06, 0.22)

    # output + background compositing
    output_size: int = 512
    background_dir: str = "assets/backgrounds"

    # context compositing control
    # - "none": never composite; only apply camera effects on QR image itself
    # - "mix": composite with probability context_prob
    # - "always": always composite
    context_mode: str = "mix"
    context_prob: float = 0.5

    # composite parameters
    qr_scale_range: Tuple[float, float] = (0.32, 0.62)  # relative to output_size
    qr_margin_px: int = 20  # keep QR away from edges


@dataclass
class ContextConfig:
    """
    Minimal context compositing config expected by src/qr/augment_qr.py.

    p_context:
      - used by offline augmenter as probability of applying context composition
        (augment_qr.py handles mix/always/none by setting p_context)
    output_size:
      - if >0, produce square output_size x output_size
      - if 0, will use min(h,w) of background crop (still square)
    """
    p_context: float = 0.75
    output_size: int = 512
    qr_scale_range: Tuple[float, float] = (0.32, 0.62)
    qr_margin_px: int = 20
    feather: int = 6  # soft edge blending


# -------------------------
# Background scanning API (for offline augmentation)
# -------------------------
def scan_backgrounds(background_dir: str) -> List[Path]:
    root = Path(background_dir)
    if not root.exists():
        return []
    out: List[Path] = []
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            out.append(f)
    return sorted(out)


# -------------------------
# Background library (for on-the-fly mode)
# -------------------------
class _BackgroundLibrary:
    def __init__(self, background_dir: str):
        self.background_dir = str(background_dir)
        self.paths: List[Path] = []
        self._scan()

    def _scan(self):
        p = Path(self.background_dir)
        if not p.exists():
            self.paths = []
            return
        out = []
        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                out.append(f)
        self.paths = sorted(out)

    def available(self) -> bool:
        return len(self.paths) > 0

    def sample(self) -> Optional[Path]:
        if not self.paths:
            return None
        i = int(np.random.randint(0, len(self.paths)))
        return self.paths[i]


_BG_CACHE: dict[str, _BackgroundLibrary] = {}


def _get_bg_lib(background_dir: str) -> _BackgroundLibrary:
    key = str(background_dir)
    lib = _BG_CACHE.get(key)
    if lib is None:
        lib = _BackgroundLibrary(background_dir)
        _BG_CACHE[key] = lib
    return lib


# -------------------------
# Helpers
# -------------------------
def _rand_uniform(a: float, b: float) -> float:
    return float(a + (b - a) * np.random.rand())


def _rand_int(a: int, b: int) -> int:
    return int(np.random.randint(a, b + 1))


def _center_crop_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) < size:
        scale = size / float(max(1, min(h, w)))
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
    y0 = max(0, (h - size) // 2)
    x0 = max(0, (w - size) // 2)
    out = img[y0:y0 + size, x0:x0 + size].copy()
    if out.shape[0] != size or out.shape[1] != size:
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    return out


def _make_soft_mask(h: int, w: int, feather: int = 6) -> np.ndarray:
    mask = np.ones((h, w), np.float32)
    if feather <= 0:
        return mask
    k = min(feather, h // 2, w // 2)
    if k <= 0:
        return mask
    mask[:k, :] *= np.linspace(0, 1, k)[:, None]
    mask[-k:, :] *= np.linspace(1, 0, k)[:, None]
    mask[:, :k] *= np.linspace(0, 1, k)[None, :]
    mask[:, -k:] *= np.linspace(1, 0, k)[None, :]
    return np.clip(mask, 0.0, 1.0)


def _alpha_blend(bg: np.ndarray, fg: np.ndarray, mask: np.ndarray, x0: int, y0: int) -> np.ndarray:
    out = bg.copy()
    h, w = fg.shape[:2]
    roi = out[y0:y0 + h, x0:x0 + w]
    m = mask[..., None].astype(np.float32)
    roi_f = roi.astype(np.float32)
    fg_f = fg.astype(np.float32)
    blended = roi_f * (1.0 - m) + fg_f * m
    out[y0:y0 + h, x0:x0 + w] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def _pad_to_square_white(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    pad_t = (m - h) // 2
    pad_b = m - h - pad_t
    pad_l = (m - w) // 2
    pad_r = m - w - pad_l
    return cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))


def _composite_on_bg(qr_bgr: np.ndarray, bg_bgr: np.ndarray, ctx: ContextConfig) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Composite QR onto the given background.
    Returns (composited_img, bbox=(x0,y0,w,h)) in the output square coordinate system.
    """
    out_size = int(ctx.output_size) if int(ctx.output_size) > 0 else int(min(bg_bgr.shape[:2]))
    out_size = max(128, out_size)

    bg = _center_crop_square(bg_bgr, out_size)

    qr = _pad_to_square_white(qr_bgr)
    scale = _rand_uniform(ctx.qr_scale_range[0], ctx.qr_scale_range[1])
    target = int(round(scale * out_size))
    target = max(96, min(target, out_size - 2 * int(ctx.qr_margin_px)))
    qr = cv2.resize(qr, (target, target), interpolation=cv2.INTER_AREA)

    x_min = int(ctx.qr_margin_px)
    y_min = int(ctx.qr_margin_px)
    x_max = max(x_min, out_size - target - int(ctx.qr_margin_px))
    y_max = max(y_min, out_size - target - int(ctx.qr_margin_px))
    x0 = _rand_int(x_min, x_max)
    y0 = _rand_int(y_min, y_max)

    mask = _make_soft_mask(target, target, feather=int(ctx.feather))

    # subtle shadow under QR
    shadow = np.zeros_like(qr)
    shadow[:] = (0, 0, 0)
    shadow_mask = (mask * 0.22).astype(np.float32)
    bg2 = _alpha_blend(bg, shadow, shadow_mask, min(out_size - target, x0 + 4), min(out_size - target, y0 + 5))

    out = _alpha_blend(bg2, qr, mask, x0, y0)
    return out, (x0, y0, target, target)


def _random_perspective(img: np.ndarray, strength: Tuple[float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    s = _rand_uniform(strength[0], strength[1])
    dx = int(round(s * w))
    dy = int(round(s * h))
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [_rand_int(0, dx), _rand_int(0, dy)],
            [_rand_int(w - 1 - dx, w - 1), _rand_int(0, dy)],
            [_rand_int(w - 1 - dx, w - 1), _rand_int(h - 1 - dy, h - 1)],
            [_rand_int(0, dx), _rand_int(h - 1 - dy, h - 1)],
        ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _random_blur(img: np.ndarray, sigma_rng: Tuple[float, float]) -> np.ndarray:
    sigma = _rand_uniform(sigma_rng[0], sigma_rng[1])
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _random_brightness(img: np.ndarray, gain_rng: Tuple[float, float]) -> np.ndarray:
    gain = _rand_uniform(gain_rng[0], gain_rng[1])
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def _random_noise(img: np.ndarray, sigma_rng: Tuple[float, float]) -> np.ndarray:
    sigma = _rand_uniform(sigma_rng[0], sigma_rng[1])
    n = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def _random_jpeg(img: np.ndarray, q_rng: Tuple[int, int]) -> np.ndarray:
    q = _rand_int(int(q_rng[0]), int(q_rng[1]))
    enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])[1]
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out if out is not None else img


def _random_occlusion(img: np.ndarray, frac_rng: Tuple[float, float]) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    out = img.copy()
    h, w = out.shape[:2]
    frac = _rand_uniform(frac_rng[0], frac_rng[1])
    area = int(frac * h * w)
    rw = int(round(np.sqrt(area) * _rand_uniform(0.7, 1.4)))
    rh = max(8, int(round(area / max(8, rw))))
    rw = max(8, min(rw, w - 1))
    rh = max(8, min(rh, h - 1))
    x0 = _rand_int(0, max(0, w - rw - 1))
    y0 = _rand_int(0, max(0, h - rh - 1))
    color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
    cv2.rectangle(out, (x0, y0), (x0 + rw, y0 + rh), color, thickness=-1)
    return out, (x0, y0, rw, rh)


def _should_apply_context(cfg: AugmentConfig) -> bool:
    mode = (cfg.context_mode or "mix").lower()
    if mode == "none":
        return False
    if mode == "always":
        return True
    return bool(np.random.rand() < float(cfg.context_prob))


def _context_features(img_bgr: np.ndarray, bbox: Tuple[int, int, int, int], occ: Optional[Tuple[int, int, int, int]]) -> Dict[str, float]:
    """
    Strategy C feature set (approx):
      qr_area_ratio, qr_x, qr_y, blur_score, contrast, bg_complexity, occlusion_ratio
    """
    h, w = img_bgr.shape[:2]
    x0, y0, bw, bh = bbox
    area_ratio = float((bw * bh) / max(1, (w * h)))
    cx = float((x0 + bw / 2) / max(1, w))
    cy = float((y0 + bh / 2) / max(1, h))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # blur score: variance of Laplacian (higher = sharper)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # contrast: std of grayscale
    contrast = float(gray.std())

    # bg complexity: edge density outside bbox
    edges = cv2.Canny(gray, 80, 160)
    mask = np.ones((h, w), np.uint8)
    x1 = min(w, x0 + bw)
    y1 = min(h, y0 + bh)
    mask[y0:y1, x0:x1] = 0
    bg_edges = edges[mask == 1]
    bg_complexity = float((bg_edges > 0).mean()) if bg_edges.size > 0 else 0.0

    # occlusion ratio: overlap of occlusion rect with bbox / bbox area
    occlusion_ratio = 0.0
    if occ is not None:
        ox, oy, ow, oh = occ
        bx0, by0, bw2, bh2 = x0, y0, bw, bh
        ix0 = max(bx0, ox)
        iy0 = max(by0, oy)
        ix1 = min(bx0 + bw2, ox + ow)
        iy1 = min(by0 + bh2, oy + oh)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        occlusion_ratio = float(inter / max(1, (bw2 * bh2)))

    return {
        "qr_area_ratio": area_ratio,
        "qr_x": cx,
        "qr_y": cy,
        "blur_score": blur_score,
        "contrast": contrast,
        "bg_complexity": bg_complexity,
        "occlusion_ratio": float(occlusion_ratio),
    }


# -------------------------
# Offline context-aware augmentation API
# -------------------------
def augment_qr_contextual(
    qr_bgr: np.ndarray,
    bg_bgr: np.ndarray,
    cam_cfg: AugmentConfig,
    ctx_cfg: ContextConfig,
    return_features: bool = False,
) -> Tuple[np.ndarray, Optional[Dict[str, float]], Tuple[int, int, int, int]]:
    """
    Explicitly composite qr onto given background, then apply camera effects.
    Returns (aug_img, feats(optional), bbox).
    """
    # If QR is tiny, upscale first (prevents CNN kernel errors downstream)
    h0, w0 = qr_bgr.shape[:2]
    if min(h0, w0) < 64:
        s = int(max(64, min(int(ctx_cfg.output_size) if int(ctx_cfg.output_size) > 0 else 256, 256)))
        qr_bgr = cv2.resize(qr_bgr, (s, s), interpolation=cv2.INTER_CUBIC)

    out, bbox = _composite_on_bg(qr_bgr, bg_bgr, ctx_cfg)

    # camera effects
    occ_rect = None
    if np.random.rand() < cam_cfg.p_perspective:
        out = _random_perspective(out, cam_cfg.perspective_strength)
    if np.random.rand() < cam_cfg.p_blur:
        out = _random_blur(out, cam_cfg.blur_sigma)
    if np.random.rand() < cam_cfg.p_brightness:
        out = _random_brightness(out, cam_cfg.brightness_gain)
    if np.random.rand() < cam_cfg.p_noise:
        out = _random_noise(out, cam_cfg.noise_sigma)
    if np.random.rand() < cam_cfg.p_jpeg:
        out = _random_jpeg(out, cam_cfg.jpeg_q)
    if np.random.rand() < cam_cfg.p_occlusion:
        out, occ_rect = _random_occlusion(out, cam_cfg.occlusion_frac)

    feats = _context_features(out, bbox, occ_rect) if return_features else None
    return out, feats, bbox


# -------------------------
# Main augmentation entrypoint used by training datasets
# -------------------------
def augment_qr_realistic(img_bgr: np.ndarray, cfg: Optional[AugmentConfig] = None) -> np.ndarray:
    """
    Used by training datasets (on-the-fly).
    If backgrounds exist and context_mode allows it, composite onto a random background
    and then apply camera effects.
    """
    cfg = cfg or AugmentConfig()
    out = img_bgr

    # if input QR is tiny, upscale first (prevents CNN kernel errors)
    h0, w0 = out.shape[:2]
    if min(h0, w0) < 64:
        s = int(max(64, min(int(cfg.output_size), 256)))
        out = cv2.resize(out, (s, s), interpolation=cv2.INTER_CUBIC)

    # On-the-fly background compositing
    lib = _get_bg_lib(cfg.background_dir)
    if lib.available() and _should_apply_context(cfg):
        bg_path = lib.sample()
        bg = cv2.imread(str(bg_path), cv2.IMREAD_COLOR) if bg_path is not None else None
        if bg is None:
            bg = np.full((int(cfg.output_size), int(cfg.output_size), 3), 235, np.uint8)

        ctx_cfg = ContextConfig(
            p_context=cfg.context_prob,
            output_size=int(cfg.output_size),
            qr_scale_range=cfg.qr_scale_range,
            qr_margin_px=int(cfg.qr_margin_px),
            feather=6,
        )
        out, _feats, _bbox = augment_qr_contextual(out, bg, cfg, ctx_cfg, return_features=False)
    else:
        # Ensure consistent square output when no background compositing
        if cfg.output_size and (out.shape[0] != cfg.output_size or out.shape[1] != cfg.output_size):
            out = _pad_to_square_white(out)
            out = cv2.resize(out, (int(cfg.output_size), int(cfg.output_size)), interpolation=cv2.INTER_AREA)

        # camera effects
        if np.random.rand() < cfg.p_perspective:
            out = _random_perspective(out, cfg.perspective_strength)
        if np.random.rand() < cfg.p_blur:
            out = _random_blur(out, cfg.blur_sigma)
        if np.random.rand() < cfg.p_brightness:
            out = _random_brightness(out, cfg.brightness_gain)
        if np.random.rand() < cfg.p_noise:
            out = _random_noise(out, cfg.noise_sigma)
        if np.random.rand() < cfg.p_jpeg:
            out = _random_jpeg(out, cfg.jpeg_q)
        if np.random.rand() < cfg.p_occlusion:
            out, _ = _random_occlusion(out, cfg.occlusion_frac)

    return out
