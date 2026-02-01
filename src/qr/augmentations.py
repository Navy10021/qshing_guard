from __future__ import annotations

"""
Real-world QR augmentation utilities.

This module supports two modes:
1) effect-only: perspective/blur/noise/jpeg/occlusion on the QR image
2) context-aware (on-the-fly): composite QR onto a random background from assets/backgrounds/*
   then apply camera effects (perspective/blur/noise/jpeg/occlusion).

The on-the-fly mode is activated when:
 - cfg.background_dir exists and contains images, AND
 - cfg.context_mode != "none"

Recommended: keep assets/backgrounds with subfolders like document/poster/receipt/screen.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


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
    qr_scale_range: Tuple[float, float] = (0.32, 0.62)  # relative to background min side
    qr_margin_px: int = 20  # keep QR away from edges

    # randomness: if you want deterministic training, seed numpy before calling augment()
    # (datasets.py already seeds numpy globally; per-sample determinism isn't guaranteed)


class _BackgroundLibrary:
    """Loads background image paths (not pixels) and samples paths uniformly."""

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


def _rand_uniform(a: float, b: float) -> float:
    return float(a + (b - a) * np.random.rand())


def _rand_int(a: int, b: int) -> int:
    # inclusive range
    return int(np.random.randint(a, b + 1))


def _resize_keep_ar(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    if h > w:
        new_h = size
        new_w = int(round(w * size / h))
    else:
        new_w = size
        new_h = int(round(h * size / w))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _read_bg(path: Path, out_size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        # fallback: simple gray
        return np.full((out_size, out_size, 3), 235, np.uint8)
    # resize to cover out_size x out_size (center crop)
    h, w = img.shape[:2]
    if min(h, w) < out_size:
        scale = out_size / float(min(h, w))
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
    # center crop square
    y0 = max(0, (h - out_size) // 2)
    x0 = max(0, (w - out_size) // 2)
    img = img[y0:y0 + out_size, x0:x0 + out_size].copy()
    if img.shape[0] != out_size or img.shape[1] != out_size:
        img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return img


def _alpha_blend(bg: np.ndarray, fg: np.ndarray, mask: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """Blend fg into bg at (x0,y0) using mask in [0,1]."""
    out = bg.copy()
    h, w = fg.shape[:2]
    roi = out[y0:y0 + h, x0:x0 + w]
    m = mask[..., None].astype(np.float32)
    roi_f = roi.astype(np.float32)
    fg_f = fg.astype(np.float32)
    blended = roi_f * (1.0 - m) + fg_f * m
    out[y0:y0 + h, x0:x0 + w] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def _make_soft_mask(h: int, w: int, feather: int = 6) -> np.ndarray:
    mask = np.ones((h, w), np.float32)
    if feather <= 0:
        return mask
    k = feather
    mask[:k, :] *= np.linspace(0, 1, k)[:, None]
    mask[-k:, :] *= np.linspace(1, 0, k)[:, None]
    mask[:, :k] *= np.linspace(0, 1, k)[None, :]
    mask[:, -k:] *= np.linspace(1, 0, k)[None, :]
    return np.clip(mask, 0.0, 1.0)


def _composite_qr_on_bg(qr_bgr: np.ndarray, cfg: AugmentConfig) -> np.ndarray:
    """Return a square out_size image: background + QR placed on top."""
    out_size = int(cfg.output_size)
    lib = _get_bg_lib(cfg.background_dir)

    # pick background or fallback
    if lib.available():
        bg_path = lib.sample()
        bg = _read_bg(bg_path, out_size=out_size)
    else:
        bg = np.full((out_size, out_size, 3), 235, np.uint8)

    # prepare QR (ensure square-ish, resize by scale)
    qr = qr_bgr.copy()
    # ensure QR is roughly square by padding to square
    h, w = qr.shape[:2]
    m = max(h, w)
    pad_t = (m - h) // 2
    pad_b = m - h - pad_t
    pad_l = (m - w) // 2
    pad_r = m - w - pad_l
    qr = cv2.copyMakeBorder(qr, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # scale
    scale = _rand_uniform(cfg.qr_scale_range[0], cfg.qr_scale_range[1])
    target = int(round(scale * out_size))
    target = max(96, min(target, out_size - 2 * cfg.qr_margin_px))
    qr = cv2.resize(qr, (target, target), interpolation=cv2.INTER_AREA)

    # random position
    x_min = cfg.qr_margin_px
    y_min = cfg.qr_margin_px
    x_max = max(x_min, out_size - target - cfg.qr_margin_px)
    y_max = max(y_min, out_size - target - cfg.qr_margin_px)
    x0 = _rand_int(x_min, x_max)
    y0 = _rand_int(y_min, y_max)

    # blending mask: mostly hard edges, slight feather for realism
    mask = _make_soft_mask(target, target, feather=6)

    # add subtle paper shadow under QR (helps realism)
    shadow = np.zeros_like(qr)
    shadow[:] = (0, 0, 0)
    shadow_mask = (mask * 0.22).astype(np.float32)
    bg = _alpha_blend(bg, shadow, shadow_mask, x0 + 4, y0 + 5)

    # blend QR
    out = _alpha_blend(bg, qr, mask, x0, y0)
    return out


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
    out = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def _random_blur(img: np.ndarray, sigma_rng: Tuple[float, float]) -> np.ndarray:
    sigma = _rand_uniform(sigma_rng[0], sigma_rng[1])
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _random_brightness(img: np.ndarray, gain_rng: Tuple[float, float]) -> np.ndarray:
    gain = _rand_uniform(gain_rng[0], gain_rng[1])
    out = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return out


def _random_noise(img: np.ndarray, sigma_rng: Tuple[float, float]) -> np.ndarray:
    sigma = _rand_uniform(sigma_rng[0], sigma_rng[1])
    n = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return out


def _random_jpeg(img: np.ndarray, q_rng: Tuple[int, int]) -> np.ndarray:
    q = _rand_int(int(q_rng[0]), int(q_rng[1]))
    enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])[1]
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out if out is not None else img


def _random_occlusion(img: np.ndarray, frac_rng: Tuple[float, float]) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    frac = _rand_uniform(frac_rng[0], frac_rng[1])
    area = int(frac * h * w)
    # random rectangle occluder
    rw = int(round(np.sqrt(area) * _rand_uniform(0.7, 1.4)))
    rh = max(8, int(round(area / max(8, rw))))
    rw = max(8, min(rw, w - 1))
    rh = max(8, min(rh, h - 1))
    x0 = _rand_int(0, max(0, w - rw - 1))
    y0 = _rand_int(0, max(0, h - rh - 1))
    color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
    cv2.rectangle(out, (x0, y0), (x0 + rw, y0 + rh), color, thickness=-1)
    return out


def _should_apply_context(cfg: AugmentConfig) -> bool:
    mode = (cfg.context_mode or "mix").lower()
    if mode == "none":
        return False
    if mode == "always":
        return True
    # mix
    p = float(cfg.context_prob)
    return bool(np.random.rand() < p)


def augment_qr_realistic(img_bgr: np.ndarray, cfg: Optional[AugmentConfig] = None) -> np.ndarray:
    """
    Main augmentation entrypoint used by training datasets.
    If backgrounds exist and context_mode allows it, we composite onto a random background
    and then apply camera effects.
    """
    cfg = cfg or AugmentConfig()
    out = img_bgr

    # if input QR is tiny, avoid CNN kernel errors by upscaling first
    h0, w0 = out.shape[:2]
    if min(h0, w0) < 64:
        s = int(max(64, min(cfg.output_size, 256)))
        out = cv2.resize(out, (s, s), interpolation=cv2.INTER_CUBIC)

    # Context-aware background compositing (on-the-fly)
    lib = _get_bg_lib(cfg.background_dir)
    if lib.available() and _should_apply_context(cfg):
        out = _composite_qr_on_bg(out, cfg)
    else:
        # Ensure consistent square output when no background compositing
        if cfg.output_size and (out.shape[0] != cfg.output_size or out.shape[1] != cfg.output_size):
            # keep aspect by padding to square then resize
            h, w = out.shape[:2]
            m = max(h, w)
            pad_t = (m - h) // 2
            pad_b = m - h - pad_t
            pad_l = (m - w) // 2
            pad_r = m - w - pad_l
            out = cv2.copyMakeBorder(out, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
            out = cv2.resize(out, (cfg.output_size, cfg.output_size), interpolation=cv2.INTER_AREA)

    # Camera effects
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
        out = _random_occlusion(out, cfg.occlusion_frac)

    return out
