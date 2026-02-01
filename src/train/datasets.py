from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..qr.augmentations import AugmentConfig, augment_qr_realistic
from ..features.url_lexical import batch_extract_url_features

# -----------------------------
# I/O + tensor helpers
# -----------------------------
def _read_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _to_chw_float(img_bgr: np.ndarray, size: int = 224) -> torch.Tensor:
    # resize
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    # BGR->RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
    return x


# -----------------------------
# Context feature extractor
# -----------------------------
# Feature vector (float32) length = 7:
#  [0] qr_area_ratio    : polygon area / image area (0..1)
#  [1] qr_x             : center x normalized (0..1)
#  [2] qr_y             : center y normalized (0..1)
#  [3] blur_score       : normalized log(var(Laplacian)) (0..1-ish)
#  [4] contrast         : gray std / 128 (0..~1)
#  [5] bg_complexity    : edge density outside QR (0..1)
#  [6] occlusion_ratio  : mid-tone ratio inside QR (0..1)  (proxy)
_QR_DETECTOR = cv2.QRCodeDetector()


def _safe01(x: float) -> float:
    if np.isnan(x) or np.isinf(x):
        return 0.0
    return float(max(0.0, min(1.0, x)))


def _extract_qr_context(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # blur (variance of Laplacian) -> log scale then squash
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var_lap = float(lap.var())
    blur_score = _safe01(np.log1p(var_lap) / 10.0)  # heuristic normalization

    # contrast
    contrast = _safe01(float(gray.std()) / 128.0)

    # QR polygon detection (no decode needed)
    ok, pts = _QR_DETECTOR.detect(gray)
    area_ratio = 0.0
    cx = 0.0
    cy = 0.0
    bg_complexity = 0.0
    occl = 0.0

    if ok and pts is not None and len(pts) > 0:
        poly = pts[0].astype(np.float32)  # shape (4,2)
        # area
        area = float(cv2.contourArea(poly))
        area_ratio = _safe01(area / float(max(1, w * h)))
        # center
        cx = _safe01(float(poly[:, 0].mean()) / float(max(1, w)))
        cy = _safe01(float(poly[:, 1].mean()) / float(max(1, h)))

        # masks: QR region + background region
        mask_qr = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask_qr, poly.astype(np.int32), 255)
        mask_bg = cv2.bitwise_not(mask_qr)

        # background complexity: edge density on bg only
        edges = cv2.Canny(gray, 80, 160)
        bg_edges = edges[mask_bg > 0]
        if bg_edges.size > 0:
            bg_complexity = _safe01(float((bg_edges > 0).mean()))

        # occlusion proxy: mid-tone ratio within QR region (blur/lighting/occlusion increases mid-tones)
        roi = gray[mask_qr > 0]
        if roi.size > 0:
            mid = ((roi >= 50) & (roi <= 205)).mean()
            occl = _safe01(float(mid))

    else:
        # if QR not detected, still compute bg_complexity on whole image (fallback)
        edges = cv2.Canny(gray, 80, 160)
        bg_complexity = _safe01(float((edges > 0).mean()))
        occl = _safe01(float(((gray >= 50) & (gray <= 205)).mean()))

    return np.array([area_ratio, cx, cy, blur_score, contrast, bg_complexity, occl], dtype=np.float32)


# -----------------------------
# Augmentation configs
# -----------------------------
@dataclass
class QRDatasetConfig:
    image_size: int = 224
    augment: bool = True
    augment_strength: str = "default"  # light/default/strong
    use_context: bool = True

    # NEW: optionally return payload string (for co-evolution decode constraints)
    return_payload: bool = False
    payload_col: str = "url_norm"  # column name to return when return_payload=True


def _cfg_from_strength(strength: str) -> AugmentConfig:
    cfg = AugmentConfig()

    # On-the-fly background compositing defaults:
    # If assets/backgrounds exists and has images, train-time augmentation will
    # automatically create "QR on real background" samples without extra CLI flags.
    cfg.background_dir = "assets/backgrounds"
    cfg.output_size = 512  # internal working resolution before model resize
    cfg.context_mode = "mix"
    if strength == "light":
        cfg.context_prob = 0.25
    elif strength == "strong":
        cfg.context_prob = 0.75
    else:
        cfg.context_prob = 0.50

    if strength == "light":
        cfg.p_perspective = 0.6
        cfg.p_blur = 0.5
        cfg.p_noise = 0.45
        cfg.noise_sigma = (1.5, 10.0)
        cfg.perspective_strength = (0.03, 0.08)
        cfg.occlusion_frac = (0.04, 0.14)
    elif strength == "strong":
        cfg.p_perspective = 0.95
        cfg.p_blur = 0.85
        cfg.p_noise = 0.8
        cfg.noise_sigma = (4.0, 26.0)
        cfg.perspective_strength = (0.06, 0.16)
        cfg.occlusion_frac = (0.08, 0.28)
        cfg.jpeg_q = (18, 85)
    return cfg


# -----------------------------
# Datasets
# -----------------------------
class QRImageDataset(Dataset):
    """Dataset over manifest CSV that contains qr_path and label.

    Returns (default):
      - if cfg.use_context: (x_img, x_ctx, y)
      - else: (x_img, y)

    If cfg.return_payload=True, appends payload string at the end:
      - if cfg.use_context: (x_img, x_ctx, y, payload)
      - else: (x_img, y, payload)

    NOTE: For QR-only payload matching to work, the CSV must include cfg.payload_col
          (default: url_norm). If missing, payload will be empty string.
    """

    def __init__(
        self,
        manifest_csv: str,
        cfg: Optional[QRDatasetConfig] = None,
        seed: int = 42,
        require_label: bool = True,
    ):
        self.df = pd.read_csv(manifest_csv)
        if "qr_path" not in self.df.columns:
            raise ValueError("manifest must include qr_path. Run generate_qr first.")

        self.has_label = ("label" in self.df.columns)
        if (not self.has_label) and require_label:
            raise ValueError("manifest must include label")
        if not self.has_label:
            self.df["label"] = 0  # inference/demo용 더미 라벨

        self.cfg = cfg or QRDatasetConfig()
        self.rng = random.Random(seed)
        self.aug_cfg = _cfg_from_strength(self.cfg.augment_strength)
        self.labels = self.df["label"].astype(int).values

        # payload column guard (no hard error to keep backward-compat)
        if self.cfg.return_payload and (self.cfg.payload_col not in self.df.columns):
            self.df[self.cfg.payload_col] = ""

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = str(row.qr_path)
        y = int(row.label)

        img = _read_image_bgr(path)
        if self.cfg.augment:
            img = augment_qr_realistic(img, self.aug_cfg)

        x_img = _to_chw_float(img, size=self.cfg.image_size)

        payload = None
        if self.cfg.return_payload:
            payload = str(row.get(self.cfg.payload_col, ""))

        if self.cfg.use_context:
            ctx = _extract_qr_context(img)
            x_ctx = torch.from_numpy(ctx)
            if self.cfg.return_payload:
                return x_img, x_ctx, torch.tensor(y, dtype=torch.long), payload
            return x_img, x_ctx, torch.tensor(y, dtype=torch.long)

        if self.cfg.return_payload:
            return x_img, torch.tensor(y, dtype=torch.long), payload
        return x_img, torch.tensor(y, dtype=torch.long)


class FusionDataset(Dataset):
    """Fusion dataset that returns (image, url_tfidf_dense, url_lex_feats, ctx_feats, label).

    If cfg.return_payload=True, appends payload string at the end:
      (x_img, x_url, x_lex, x_ctx, y, payload)

    Payload defaults to cfg.payload_col (url_norm).
    """

    def __init__(
        self,
        csv_path: str,
        url_vec: np.ndarray,
        cfg: Optional[QRDatasetConfig] = None,
        seed: int = 42,
        require_label: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        if "qr_path" not in self.df.columns:
            raise ValueError("CSV must include qr_path")
        if "url_norm" not in self.df.columns:
            raise ValueError("CSV must include url_norm")

        self.has_label = ("label" in self.df.columns)
        if (not self.has_label) and require_label:
            raise ValueError("CSV must include label")
        if not self.has_label:
            self.df["label"] = 0  # inference/demo용 더미 라벨

        if len(self.df) != url_vec.shape[0]:
            raise ValueError(f"url_vec rows must match df: {url_vec.shape[0]} vs {len(self.df)}")

        self.url_vec = url_vec.astype(np.float32)
        self.lex = batch_extract_url_features(self.df["url_norm"].astype(str).tolist()).astype(np.float32)
        self.labels = self.df["label"].astype(int).values  # sampler용

        self.cfg = cfg or QRDatasetConfig()
        self.aug_cfg = _cfg_from_strength(self.cfg.augment_strength)
        self.rng = random.Random(seed)

        if self.cfg.return_payload and (self.cfg.payload_col not in self.df.columns):
            # keep compat: allow custom payload col, fallback to empty string
            self.df[self.cfg.payload_col] = ""

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = _read_image_bgr(str(row.qr_path))
        if self.cfg.augment:
            img = augment_qr_realistic(img, self.aug_cfg)

        x_img = _to_chw_float(img, size=self.cfg.image_size)
        x_url = torch.from_numpy(self.url_vec[idx])
        x_lex = torch.from_numpy(self.lex[idx])
        y = torch.tensor(int(row.label), dtype=torch.long)

        if self.cfg.use_context:
            ctx = _extract_qr_context(img)
            x_ctx = torch.from_numpy(ctx)
        else:
            x_ctx = torch.zeros(7, dtype=torch.float32)

        if self.cfg.return_payload:
            payload = str(row.get(self.cfg.payload_col, ""))
            return x_img, x_url, x_lex, x_ctx, y, payload

        return x_img, x_url, x_lex, x_ctx, y
