# src/utils/qr_decode.py
from __future__ import annotations

from typing import Optional
import contextlib
import io
from urllib.parse import urlsplit, urlunsplit

import cv2
from pyzbar.pyzbar import decode, ZBarSymbol


@contextlib.contextmanager
def _suppress_stderr(enabled: bool = True):
    """Suppress noisy zbar warnings (databar assertions, etc.)."""
    if not enabled:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        yield


def normalize_url_for_match(s: str) -> str:
    """Best-effort canonicalization for matching decoded QR payload to manifest URL.
    Not meant to be a full normalizer; just stable for equality checks.
    """
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""

    try:
        parts = urlsplit(s)
        scheme = (parts.scheme or "").lower()
        netloc = (parts.netloc or "").lower()

        # If QR contains bare domain without scheme, urlsplit puts it in path.
        if not scheme and not netloc and parts.path and "." in parts.path.split("/")[0]:
            # treat as //host/path
            parts = urlsplit("http://" + s)
            scheme = "http"
            netloc = (parts.netloc or "").lower()

        # remove default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        if netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]

        path = parts.path or ""
        # strip trailing slash (except root)
        if path.endswith("/") and len(path) > 1:
            path = path[:-1]

        # keep query; drop fragment
        query = parts.query or ""
        return urlunsplit((scheme, netloc, path, query, ""))
    except Exception:
        # fallback: lowercase + strip trailing slash
        s2 = s.lower()
        if s2.endswith("/") and len(s2) > 1:
            s2 = s2[:-1]
        return s2


def decode_qr_array(img_bgr) -> Optional[str]:
    # zbar can be noisy; suppress stderr to avoid log spam
    with _suppress_stderr(True):
        codes = decode(img_bgr, symbols=[ZBarSymbol.QRCODE])
    if not codes:
        return None
    data = codes[0].data
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return None


def decode_qr_tensor(x_rgb01, size: int = 224) -> Optional[str]:
    """Decode a single image tensor in [0,1], shape (3,H,W) RGB."""
    import numpy as np
    if hasattr(x_rgb01, "detach"):
        x = x_rgb01.detach().float().clamp(0, 1).cpu().numpy()
    else:
        x = x_rgb01
    # CHW->HWC
    x = (x.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    # ensure target size for decoder stability
    if x.shape[0] != size or x.shape[1] != size:
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    # RGB->BGR
    bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return decode_qr_array(bgr)
