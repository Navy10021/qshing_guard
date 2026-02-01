from __future__ import annotations

"""Lightweight URL lexical feature extraction.

These features are cheap and stable for production scoring.
They are also useful for fusion models.
"""

import math
import re
from urllib.parse import urlparse

import numpy as np


_RE_IP = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_RE_HEX = re.compile(r"%[0-9A-Fa-f]{2}")


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log(p + 1e-12, 2)
    return float(ent)


def extract_url_features(url: str) -> np.ndarray:
    """Return a fixed-size numeric feature vector for a URL."""
    try:
        p = urlparse(url)
    except Exception:
        p = urlparse("")

    host = (p.hostname or "")
    path = (p.path or "")
    query = (p.query or "")

    full = url or ""
    host_l = host.lower()

    # basic counts
    length = len(full)
    host_len = len(host)
    path_len = len(path)
    query_len = len(query)

    n_dots = host.count(".")
    n_hyphen = host.count("-")
    n_digits = sum(ch.isdigit() for ch in full)
    n_special = sum(ch in "@:/?&=%_" for ch in full)
    n_slashes = full.count("/")
    n_at = full.count("@")
    n_qm = full.count("?")
    n_eq = full.count("=")
    n_amp = full.count("&")

    has_ip = 1.0 if _RE_IP.match(host_l) else 0.0
    has_https = 1.0 if (p.scheme or "").lower() == "https" else 0.0
    has_port = 1.0 if p.port is not None else 0.0

    # suspicious patterns
    pct_hex = len(_RE_HEX.findall(full))
    has_punycode = 1.0 if "xn--" in host_l else 0.0
    has_double_slash = 1.0 if "//" in (p.path or "") else 0.0

    # entropy
    ent_host = _shannon_entropy(host)
    ent_path = _shannon_entropy(path)
    ent_full = _shannon_entropy(full)

    feats = np.array(
        [
            length,
            host_len,
            path_len,
            query_len,
            n_dots,
            n_hyphen,
            n_digits,
            n_special,
            n_slashes,
            n_at,
            n_qm,
            n_eq,
            n_amp,
            pct_hex,
            has_ip,
            has_https,
            has_port,
            has_punycode,
            has_double_slash,
            ent_host,
            ent_path,
            ent_full,
        ],
        dtype=np.float32,
    )
    return feats


def batch_extract_url_features(urls: list[str]) -> np.ndarray:
    return np.stack([extract_url_features(u) for u in urls], axis=0)
