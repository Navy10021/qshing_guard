# qshing_guard/src/data/utils.py
from __future__ import annotations

import re
import hashlib
from typing import List, Optional
from urllib.parse import urlsplit, urlunsplit, quote

# Optional: robust registrable domain (eTLD+1)
try:
    import tldextract  # type: ignore
    _HAS_TLDEXTRACT = True
except Exception:  # pragma: no cover
    tldextract = None
    _HAS_TLDEXTRACT = False

# Conservative URL-ish pattern: http(s)/www or bare domain.tld[/path]
_URL_REGEX = re.compile(
    r'(?i)\b((?:https?://|www\.)[^\s<>\"]+|[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s<>\"]*)?)'
)

_TRAIL_PUNCT = " ).,;:'\"!?)]}".strip()


def extract_urls(text: Optional[str]) -> List[str]:
    """Extract URL-like strings from text."""
    if not isinstance(text, str):
        return []
    matches = _URL_REGEX.findall(text)
    out: List[str] = []
    for m in matches:
        u = m.strip().rstrip(_TRAIL_PUNCT)
        if len(u) < 6:
            continue
        out.append(u)
    return out


def ensure_scheme(url: str) -> str:
    url = url.strip()
    if url.lower().startswith("http://") or url.lower().startswith("https://"):
        return url
    if url.lower().startswith("www."):
        return "http://" + url
    # bare domain/path
    return "http://" + url


def normalize_url(url: str, drop_fragment: bool = True) -> Optional[str]:
    """Normalize URL for dedup/training.

    Rules:
    - add scheme if missing
    - lowercase host
    - remove default ports
    - percent-encode path safely
    - optionally drop fragment
    """
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not url:
        return None
    url = ensure_scheme(url)

    try:
        parts = urlsplit(url)
    except Exception:
        return None

    scheme = parts.scheme.lower() if parts.scheme else "http"
    netloc = parts.netloc.lower()

    # strip default ports
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    path = parts.path or "/"
    path = quote(path, safe="/%:@")
    query = parts.query
    fragment = "" if drop_fragment else parts.fragment

    norm = urlunsplit((scheme, netloc, path, query, fragment))

    # sanity check netloc
    if not netloc or "." not in netloc:
        return None
    return norm


def url_hash(url_norm: str) -> str:
    return hashlib.sha256(url_norm.encode("utf-8")).hexdigest()[:16]


def _host_from_url_norm(url_norm: str) -> Optional[str]:
    if not isinstance(url_norm, str) or not url_norm:
        return None
    try:
        host = urlsplit(url_norm).netloc.lower()
    except Exception:
        return None
    if not host or "." not in host:
        return None
    # strip credentials if any
    if "@" in host:
        host = host.split("@", 1)[-1]
    # strip port
    if ":" in host:
        host = host.split(":", 1)[0]
    return host


def registrable_domain(url_norm: str) -> Optional[str]:
    """Return registrable domain (effective TLD+1) for leakage-robust splits.

    - Uses tldextract when available (offline; network disabled)
    - Falls back to a conservative heuristic (last 2 labels)
      (less accurate for ccTLDs like .co.kr, but still better than URL-level split)
    """
    host = _host_from_url_norm(url_norm)
    if not host:
        return None

    if _HAS_TLDEXTRACT:
        try:
            # disable any network fetch; rely on packaged snapshot/cache
            ext = tldextract.TLDExtract(cache_dir=False, suffix_list_urls=None)
            r = ext(host)
            if r.domain and r.suffix:
                return f"{r.domain}.{r.suffix}"
        except Exception:
            pass

    parts = [p for p in host.split(".") if p]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host
