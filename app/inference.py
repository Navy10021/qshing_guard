"""
모델 로딩 및 추론
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import joblib
import numpy as np
import torch


@dataclass
class PredictResult:
    prob: float
    decision: str
    warn_thr: float
    block_thr: float
    use_context: bool


def _add_repo_to_path(qshing_root: Path) -> None:
    """qshing_guard 루트를 sys.path에 추가해 src 패키지 임포트가 가능하도록 함"""
    root = str(qshing_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_thresholds(path: Path) -> Tuple[float, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if "thresholds" in d and isinstance(d["thresholds"], dict):
        return float(d["thresholds"].get("warn", 0.5)), float(d["thresholds"].get("block", 0.5))
    return float(d.get("warn_threshold", 0.5)), float(d.get("block_threshold", 0.5))


def _load_temperature(calib_json: Path) -> float:
    d = json.loads(calib_json.read_text(encoding="utf-8"))
    return float(d.get("temperature", 1.0))


def _decision(prob: float, warn_thr: float, block_thr: float) -> str:
    if prob >= block_thr:
        return "BLOCK"
    if prob >= warn_thr:
        return "WARN"
    return "ALLOW"


def _read_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _to_chw_float(img_bgr: np.ndarray, size: int = 224) -> torch.Tensor:
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
    return x


def decode_qr_from_image(img_bgr: np.ndarray) -> Optional[str]:
    """OpenCV QRCodeDetector로 QR을 디코딩"""
    detector = cv2.QRCodeDetector()
    text, _points, _ = detector.detectAndDecode(img_bgr)
    text = (text or "").strip()
    return text if text else None


def predict_single(
    qshing_root: Path,
    model_dir: Path,
    thresholds_json: Path,
    calibration_json: Optional[Path],
    qr_path: Path,
    url_norm: str,
) -> PredictResult:
    """단일 QR 이미지 + URL로 예측"""
    _add_repo_to_path(qshing_root)

    # src 패키지로 절대 임포트
    from src.train.modeling_fusion import build_fusion_model  # type: ignore
    from src.train.modeling_qr import load_model_state  # type: ignore
    from src.train.datasets import _extract_qr_context  # type: ignore
    from src.features.url_lexical import batch_extract_url_features  # type: ignore
    from src.eval.calibration import apply_temperature  # type: ignore

    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")
    if not thresholds_json.exists():
        raise FileNotFoundError(f"thresholds_json not found: {thresholds_json}")

    warn_thr, block_thr = _load_thresholds(thresholds_json)

    vec_path = model_dir / "tfidf_vectorizer.joblib"
    if not vec_path.exists():
        raise FileNotFoundError(f"Missing vectorizer: {vec_path}")
    vec = joblib.load(vec_path)

    # URL features
    X = vec.transform([str(url_norm)]).astype(np.float32).toarray()
    x_url = torch.from_numpy(X[0])
    lex = batch_extract_url_features([str(url_norm)]).astype(np.float32)
    x_lex = torch.from_numpy(lex[0])

    # Image + context
    img = _read_image_bgr(qr_path)
    x_img = _to_chw_float(img, size=224)
    x_ctx = torch.from_numpy(_extract_qr_context(img))

    # 체크포인트 설정
    ckpt_path = model_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    dev = _device()
    try:
        ckpt = torch.load(ckpt_path, map_location=dev)
        fusion_mode = ckpt.get("fusion_mode", "gated")
        use_context = bool(ckpt.get("use_context", True))
    except Exception:
        fusion_mode = "gated"
        use_context = True

    try:
        model = build_fusion_model(fusion_mode, url_dim=X.shape[1], lex_dim=x_lex.numel(), num_classes=2, ctx_dim=7).to(dev)
    except TypeError:
        model = build_fusion_model(fusion_mode, url_dim=X.shape[1], lex_dim=x_lex.numel(), num_classes=2).to(dev)

    load_model_state(model, str(ckpt_path), device=dev)
    model.eval()

    # 배치 추론
    ximg = x_img.unsqueeze(0).to(dev)
    xurl = x_url.unsqueeze(0).to(dev)
    xlex = x_lex.unsqueeze(0).to(dev)
    xctx = x_ctx.unsqueeze(0).to(dev)

    try:
        logits = model(ximg, xurl, xlex, xctx) if use_context else model(ximg, xurl, xlex)
    except TypeError:
        logits = model(ximg, xurl, xlex)

    if calibration_json and calibration_json.exists():
        temperature = _load_temperature(calibration_json)
        prob = float(apply_temperature(logits.detach().cpu().numpy(), temperature)[0])
    else:
        prob = float(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()[0])

    dec = _decision(prob, warn_thr, block_thr)
    return PredictResult(prob=prob, decision=dec, warn_thr=warn_thr, block_thr=block_thr, use_context=use_context)
