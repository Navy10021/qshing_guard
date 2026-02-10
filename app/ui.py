"""
모바일 친화 UI
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
import cv2

from config import AppConfig
from inference import predict_single, decode_qr_from_image


def _apply_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&family=Noto+Sans+KR:wght@400;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Noto Sans KR', sans-serif;
            background: #f6f8fb;
        }

        .hero {
            padding: 18px 20px;
            border-radius: 18px;
            background: linear-gradient(120deg, #dbeafe 0%, #fce7f3 55%, #fef3c7 100%);
            border: 1px solid #e8eef7;
            box-shadow: 0 10px 24px rgba(16,24,40,0.08);
            margin-bottom: 14px;
        }

        .title {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.8rem;
            font-weight: 800;
            color: #0f172a;
        }

        .sub {
            color: #475467;
            font-size: 0.95rem;
        }

        .card {
            background: #ffffff;
            border-radius: 14px;
            border: 1px solid #e4e7ec;
            padding: 14px 16px;
            box-shadow: 0 6px 16px rgba(16,24,40,0.06);
        }

        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            background: #0f172a;
            color: #fff;
            margin-right: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sidebar(cfg: AppConfig):
    with st.sidebar:
        st.markdown("## 설정")
        st.markdown("모델/임계값 경로를 확인해 주세요.")
        st.divider()

        qshing_root = st.text_input("qshing_guard 루트", str(cfg.qshing_root))
        model_dir = st.text_input("모델 디렉터리", str(cfg.model_dir))
        thresholds_json = st.text_input("임계값 JSON", str(cfg.thresholds_json))
        calibration_json = st.text_input("보정 JSON (선택)", str(cfg.calibration_json))

        st.divider()
        st.markdown("### 상태 확인")
        st.write("루트:", "✅" if Path(qshing_root).exists() else "❌")
        st.write("모델:", "✅" if Path(model_dir).exists() else "❌")
        st.write("임계값:", "✅" if Path(thresholds_json).exists() else "❌")
        st.write("보정:", "✅" if Path(calibration_json).exists() else "❌")

    return Path(qshing_root), Path(model_dir), Path(thresholds_json), Path(calibration_json)


def _save_upload(upload) -> Path:
    tmp_dir = Path(tempfile.gettempdir()) / "qshing_guard_mobile"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / upload.name
    path.write_bytes(upload.getvalue())
    return path


def render_app():
    st.set_page_config(page_title="QShing Guard Mobile", page_icon="🛡️", layout="centered")
    _apply_style()

    cfg = AppConfig()
    qshing_root, model_dir, thresholds_json, calibration_json = _sidebar(cfg)

    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>QShing Guard 모바일 판별</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>QR 이미지와 URL을 입력하면 WARN/BLOCK 여부를 즉시 판단합니다.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<span class='badge'>QR</span><span class='badge'>URL</span><span class='badge'>Fusion</span>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        qr_file = st.file_uploader("QR 이미지 업로드", type=["png", "jpg", "jpeg", "webp", "bmp"])
        url_norm = st.text_input("스캔된 URL 또는 정규화 URL 입력 (선택)")
        run = st.button("판별하기", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if run:
        if not qr_file:
            st.warning("QR 이미지를 업로드해 주세요.")
            return

        if not model_dir.exists() or not thresholds_json.exists():
            st.error("모델 경로나 임계값 JSON이 올바르지 않습니다. 사이드바에서 확인해 주세요.")
            return

        with st.spinner("모델이 판별 중입니다..."):
            qr_path = _save_upload(qr_file)
            final_url = url_norm.strip()
            if not final_url:
                img_bgr = cv2.imread(str(qr_path), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    st.error("업로드한 이미지를 읽을 수 없습니다.")
                    return
                decoded = decode_qr_from_image(img_bgr)
                if not decoded:
                    st.warning("QR에서 URL을 추출하지 못했습니다. URL을 직접 입력해 주세요.")
                    return
                final_url = decoded
                st.session_state["decoded_url"] = final_url

            result = predict_single(
                qshing_root=qshing_root,
                model_dir=model_dir,
                thresholds_json=thresholds_json,
                calibration_json=calibration_json if calibration_json.exists() else None,
                qr_path=qr_path,
                url_norm=final_url,
            )

        st.markdown("### 결과")
        c1, c2, c3 = st.columns(3)
        c1.metric("결정", result.decision)
        c2.metric("피싱 확률", f"{result.prob:.3f}")
        c3.metric("Context 사용", "ON" if result.use_context else "OFF")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if "decoded_url" in st.session_state and st.session_state["decoded_url"]:
            st.write("QR에서 추출된 URL:", st.session_state["decoded_url"])
        st.write("WARN 임계값:", result.warn_thr)
        st.write("BLOCK 임계값:", result.block_thr)
        st.markdown("</div>", unsafe_allow_html=True)

        st.image(qr_path, caption="업로드한 QR", use_container_width=True)

    with st.expander("사용 팁"):
        st.markdown(
            """
- URL이 길거나 난독화되어 있어도 QR 이미지와 함께 판단합니다.
- 보정 JSON이 있으면 더 안정적인 확률을 제공합니다.
- 모바일에서는 이미지를 가볍게(800px 이하) 올리면 빠릅니다.
            """
        )
