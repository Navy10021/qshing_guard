# QShing Guard Mobile (Streamlit)

모바일 브라우저에서 사용할 수 있는 QR 피싱 판별 웹앱입니다.

## 실행 방법
```powershell
cd d:\PPT\qshing_guard_mobile_app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```powershell
streamlit run app.py
```

## 준비물
- `qshing_guard` 학습 결과(모델/벡터라이저)
- 임계값 JSON (`thresholds.json`)
- (선택) 보정 JSON (`calibration.json`)

기본 경로는 `config.py`에 설정되어 있으며, 앱 사이드바에서 수정할 수 있습니다.
