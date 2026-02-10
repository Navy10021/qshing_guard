"""
환경 설정
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    """앱 설정 (필요 시 수정)"""
    # qshing_guard 루트 경로
    qshing_root: Path = Path("d:/PPT/qshing_guard")

    # 모델 아티팩트 경로 (학습 결과 폴더)
    model_dir: Path = Path("d:/PPT/qshing_guard/artifacts/models/fusion")

    # 운영 임계값 JSON (warn/block)
    thresholds_json: Path = Path("d:/PPT/qshing_guard/artifacts/reports/fusion_eval/thresholds.json")

    # 보정(temperature) JSON (선택)
    calibration_json: Path = Path("d:/PPT/qshing_guard/artifacts/models/fusion/calibration.json")
