"""데이터 로드와 타깃 정리."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .errors import StageError

DOWNLOAD_HINT = (
    "원본 데이터가 없습니다. ml_workflow/data/README.md의 안내대로 ProBayes v2 데이터를 내려받아 "
    "ml_workflow/data/에 두거나, 환경변수 ICT_DATA_PATH로 파일 경로를 지정하세요."
)


def load_dataset(cfg) -> tuple[pd.DataFrame, Path]:
    """parquet를 우선 읽고, 엔진이 없으면 설정된 CSV로 대체한다."""
    path = cfg.data_path
    if not path.exists():
        fb = cfg.fallback_path
        if fb is not None and fb.exists():
            print(f"[경고] {path.name}이 없어 {fb.name}을 읽습니다.")
            return pd.read_csv(fb, low_memory=False), fb
        raise StageError(f"{DOWNLOAD_HINT} (찾은 경로: {path})")
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path), path
        except ImportError:
            fb = cfg.fallback_path
            if fb is None or not fb.exists():
                raise StageError("parquet를 읽으려면 pyarrow가 필요합니다: pip install -r ml_workflow/requirements-train.txt")
            print(f"[경고] parquet 엔진이 없어 CSV로 읽습니다: {fb.name}")
            return pd.read_csv(fb, low_memory=False), fb
    return pd.read_csv(path, low_memory=False), path


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def target_rows(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """타깃이 있는 행만 남기고 정수형 타깃을 반환한다."""
    mask = df[target].notna()
    out = df.loc[mask]
    return out, out[target].astype(int)
