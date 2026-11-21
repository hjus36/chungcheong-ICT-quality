"""실행(run) 폴더 입출력."""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_dataset
from .errors import StageError  # noqa: F401  (기존 import 경로 유지)

# 행 단위 정보가 담겨 공개하지 않는 산출물 (runs/<이름>/public/에 복사하지 않음)
PRIVATE_FILES = {
    "02_split/split.csv",
    "04_wrapper/cv_folds.csv",
    "05_model/oof_predictions.csv",
    "05_model/holdout_predictions.csv",
    "06_explain/model.joblib",
    "07_release/model.joblib",   # 모델은 promote.py로만 배포
}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (Path,)):
        return str(o)
    raise TypeError(type(o))


class RunContext:
    def __init__(self, run_dir: Path, cfg):
        self.run_dir = run_dir
        self.cfg = cfg

    @cached_property
    def _loaded(self):
        return load_dataset(self.cfg)

    @property
    def df(self) -> pd.DataFrame:
        return self._loaded[0]

    @property
    def data_file(self) -> Path:
        return self._loaded[1]

    def stage_dir(self, stage: str) -> Path:
        d = self.run_dir / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_csv(self, stage: str, name: str, df: pd.DataFrame) -> Path:
        path = self.stage_dir(stage) / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def write_json(self, stage: str, name: str, obj) -> Path:
        path = self.stage_dir(stage) / f"{name}.json"
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                   default=_json_default), encoding="utf-8")
        return path

    def read_json(self, stage: str, name: str):
        path = self.run_dir / stage / f"{name}.json"
        if not path.exists():
            raise StageError(f"선행 단계 결과가 없습니다: {path.relative_to(self.run_dir)}")
        return json.loads(path.read_text(encoding="utf-8"))

    def exists(self, stage: str, name: str) -> bool:
        return (self.run_dir / stage / name).exists()

    def read_csv(self, stage: str, name: str) -> pd.DataFrame:
        path = self.run_dir / stage / f"{name}.csv"
        if not path.exists():
            raise StageError(f"선행 단계 결과가 없습니다: {path.relative_to(self.run_dir)}")
        return pd.read_csv(path, encoding="utf-8-sig")
