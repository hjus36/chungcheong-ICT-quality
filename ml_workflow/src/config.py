"""설정 파일 로드."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = WORKFLOW_DIR / "config"
CONFIG_FILES = ("columns.yaml", "experiment.yaml", "release.yaml")
CONFIG_ENV = "ICT_CONFIG_DIR"     # 테스트 등에서 다른 설정 폴더를 쓸 때
DATA_ENV = "ICT_DATA_PATH"


@dataclass
class Config:
    columns: dict
    experiment: dict
    release: dict
    config_dir: Path

    @property
    def target(self) -> str:
        return self.columns["target"]

    @property
    def data_path(self) -> Path:
        """환경변수 ICT_DATA_PATH가 있으면 우선, 없으면 experiment.yaml의 data.path."""
        env = os.environ.get(DATA_ENV)
        if env:
            return Path(env).expanduser().resolve()
        return (WORKFLOW_DIR / self.experiment["data"]["path"]).resolve()

    @property
    def fallback_path(self) -> Path | None:
        fb = self.experiment["data"].get("fallback_csv")
        return (WORKFLOW_DIR / fb).resolve() if fb else None

    @property
    def expected_sha256(self) -> str | None:
        return (self.experiment.get("data") or {}).get("expected_sha256")

    @property
    def seed(self) -> int:
        return int(self.experiment.get("seed", 42))

    def files(self) -> list[Path]:
        return [self.config_dir / name for name in CONFIG_FILES]

    def sha256(self) -> str:
        h = hashlib.sha256()
        for p in self.files():
            h.update(p.read_bytes())
        return h.hexdigest()


def load_config(config_dir: Path | None = None) -> Config:
    config_dir = Path(config_dir or os.environ.get(CONFIG_ENV) or CONFIG_DIR)
    loaded = {}
    for name in CONFIG_FILES:
        with open(config_dir / name, encoding="utf-8") as f:
            loaded[name] = yaml.safe_load(f) or {}
    return Config(columns=loaded["columns.yaml"], experiment=loaded["experiment.yaml"],
                  release=loaded["release.yaml"], config_dir=config_dir)
