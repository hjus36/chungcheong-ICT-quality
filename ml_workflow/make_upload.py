"""원본 데이터셋 파일을 웹 업로드 형식 CSV로 바꾼다.

    python make_upload.py --input data/dataset_V2.parquet --output ../준비패키지/local/upload.csv
    python make_upload.py --input data/dataset_V2.parquet --holdout-of im_v2_r3 --output …   # holdout 행만

- 열 구성·이름은 승격된 데이터 사전(기본: 준비패키지/release/data_dictionary.json)을 따른다.
- 원본 열 이름 대응은 config/columns.yaml의 service 블록에서 읽는다.
- --holdout-of를 주지 않으면 학습에 쓴 행이 섞일 수 있다. 그 결과로 성능을 판단하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

WORKFLOW_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKFLOW_DIR))
from src.config import load_config  # noqa: E402
from src.contract import check_service_input, to_service_input  # noqa: E402
from src.data import file_sha256  # noqa: E402
from src.errors import StageError  # noqa: E402

RUNS_DIR = Path(os.environ.get("ICT_RUNS_DIR") or WORKFLOW_DIR / "runs")
DEFAULT_RELEASE = WORKFLOW_DIR.parent / "준비패키지" / "release"


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in (".csv", ".txt"):
        return pd.read_csv(path, low_memory=False)
    raise StageError(f"parquet 또는 csv만 읽을 수 있습니다: {path.name}")


def holdout_rows(df: pd.DataFrame, input_path: Path, run: str) -> pd.DataFrame:
    run_dir = RUNS_DIR / run
    split_path = run_dir / "02_split" / "split.csv"
    if not split_path.exists():
        raise StageError(f"분할표가 없습니다: {split_path}")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("data_sha256") and manifest["data_sha256"] != file_sha256(input_path):
        raise StageError(f"입력 파일이 {run} 실행에 쓴 데이터와 다릅니다 (SHA-256 불일치).")
    split = pd.read_csv(split_path)
    return df.iloc[split.loc[split["split"] == "holdout", "row_index"].to_numpy()]


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--release", type=Path, default=DEFAULT_RELEASE, help="data_dictionary.json이 있는 폴더")
    ap.add_argument("--holdout-of", metavar="RUN", help="이 실행의 holdout 행만 변환")
    a = ap.parse_args(argv)
    try:
        cfg = load_config()
        dictionary = json.loads((a.release / "data_dictionary.json").read_text(encoding="utf-8"))
        df = read_any(a.input)
        if a.holdout_of:
            df = holdout_rows(df, a.input, a.holdout_of)
        out = to_service_input(df, cfg, dictionary)
    except (StageError, FileNotFoundError) as e:
        print(f"✗ {e}")
        return 2
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False, encoding="utf-8-sig", lineterminator="\n")
    pos = int(out["actual_label"].sum()) if "actual_label" in out else None
    print(f"✓ {len(out)}행 → {a.output} (기준 데이터 사전 {dictionary['version']}, 불량 {pos})")
    if not a.holdout_of:
        print("  [주의] 학습에 쓴 행이 섞여 있을 수 있습니다. 성능 판단에 쓰지 마세요. (--holdout-of로 holdout만 변환)")
    for p in check_service_input(out, dictionary):
        print(f"  [경고] 서비스에서 제외될 행: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
