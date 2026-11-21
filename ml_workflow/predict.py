"""학습된 하네스 모델로 새 데이터의 품질위험도를 예측한다 (웹 MVP 연동용).

사용 예:
    python predict.py runs/20261121-081834_full new_data.csv -o predictions.csv
입력은 .csv 또는 .parquet. model_card.json의 특징 컬럼이 모두 있어야 한다(결측값은 허용).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("predictions.csv"))
    ap.add_argument("--keep", nargs="*", default=["MET_MachineCycleID", "MET_JobCycleID", "MET_TimeCycleStart"],
                    help="결과에 함께 남길 식별 컬럼")
    a = ap.parse_args(argv)

    d = a.run_dir / "06_explain"
    card = json.loads((d / "model_card.json").read_text(encoding="utf-8"))
    model = joblib.load(d / "model.joblib")
    df = pd.read_parquet(a.input) if a.input.suffix == ".parquet" else pd.read_csv(a.input, low_memory=False)

    missing = [f for f in card["features"] if f not in df.columns]
    if missing:
        sys.exit(f"입력에 필요한 특징 컬럼이 없습니다: {missing}")
    score = model.predict_proba(df[card["features"]])[:, 1] if hasattr(model, "predict_proba") \
        else model.decision_function(df[card["features"]])
    b = card["bands"]
    band = np.where(score >= b["defect"], b["labels"][2], np.where(score >= b["caution"], b["labels"][1], b["labels"][0]))

    out = df[[c for c in a.keep if c in df.columns]].copy()
    out["risk_score"] = score
    out["risk_band"] = band
    for f in card["features"]:
        lo, hi = card["normal_ranges"][f]
        out[f"{f}__out_of_normal"] = (df[f] < lo) | (df[f] > hi)
    out.to_csv(a.output, index=False, encoding="utf-8-sig")
    print(f"{len(out)}행 예측 → {a.output}")
    print(out["risk_band"].value_counts().reindex(b["labels"], fill_value=0).to_string())


if __name__ == "__main__":
    main()
