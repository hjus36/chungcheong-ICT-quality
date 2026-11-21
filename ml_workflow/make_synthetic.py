"""승격된 release 기준으로 합성 예제 CSV를 만든다 (실측 아님).

    python make_synthetic.py                       # 준비패키지/release → 준비패키지/share
    python make_synthetic.py --release DIR --out DIR

만드는 파일
- synthetic_demo.csv         정상 형식 120행 (공개 데모 기본 자료)
- test_upload.csv            20행 + 검증 오류 행 (업로드 검증 확인용)
- synthetic_definition.json  생성 규칙·범위·해시 (재현용)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

WORKFLOW_DIR = Path(__file__).resolve().parent
REPO = WORKFLOW_DIR.parent


def ranges_of(dictionary, scfg):
    out = {}
    w = float(scfg.get("widen", 0.3))
    over = scfg.get("ranges") or {}
    for f in dictionary["features"]:
        if f["name"] in over:
            lo, hi = float(over[f["name"]]["low"]), float(over[f["name"]]["high"])
        else:
            q05, q95 = f["reference_q05"], f["reference_q95"]
            span = max(q95 - q05, abs(f["reference_median"]) * 0.05, 1e-6)
            lo, hi = q05 - w * span, q95 + w * span
        if f["min"] is not None:
            lo = max(lo, float(f["min"]))
        if f["max"] is not None:
            hi = min(hi, float(f["max"]))
        out[f["name"]] = (lo, hi)
    return out


def make_rows(n, start, rng, dictionary, scfg, rngs, id_prefix, offset=0):
    step = int(scfg["cycle_seconds"]) + int(scfg["gap_seconds"])
    t0 = pd.Timestamp(start) + pd.Timedelta(seconds=step * offset)
    starts = [t0 + pd.Timedelta(seconds=step * i) for i in range(n)]
    df = pd.DataFrame({
        "part_id": [f"{id_prefix}-{offset + i + 1:04d}" for i in range(n)],
        "cycle_start_at": [s.strftime("%Y-%m-%d %H:%M:%S") for s in starts],
        "cycle_end_at": [(s + pd.Timedelta(seconds=int(scfg["cycle_seconds"]))).strftime("%Y-%m-%d %H:%M:%S")
                         for s in starts],
        "data_source": dictionary["synthetic_source"],
        "material": rng.choice(scfg.get("materials") or ["PP"], size=n),
    })
    for f in dictionary["features"]:
        lo, hi = rngs[f["name"]]
        v = rng.uniform(lo, hi, size=n).round(4)
        if not f["required"]:
            v = np.where(rng.random(n) < float(scfg.get("optional_missing_rate", 0)), np.nan, v)
        for prefix, mat in (scfg.get("only_for_material") or {}).items():
            if f["name"].startswith(prefix) and not f["required"]:
                v = np.where(df["material"].to_numpy() == mat, v, np.nan)
        df[f["name"]] = v
    return df


def add_label(df, bundle, rng):
    p = bundle["pipeline"].predict_proba(df[bundle["features"]])[:, 1]
    df["actual_label"] = (rng.random(len(df)) < p).astype(int)
    return df


def error_rows(base, dictionary, kinds):
    rows = []
    feats = dictionary["features"]
    req = next(f for f in feats if f["required"])
    for i, kind in enumerate(kinds):
        r = base.iloc[i % len(base)].copy()
        r["part_id"] = f"ERR-{i + 1:02d}-{kind}"
        if kind == "duplicate_part_id":
            r["part_id"] = base.iloc[0]["part_id"]
        elif kind == "out_of_range":
            f = next((f for f in feats if f["max"] is not None), None) or feats[0]
            r[f["name"]] = (f["max"] or 0) * 10 + 1e6
        elif kind == "missing_required":
            r[req["name"]] = np.nan
        elif kind == "bad_timestamp":
            r["cycle_start_at"] = "2026-13-45 99:99:99"
        elif kind == "bad_source":
            r["data_source"] = "unknown_source"
        elif kind == "bad_label":
            r["actual_label"] = 2
        rows.append(r)
    return pd.DataFrame(rows)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=Path, default=REPO / "준비패키지" / "release")
    ap.add_argument("--out", type=Path, default=REPO / "준비패키지" / "share")
    ap.add_argument("--config", type=Path, default=WORKFLOW_DIR / "config" / "synthetic.yaml")
    a = ap.parse_args(argv)

    scfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    dictionary = json.loads((a.release / "data_dictionary.json").read_text(encoding="utf-8"))
    bundle = joblib.load(a.release / "model.joblib")
    rng = np.random.default_rng(int(scfg["seed"]))
    rngs = ranges_of(dictionary, scfg)

    demo = make_rows(int(scfg["rows"]), scfg["start_at"], rng, dictionary, scfg, rngs, "SYN")
    test = make_rows(int(scfg["test_upload_rows"]), scfg["start_at"], rng, dictionary, scfg, rngs, "UPL",
                     offset=int(scfg["rows"]))
    if (scfg.get("label") or {}).get("enabled", True):
        demo, test = add_label(demo, bundle, rng), add_label(test, bundle, rng)
    test = pd.concat([test, error_rows(test, dictionary, scfg.get("error_rows") or [])], ignore_index=True)

    a.out.mkdir(parents=True, exist_ok=True)
    demo.to_csv(a.out / "synthetic_demo.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    test.to_csv(a.out / "test_upload.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    definition = {
        "notice": "합성 자료이며 실측이 아니다. 원본 행을 복사하거나 섞지 않았다. actual_label은 모델 위험도로 뽑은 가상값이다.",
        "release_version": dictionary["version"], "data_source": dictionary["synthetic_source"],
        "config": scfg, "ranges": {k: {"low": v[0], "high": v[1]} for k, v in rngs.items()},
        "files": {"synthetic_demo.csv": {"rows": len(demo), "sha256": sha(a.out / "synthetic_demo.csv")},
                  "test_upload.csv": {"rows": len(test), "error_rows": scfg.get("error_rows") or [],
                                      "sha256": sha(a.out / "test_upload.csv")}},
    }
    (a.out / "synthetic_definition.json").write_text(json.dumps(definition, ensure_ascii=False, indent=2),
                                                     encoding="utf-8", newline="\n")
    pos = int(demo["actual_label"].sum()) if "actual_label" in demo else 0
    print(f"✓ 합성 예제 → {a.out}: demo {len(demo)}행(가상 불량 {pos}), test_upload {len(test)}행")


if __name__ == "__main__":
    main()
