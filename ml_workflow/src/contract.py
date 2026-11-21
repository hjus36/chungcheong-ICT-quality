"""서비스 계약: 특징 단위·범위·설명, 지표 이름 변환."""
from __future__ import annotations

import re

import numpy as np

METRIC_KEYS = {"n": "count", "n_pos": "positive_count", "TP": "tp", "FN": "fn", "FP": "fp", "TN": "tn",
               "PR_AUC": "average_precision", "ROC_AUC": "roc_auc", "Recall": "recall", "Precision": "precision",
               "F1": "f1", "F2": "f2", "Balanced_Acc": "balanced_accuracy", "Alert_rate": "alert_rate",
               "Brier": "brier", "threshold": "alert_threshold"}


def feature_info(cfg, feature: str) -> dict:
    """columns.yaml feature_info 규칙으로 단위·허용 범위·설명을 만든다."""
    fi = cfg.columns.get("feature_info") or {}
    rule = next((r for r in fi.get("rules", []) if re.search(r["pattern"], feature)), {})
    return {"unit": rule.get("unit", ""), "min": rule.get("min"), "max": rule.get("max"),
            "description": (fi.get("descriptions") or {}).get(feature, "")}


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def lower_metrics(m: dict) -> dict:
    """하네스 지표 이름 → 서비스·화면이 쓰는 소문자 키 (+accuracy)."""
    out = {METRIC_KEYS.get(k, k): _clean(v) for k, v in m.items()}
    if all(k in out for k in ("tp", "tn", "count")) and out["count"]:
        out["accuracy"] = (out["tp"] + out["tn"]) / out["count"]
    return out


# ---------------------------------------------------------------- 원본 → 서비스 업로드 형식
def _source_map(cfg, dictionary: dict) -> dict:
    """서비스 항목 → 원본 컬럼. 데이터 사전의 source_column을 쓰고, 없으면 columns.yaml service.meta."""
    meta_cfg = (cfg.columns.get("service") or {}).get("meta") or {}
    out = {}
    for m in dictionary.get("meta", []) + dictionary.get("optional", []):
        src = m.get("source_column") or meta_cfg.get(m["name"])
        if src:
            out[m["name"]] = src
    return out


def service_columns(dictionary: dict) -> list[str]:
    """업로드 파일의 열 순서: 메타 → 특징 → 선택 항목."""
    return ([m["name"] for m in dictionary["meta"]] + [f["name"] for f in dictionary["features"]]
            + [m["name"] for m in dictionary.get("optional", [])])


def to_service_input(df, cfg, dictionary: dict):
    """원본 데이터셋 행을 웹 업로드 형식(데이터 사전 기준)으로 바꾼다.

    - 열 이름 대응은 데이터 사전의 source_column(없으면 columns.yaml service.meta)에서 읽는다.
    - 특징 값은 그대로 둔다(단위 변환·결측 대체 없음, 결측은 빈칸).
    - data_source는 원본에 없으므로 데이터 사전의 실측 출처 값을 넣는다.
    """
    import pandas as pd

    from src.errors import StageError

    src = _source_map(cfg, dictionary)
    features = [f["name"] for f in dictionary["features"]]
    needed = [src[n] for n in ("part_id", "cycle_start_at", "cycle_end_at") if n in src] + features
    needed += [c for n, c in src.items() if n not in ("part_id", "cycle_start_at", "cycle_end_at")]
    missing = sorted({c for c in needed if c not in df.columns})
    if missing:
        raise StageError(f"원본에 없는 열: {missing} — columns.yaml의 service.meta 또는 데이터 사전을 확인하세요.")
    out = pd.DataFrame(index=df.index)
    for name in service_columns(dictionary):
        if name == "data_source":
            out[name] = dictionary["data_source"]
        elif name in ("cycle_start_at", "cycle_end_at"):
            t = pd.to_datetime(df[src[name]], errors="coerce")
            if t.isna().any():
                raise StageError(f"{src[name]}에 시각으로 읽을 수 없는 값이 {int(t.isna().sum())}개 있습니다.")
            out[name] = t.dt.strftime("%Y-%m-%d %H:%M:%S")
        elif name == "part_id":
            v = df[src[name]]
            if pd.api.types.is_float_dtype(v) and (v.dropna() % 1 == 0).all():
                v = v.astype("Int64")
            out[name] = v.astype(str)
        elif name in features:
            out[name] = pd.to_numeric(df[name], errors="coerce")
        elif name == ((cfg.columns.get("service") or {}).get("label_column") or "actual_label"):
            col = src.get(name) or cfg.target
            out[name] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif name in src:
            out[name] = df[src[name]]
    return out.reset_index(drop=True)


def check_service_input(out, dictionary: dict) -> list[str]:
    """서비스가 행을 제외하게 될 문제를 미리 찾는다 (형식 검사). 문제가 없으면 빈 목록."""
    import pandas as pd

    problems = []
    dup = out["part_id"].duplicated(keep=False)
    if dup.any():
        problems.append(f"part_id 중복 {int(dup.sum())}행")
    start = pd.to_datetime(out["cycle_start_at"], errors="coerce")
    end = pd.to_datetime(out["cycle_end_at"], errors="coerce")
    if start.isna().any() or end.isna().any():
        problems.append("시각 형식 오류")
    elif (end < start).any():
        problems.append(f"종료 시각이 시작보다 빠른 행 {int((end < start).sum())}개")
    for f in dictionary["features"]:
        v = pd.to_numeric(out[f["name"]], errors="coerce")
        if f["required"] and v.isna().any():
            problems.append(f"필수 특징 {f['name']} 결측 {int(v.isna().sum())}행")
        lo, hi = f.get("min"), f.get("max")
        bad = ((v < lo) if lo is not None else False) | ((v > hi) if hi is not None else False)
        if getattr(bad, "any", lambda: False)():
            problems.append(f"{f['name']} 허용 범위({lo}~{hi}) 밖 {int(bad.sum())}행")
    if "actual_label" in out:
        lab = out["actual_label"].dropna()
        if not lab.isin([0, 1]).all():
            problems.append("actual_label에 0/1이 아닌 값")
    return problems
