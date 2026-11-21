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
