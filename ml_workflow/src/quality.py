"""변수 품질 검사: 전체 결측, 상수, 완전 중복, 근사 중복."""
from __future__ import annotations

import numpy as np
import pandas as pd


def all_missing(X: pd.DataFrame) -> list[str]:
    return X.columns[X.isna().all()].tolist()


def constant_features(X: pd.DataFrame) -> list[str]:
    """관측값의 고유값이 1개 이하인 특징(결측은 세지 않음)."""
    return [c for c in X.columns if X[c].nunique(dropna=True) <= 1]


def exact_duplicates(X: pd.DataFrame) -> pd.DataFrame:
    """값(결측 위치 포함)이 완전히 같은 쌍. 앞쪽 컬럼을 유지 대상으로 표시한다."""
    values = X.to_numpy(dtype=float)
    buckets: dict[bytes, list[int]] = {}
    rows = []
    for j, col in enumerate(X.columns):
        v = values[:, j]
        key = np.nan_to_num(v, nan=np.inf).tobytes()
        match = next((k for k in buckets.get(key, [])
                      if np.array_equal(values[:, k], v, equal_nan=True)), None)
        if match is None:
            buckets.setdefault(key, []).append(j)
        else:
            rows.append({"kept": X.columns[match], "dropped": col})
    return pd.DataFrame(rows, columns=["kept", "dropped"])


def near_duplicates(X: pd.DataFrame, threshold: float,
                    min_periods: int = 30) -> pd.DataFrame:
    """|Spearman rho| >= threshold 인 쌍 (컬럼 순서 기준 a가 앞)."""
    cols = X.columns.tolist()
    corr = X.corr(method="spearman", min_periods=min_periods)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rho = corr.loc[a, b]
            if pd.notna(rho) and abs(rho) >= threshold:
                n = int((X[a].notna() & X[b].notna()).sum())
                rows.append({"a": a, "b": b, "spearman_rho": float(rho), "n_obs": n})
    return pd.DataFrame(rows, columns=["a", "b", "spearman_rho", "n_obs"])


def greedy_drop(pairs: pd.DataFrame, order: list[str]) -> dict[str, str]:
    """쌍 목록에서 앞쪽 컬럼을 남기고 뒤쪽을 제거한다. {제거 컬럼: 유지 파트너}"""
    pos = {c: i for i, c in enumerate(order)}
    dropped: dict[str, str] = {}
    for a, b in sorted(zip(pairs["a"], pairs["b"]),
                       key=lambda p: (pos[p[0]], pos[p[1]])):
        if a in dropped or b in dropped:
            continue
        dropped[b] = a
    return dropped


def select_features(X: pd.DataFrame, feat_cfg: dict) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """설정에 따라 품질 기준으로 특징을 거른다.

    반환: (유지 특징, 제거 표[column, reason, partner], 근사중복 쌍 표)
    """
    kept = X.columns.tolist()
    drops = []

    def drop(cols, reason, partner=None):
        for c in cols:
            if c in kept:
                kept.remove(c)
                drops.append({"column": c, "reason": reason,
                              "partner": (partner or {}).get(c)})

    if feat_cfg.get("drop_all_missing", True):
        drop(all_missing(X[kept]), "전체 결측")
    if feat_cfg.get("drop_constant", True):
        drop(constant_features(X[kept]), "상수(관측값 1종)")
    if feat_cfg.get("drop_exact_duplicates", True):
        dup = exact_duplicates(X[kept])
        drop(dup["dropped"].tolist(), "완전 중복",
             dict(zip(dup["dropped"], dup["kept"])))

    nd_cfg = feat_cfg.get("near_duplicate") or {}
    near = pd.DataFrame(columns=["a", "b", "spearman_rho", "n_obs"])
    if nd_cfg:
        near = near_duplicates(X[kept], float(nd_cfg.get("threshold", 0.99)),
                               int(nd_cfg.get("min_periods", 30)))
        if nd_cfg.get("action", "report") == "drop" and len(near):
            mapping = greedy_drop(near, kept)
            drop(list(mapping), f"근사 중복(|rho|>={nd_cfg.get('threshold')})", mapping)

    return kept, pd.DataFrame(drops, columns=["column", "reason", "partner"]), near
