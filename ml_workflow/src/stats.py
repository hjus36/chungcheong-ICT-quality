"""통계 유틸: FDR 보정, 단일 변수 AUC·Mann–Whitney U."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


def benjamini_hochberg(p_values) -> np.ndarray:
    """BH 보정 p-value. NaN은 NaN으로 유지한다."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return adjusted
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def univariate_table(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """특징별로 관측된 행만 사용해 AUC와 Mann–Whitney U를 계산한다."""
    rows = []
    for col in X.columns:
        mask = X[col].notna()
        x, t = X.loc[mask, col].astype(float), y.loc[mask]
        row = {"feature": col, "n_obs": int(mask.sum()),
               "missing_rate": float(1 - mask.mean())}
        if t.nunique() == 2 and x.nunique() > 1:
            auc = roc_auc_score(t, x)
            x0, x1 = x[t == 0], x[t == 1]
            u, p = mannwhitneyu(x0, x1, alternative="two-sided")
            med0, med1 = float(x0.median()), float(x1.median())
            row.update({
                "auc_raw": float(auc), "auc_strength": float(max(auc, 1 - auc)),
                "median_ok": med0, "median_nok": med1,
                "nok_tendency": ("NOK에서 높음" if med1 > med0
                                 else "NOK에서 낮음" if med1 < med0 else "중앙값 동일"),
                "mwu_U": float(u), "mwu_p": float(p),
            })
        rows.append(row)
    table = pd.DataFrame(rows)
    if "mwu_p" in table:
        table["mwu_q"] = benjamini_hochberg(table["mwu_p"])
        table = table.sort_values("auc_strength", ascending=False, na_position="last")
    return table.reset_index(drop=True)
