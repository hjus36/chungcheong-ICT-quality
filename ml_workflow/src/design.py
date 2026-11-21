"""학습 데이터 기준 결측 대체 + F-test/FDR 순위 계산.

Wrapper·모델 단계에서도 이 함수를 그대로 재사용한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer

from .stats import benjamini_hochberg


def make_design_matrices(X_train: pd.DataFrame, X_valid: pd.DataFrame,
                         missing_indicators: bool = True):
    """학습 데이터 중앙값으로 대체하고, 필요하면 결측 indicator를 붙인다.

    반환: (train 행렬, valid 행렬, components 표, {원본 특징: [행렬 열 위치]})
    """
    kept = X_train.columns[~X_train.isna().all()].tolist()
    if not kept:
        raise ValueError("학습 데이터에 관측값이 있는 특징이 없습니다.")
    imputer = SimpleImputer(strategy="median", add_indicator=missing_indicators)
    train_m = imputer.fit_transform(X_train[kept])
    valid_m = imputer.transform(X_valid[kept])

    grouped = {f: [i] for i, f in enumerate(kept)}
    comps = [{"component": f, "feature": f, "kind": "value"} for f in kept]
    if missing_indicators and imputer.indicator_ is not None:
        for offset, src in enumerate(imputer.indicator_.features_):
            f = kept[src]
            grouped[f].append(len(kept) + offset)
            comps.append({"component": f"{f}__missing", "feature": f, "kind": "missing"})
    return train_m, valid_m, pd.DataFrame(comps), grouped


def rank_fdr_candidates(X: pd.DataFrame, y: pd.Series, alpha: float,
                        missing_indicators: bool = True):
    """값 열과 결측 indicator를 각각 F-test → BH 보정.

    둘 중 하나라도 q < alpha 이면 원본 특징을 후보로 두고,
    통과한 항목 중 가장 큰 F 점수로 순위를 매긴다.
    반환: (순위 표, 전체 항목 점수 표)
    """
    matrix, _, comps, _ = make_design_matrices(X, X, missing_indicators)
    variable = np.nanvar(matrix, axis=0) > 0
    table = comps.copy()
    table["F_score"] = np.nan
    table["p_value"] = np.nan
    if variable.any():
        f, p = f_classif(matrix[:, variable], y)
        table.loc[variable, "F_score"] = f
        table.loc[variable, "p_value"] = p
    table["FDR_q_value"] = benjamini_hochberg(table["p_value"])
    table["passed"] = table["FDR_q_value"] < alpha
    table["note"] = np.where(variable, "", "상수 열(F-test 불가)")

    passed = table[table["passed"]]
    idx = passed.groupby("feature")["F_score"].idxmax()
    ranking = (passed.loc[idx, ["feature", "component", "F_score", "FDR_q_value"]]
               .rename(columns={"component": "best_component"})
               .sort_values(["F_score", "feature"], ascending=[False, True])
               .reset_index(drop=True))
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    table = table.sort_values("F_score", ascending=False, na_position="last").reset_index(drop=True)
    return ranking, table
