"""단계 간 공유: 분할 읽기, 사용 특징, 그룹 CV fold."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .errors import StageError


def load_split(ctx):
    """(dev DataFrame, holdout DataFrame) — 02_split 결과 기준."""
    split = ctx.read_csv("02_split", "split")
    dev_idx = split.loc[split["split"] == "dev", "row_index"].to_numpy()
    hold_idx = split.loc[split["split"] == "holdout", "row_index"].to_numpy()
    return ctx.df.loc[dev_idx], ctx.df.loc[hold_idx]


def used_features(ctx) -> list[str]:
    fs = ctx.read_csv("03_filter", "feature_set")
    return fs.loc[fs["used"].astype(bool), "feature"].tolist()


def group_folds(y: pd.Series, groups: pd.Series, n_splits: int, seed: int):
    """그룹 단위 층화 K-fold. 위치 인덱스 (train, valid) 목록을 반환한다."""
    n_groups = groups.nunique()
    if n_groups < n_splits:
        raise StageError(f"그룹 수({n_groups})가 fold 수({n_splits})보다 적습니다.")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(len(y)), y.to_numpy(), groups.to_numpy()))
    for i, (_, va) in enumerate(folds, 1):
        if y.iloc[va].nunique() < 2:
            raise StageError(f"seed {seed}의 {i}번째 fold 검증 데이터에 한 클래스만 있습니다. "
                             "cv.n_splits를 줄이거나 seed를 바꾸세요.")
    return folds


def repeated_group_folds(y, groups, n_splits: int, n_repeats: int, seed: int):
    """반복 그룹 K-fold. [(repeat, fold, train_pos, valid_pos), ...] — repeat마다 seed+repeat로 다시 섞는다."""
    out = []
    for r in range(n_repeats):
        for f, (tr, va) in enumerate(group_folds(y, groups, n_splits, seed + r), 1):
            out.append((r + 1, f, tr, va))
    return out


def fold_table(index, groups, folds, n_repeats) -> pd.DataFrame:
    tab = pd.DataFrame({"row_index": index, "group": np.asarray(groups)})
    for r in range(1, n_repeats + 1):
        col = np.zeros(len(tab), dtype=int)
        for rr, f, _, va in folds:
            if rr == r:
                col[va] = f
        tab[f"fold_r{r}"] = col
    return tab


def folds_from_table(tab: pd.DataFrame):
    """cv_folds.csv(행 순서 = dev 순서) → repeated_group_folds와 같은 형식."""
    out = []
    reps = sorted(int(c[6:]) for c in tab.columns if c.startswith("fold_r"))
    for r in reps:
        ids = tab[f"fold_r{r}"].to_numpy()
        for f in sorted(np.unique(ids)):
            out.append((r, int(f), np.flatnonzero(ids != f), np.flatnonzero(ids == f)))
    return out
