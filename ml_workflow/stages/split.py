"""02 split — Holdout 분할을 한 번 정하고 저장한다.

이후 단계(filter, wrapper, model)는 모두 이 파일을 읽어 같은 분할을 쓴다.
group 전략: 그룹 단위로 나누되, 그룹의 NOK 여부(다수 라벨)로 층화한다.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import target_rows
from src.groups import make_groups
from src.runio import StageError

STAGE = "02_split"


def run(ctx):
    cfg = ctx.cfg
    df, y = target_rows(ctx.df, cfg.target)
    groups = make_groups(df, cfg)
    sc = cfg.experiment.get("split") or {}
    strategy = sc.get("strategy", "group")
    size = float(sc.get("holdout_size", 0.2))

    if strategy == "group":
        g = pd.DataFrame({"group": groups, "y": y}).groupby("group")["y"].mean()
        g_label = (g >= 0.5).astype(int)
        try:
            _, hold_groups = train_test_split(g.index.to_numpy(), test_size=size,
                                              random_state=cfg.seed, stratify=g_label)
        except ValueError as e:
            raise StageError(f"그룹 층화 분할 실패: {e}")
        is_hold = groups.isin(set(hold_groups))
    elif strategy == "random":
        _, hold_idx = train_test_split(df.index.to_numpy(), test_size=size,
                                       random_state=cfg.seed, stratify=y)
        is_hold = pd.Series(df.index.isin(hold_idx), index=df.index)
    else:
        raise StageError(f"지원하지 않는 split.strategy: {strategy}")

    split = pd.DataFrame({"row_index": df.index, "group": groups.values,
                          "split": ["holdout" if h else "dev" for h in is_hold],
                          cfg.target: y.values})
    for part in ("dev", "holdout"):
        if split.loc[split["split"] == part, cfg.target].nunique() < 2:
            raise StageError(f"{part}에 한 클래스만 있습니다. holdout_size나 seed를 조정하세요.")
    ctx.write_csv(STAGE, "split", split)

    summ = (split.groupby("split").agg(rows=(cfg.target, "size"), nok_rate=(cfg.target, "mean"),
                                       n_groups=("group", "nunique")).reset_index())
    overlap = set(split.loc[split.split == "dev", "group"]) & set(split.loc[split.split == "holdout", "group"])
    ctx.write_json(STAGE, "split_summary", {
        "strategy": strategy, "holdout_size": size, "seed": cfg.seed,
        "parts": summ.to_dict("records"), "groups_in_both": len(overlap),
        "holdout_groups": sorted(split.loc[split.split == "holdout", "group"].unique()),
    })
    for r in summ.to_dict("records"):
        print(f"  {r['split']:8s} 행 {r['rows']:4d}, 그룹 {r['n_groups']:3d}, NOK {r['nok_rate']:.1%}")
    print(f"  dev·holdout에 모두 속한 그룹: {len(overlap)}개")
