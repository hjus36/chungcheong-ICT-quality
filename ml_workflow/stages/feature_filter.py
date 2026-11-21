"""03 filter — 개발(dev) 데이터만으로 품질 검사 + F-test/FDR 순위.

Holdout 행은 이 단계에서 사용하지 않는다.
"""
from __future__ import annotations

import pandas as pd

from src.columns import candidate_columns, resolve_columns
from src.design import rank_fdr_candidates
from src.quality import select_features

STAGE = "03_filter"


def run(ctx):
    cfg = ctx.cfg
    split = ctx.read_csv("02_split", "split")
    dev_idx = split.loc[split["split"] == "dev", "row_index"].to_numpy()
    df = ctx.df.loc[dev_idx]
    y = df[cfg.target].astype(int)

    roles = resolve_columns(ctx.df, cfg)
    cand = candidate_columns(roles)
    feat_cfg = cfg.experiment.get("features") or {}
    kept, drops, near = select_features(df[cand], feat_cfg)

    feature_set = pd.DataFrame({"feature": cand})
    feature_set = feature_set.merge(drops, how="left", left_on="feature", right_on="column") \
                             .drop(columns="column")
    feature_set["used"] = feature_set["reason"].isna()
    feature_set["reason"] = feature_set["reason"].fillna("사용")
    ctx.write_csv(STAGE, "feature_set", feature_set)
    ctx.write_csv(STAGE, "near_duplicates_dev", near)

    alpha = float(cfg.experiment["filter"].get("fdr_alpha", 0.05))
    indicators = bool(feat_cfg.get("missing_indicators", True))
    ranking, table = rank_fdr_candidates(df[kept], y, alpha, indicators)
    miss_rate = df[kept].isna().mean()
    table["feature_missing_rate"] = table["feature"].map(miss_rate)
    ctx.write_csv(STAGE, "filter_scores", table)
    ctx.write_csv(STAGE, "filter_ranking", ranking)

    summary = {
        "dev_rows": len(df), "fdr_alpha": alpha, "missing_indicators": indicators,
        "n_candidates": len(cand), "n_used": len(kept),
        "dropped_by_reason": drops["reason"].value_counts().to_dict(),
        "near_duplicate_action": (feat_cfg.get("near_duplicate") or {}).get("action", "report"),
        "near_duplicate_pairs_dev": len(near),
        "n_components": len(table),
        "n_components_value": int((table["kind"] == "value").sum()),
        "n_components_missing": int((table["kind"] == "missing").sum()),
        "n_components_passed": int(table["passed"].sum()),
        "n_features_ranked": len(ranking),
        "top10": ranking.head(10)["feature"].tolist(),
    }
    ctx.write_json(STAGE, "filter_summary", summary)
    print(f"  dev {len(df)}행 | 후보 {len(cand)} → 사용 {len(kept)} "
          f"(제거: {summary['dropped_by_reason'] or '없음'})")
    print(f"  평가 항목 {len(table)}개 (값 {summary['n_components_value']}, "
          f"결측 indicator {summary['n_components_missing']}) → FDR<{alpha} 통과 "
          f"{summary['n_components_passed']}개")
    print(f"  순위에 오른 특징 {len(ranking)}개, 상위 5: {summary['top10'][:5]}")
