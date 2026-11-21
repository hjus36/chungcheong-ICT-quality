"""01 eda — 전체 데이터 기술 통계. 문서의 EDA 표를 모두 이 단계에서 생성한다.

탐색용이므로 전체 행을 사용한다. 특징 선택은 split 이후 개발 데이터로만 한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.columns import candidate_columns, resolve_columns
from src.data import target_rows
from src.groups import group_parts, make_groups
from src.quality import constant_features, exact_duplicates, near_duplicates
from src.stats import univariate_table

STAGE = "01_eda"


def _pattern_letters(n):
    return [chr(ord("A") + i) if i < 26 else f"P{i}" for i in range(n)]


def run(ctx):
    cfg = ctx.cfg
    df, y = target_rows(ctx.df, cfg.target)
    roles = resolve_columns(df, cfg)
    cand = candidate_columns(roles)
    X = df[cand]
    alpha = float(cfg.experiment["eda"].get("fdr_alpha", 0.05))
    feat_cfg = cfg.experiment.get("features") or {}
    summary = {}

    # 1) 데이터·타깃 개요
    summary["rows"] = len(df)
    summary["columns"] = df.shape[1]
    summary["target"] = {"name": cfg.target, "n_ok": int((y == 0).sum()),
                         "n_nok": int((y == 1).sum()), "nok_rate": float(y.mean())}
    dtype_counts = df.dtypes.astype(str).value_counts().rename_axis("dtype").reset_index(name="n")
    ctx.write_csv(STAGE, "dtype_counts", dtype_counts)

    # 2) 제외 컬럼 요약 (역할·사유·접두사별)
    excl = (roles[roles["status"] == "excluded"]
            .groupby(["reason", "prefix"], dropna=False)
            .agg(n=("column", "size"), columns=("column", lambda s: ", ".join(s)))
            .reset_index())
    ctx.write_csv(STAGE, "excluded_summary", excl)
    ctx.write_csv(STAGE, "candidate_features",
                  roles[roles["status"] == "candidate"][["column", "prefix", "stage", "dtype"]])
    summary["n_candidates"] = len(cand)
    summary["candidates_by_prefix"] = roles[roles["status"] == "candidate"] \
        .groupby("prefix").size().to_dict()

    # 3) 라벨 분포
    label_prefixes = tuple((cfg.columns.get("roles") or {}).get("label_prefixes") or [])
    label_rows = []
    for col in [c for c in ctx.df.columns if label_prefixes and c.startswith(label_prefixes)]:
        s = ctx.df[col]
        if pd.api.types.is_numeric_dtype(s):
            label_rows.append({"label": col, "n_0": int((s == 0).sum()), "n_1": int((s == 1).sum()),
                               "n_missing": int(s.isna().sum()),
                               "rate_1_observed": float(s.dropna().mean())})
    ctx.write_csv(STAGE, "label_distribution", pd.DataFrame(label_rows))

    # 4) 그룹별 라벨 순도 — 라벨이 그룹마다 고정이면 행 단위 분할 점수는 과대평가된다
    groups = make_groups(df, cfg)
    gp = group_parts(df, cfg)
    g_tab = (pd.DataFrame({"group": groups, "y": y}).join(gp)
             .groupby("group").agg(n=("y", "size"), nok_rate=("y", "mean"))
             .reset_index())
    g_tab["pure"] = g_tab["nok_rate"].isin([0.0, 1.0])
    ctx.write_csv(STAGE, "group_label_purity", g_tab)
    n_mixed = int((~g_tab["pure"]).sum())
    summary["groups"] = {"n_groups": len(g_tab),
                         "pure_ok": int((g_tab["nok_rate"] == 0).sum()),
                         "pure_nok": int((g_tab["nok_rate"] == 1).sum()),
                         "mixed": n_mixed}
    if n_mixed == 0:
        summary["groups"]["warning"] = ("모든 그룹에서 라벨이 고정되어 있습니다. "
                                        "행 단위 무작위 분할은 그룹 식별만으로 점수가 높아질 수 있습니다.")

    # 첫 번째 그룹 키(예: 실험일)별 구성
    first = gp.columns[0]
    by_first = (pd.DataFrame({first: gp[first], "y": y, "group": groups})
                .groupby(first).agg(n=("y", "size"), nok_rate=("y", "mean"),
                                    n_groups=("group", "nunique")).reset_index())
    ctx.write_csv(STAGE, f"composition_by_{first}", by_first)

    # 5) 결측률
    miss = X.isna().sum()
    miss_tab = pd.DataFrame({"feature": cand, "n_missing": miss.values,
                             "missing_rate": (miss / len(X)).values}) \
        .sort_values(["missing_rate", "feature"], ascending=[False, True])
    ctx.write_csv(STAGE, "missing_rates", miss_tab)
    bins = cfg.experiment["eda"].get("missing_bins", [0, 10, 30, 50, 70, 100])
    pct = miss_tab["missing_rate"] * 100
    labels = [f"{a}~{b}%" for a, b in zip(bins[:-1], bins[1:])]
    binned = pd.cut(pct, bins=[bins[0] - 1e-9] + bins[1:], labels=labels, include_lowest=True)
    ctx.write_csv(STAGE, "missing_bins",
                  binned.value_counts().reindex(labels, fill_value=0)
                  .rename_axis("missing_rate_bin").reset_index(name="n_features"))
    summary["n_features_with_missing"] = int((miss > 0).sum())

    # 6) 결측 패턴 (행 단위 결측 조합)
    miss_cols = miss[miss > 0].index.tolist()
    if miss_cols:
        sig = X[miss_cols].isna().apply(lambda r: tuple(r.values), axis=1)
        pat = pd.DataFrame({"sig": sig, "y": y, first: gp[first]})
        counts = pat.groupby("sig").agg(n_rows=("y", "size"), nok_rate=("y", "mean")) \
                    .sort_values("n_rows", ascending=False).reset_index()
        counts["pattern"] = _pattern_letters(len(counts))
        counts["n_missing_features"] = counts["sig"].map(sum)
        counts["missing_features"] = counts["sig"].map(
            lambda s: ", ".join(c for c, m in zip(miss_cols, s) if m))
        name_of = dict(zip(counts["sig"], counts["pattern"]))
        pat["pattern"] = pat["sig"].map(name_of)
        cross = pd.crosstab(pat["pattern"], pat[first]).reset_index()
        ctx.write_csv(STAGE, "missing_patterns",
                      counts.drop(columns="sig")[["pattern", "n_rows", "n_missing_features",
                                                  "nok_rate", "missing_features"]])
        ctx.write_csv(STAGE, f"missing_patterns_by_{first}", cross)

        # 결측 위치가 같은 특징 묶음별: 결측 여부와 NOK
        masks = X[miss_cols].isna()
        key = masks.apply(lambda s: s.values.tobytes())
        rows = []
        for _, cols in key.groupby(key).groups.items():
            cols = list(cols)
            m = masks[cols[0]]
            rows.append({"n_features": len(cols), "missing_rate": float(m.mean()),
                         "nok_observed": float(y[~m].mean()), "nok_missing": float(y[m].mean()),
                         "diff_pp": float((y[m].mean() - y[~m].mean()) * 100),
                         "features": ", ".join(cols)})
        ctx.write_csv(STAGE, "missingness_vs_target",
                      pd.DataFrame(rows).sort_values("missing_rate", ascending=False))

    # 7) 변수 품질 (보고용 — 실제 제거는 filter 단계에서 개발 데이터로 다시 판정)
    nd = feat_cfg.get("near_duplicate") or {}
    const = constant_features(X)
    dup = exact_duplicates(X)
    near = near_duplicates(X, float(nd.get("threshold", 0.99)), int(nd.get("min_periods", 30)))
    ctx.write_csv(STAGE, "constant_features", pd.DataFrame({"feature": const}))
    ctx.write_csv(STAGE, "exact_duplicates", dup)
    ctx.write_csv(STAGE, "near_duplicates", near)
    summary["quality"] = {"constant": len(const), "exact_duplicate_pairs": len(dup),
                          "near_duplicate_pairs": len(near)}

    # 8) 단일 변수 AUC + Mann–Whitney U (FDR 보정)
    uni = univariate_table(X, y)
    ctx.write_csv(STAGE, "univariate", uni)
    if "mwu_q" in uni:
        summary["univariate"] = {"n_tested": int(uni["mwu_p"].notna().sum()),
                                 "n_fdr_significant": int((uni["mwu_q"] < alpha).sum()),
                                 "fdr_alpha": alpha}

    ctx.write_json(STAGE, "eda_summary", summary)
    print(f"  입력 후보 {len(cand)}개, 결측 있는 특징 {summary['n_features_with_missing']}개")
    g = summary["groups"]
    print(f"  그룹 {g['n_groups']}개 (정상 고정 {g['pure_ok']}, 불량 고정 {g['pure_nok']}, 혼합 {g['mixed']})")
    if "warning" in g:
        print(f"  [경고] {g['warning']}")
    print(f"  상수 {len(const)}, 완전중복 쌍 {len(dup)}, 근사중복 쌍 {len(near)}")
    if "univariate" in summary:
        u = summary["univariate"]
        print(f"  Mann–Whitney FDR<{alpha}: {u['n_fdr_significant']}/{u['n_tested']}")
