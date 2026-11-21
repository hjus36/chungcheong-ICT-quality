"""04 wrapper — 모델마다 자기 특징 수(k)를 고른다 (B안).

- dev 안에서 그룹 단위 K-fold를 cv.n_repeats번 반복한다(반복마다 seed를 바꿔 다시 섞음).
  fold 배정 하나에 결과가 좌우되지 않도록 모든 반복·fold의 점수를 평균한다.
  매 fold 학습 부분으로 F-test/FDR 순위를 다시 계산한다.
  (순위는 모델과 무관하므로 fold·indicator 조건마다 한 번만 계산해 모든 모델이 공유)
- 모델 × indicator 조건마다 k별 PR-AUC 곡선을 그리고 k_rule로 각자의 k를 고른다.
- 다음 단계로 넘기는 특징 집합은 features.missing_indicators 조건 기준.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from src.design import rank_fdr_candidates
from src.evaluate import fit_predict, metrics
from src.groups import make_groups
from src.models import DISPLAY, is_available
from src.pipeline import fold_table, load_split, repeated_group_folds, used_features
from src.runio import StageError

STAGE = "04_wrapper"
VARIANT = {True: "with_indicator", False: "no_indicator"}


def parse_k_grid(spec, k_max: int) -> list[int]:
    """"1-20, 25-max:5" → [1..20, 25, 30, ..., k_max]. 비어 있으면 1..k_max."""
    if not spec:
        return list(range(1, k_max + 1))
    ks = set()
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        step = 1
        if ":" in tok:
            tok, st = tok.split(":")
            step = int(st)
        if "-" in tok:
            a, b = (x.strip() for x in tok.split("-"))
            a, b = int(a), (k_max if b == "max" else int(b))
        else:
            a = b = k_max if tok == "max" else int(tok)
        ks.update(range(a, min(b, k_max) + 1, step))
    ks.add(k_max)
    return sorted(k for k in ks if 1 <= k <= k_max)


def estimators_of(wc) -> list[str]:
    e = wc.get("estimators") or wc.get("estimator") or "svm"
    return [e] if isinstance(e, str) else list(e)


def choose_k(curve, rule, tol, min_recall, n_folds):
    """one_se: 최고 PR-AUC − 표준오차 이상 중 최소 k / tolerance: 최고의 tol 비율 이내 중 최소 k"""
    best_row = curve.loc[curve["PR_AUC_mean"].idxmax()]
    best = float(best_row["PR_AUC_mean"])
    if rule == "one_se":
        floor = best - float(np.nan_to_num(best_row["PR_AUC_std"])) / math.sqrt(n_folds)
    elif rule == "tolerance":
        floor = best - tol * abs(best)
    else:
        raise StageError(f"지원하지 않는 wrapper.k_rule: {rule}")
    ok = curve[curve["PR_AUC_mean"] >= floor - 1e-12]
    if min_recall is not None:
        ok = ok[ok["Recall_mean"] >= float(min_recall)]
    if ok.empty:
        raise StageError("k 선택 기준과 최소 Recall을 동시에 만족하는 특징 수가 없습니다.")
    return int(ok.iloc[0]["n_features"]), best, int(best_row["n_features"])


def _plot(ctx, curves, chosen_k, models, primary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    n = len(models)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 3.8 * nrows), squeeze=False, sharey=True)
    for ax, m in zip(axes.flat, models):
        for (mm, var), c in curves.items():
            if mm != m:
                continue
            main = var == primary
            ax.plot(c["n_features"], c["PR_AUC_mean"], marker="o", ms=2.5, lw=1.8 if main else 1.0,
                    alpha=1 if main else 0.7, label=var)
            if main:
                ax.fill_between(c["n_features"], c["PR_AUC_mean"] - c["PR_AUC_std"],
                                c["PR_AUC_mean"] + c["PR_AUC_std"], alpha=0.15)
                ax.axvline(chosen_k[m], ls="--", lw=1, color="black", alpha=0.6)
        ax.set_title(f"{DISPLAY[m]} (k={chosen_k[m]})")
        ax.set_xlabel("Number of features")
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("Group-CV PR-AUC (dev)")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Feature-count curves per model (shared F-score ranking, group CV)")
    fig.tight_layout()
    path = ctx.stage_dir(STAGE) / "feature_count_curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def run(ctx):
    cfg = ctx.cfg
    ex = cfg.experiment
    wc, mc = ex["wrapper"], ex["model"]
    alpha = float(ex["filter"].get("fdr_alpha", 0.05))
    params = mc.get("params") or {}
    overrides = wc.get("param_overrides") or {}
    rule = wc.get("k_rule", "tolerance")
    tol = float(wc.get("pr_auc_relative_tolerance", 0.01))

    wanted = estimators_of(wc)
    models = [m for m in wanted if is_available(m)]
    skipped = [m for m in wanted if m not in models]
    for m in skipped:
        print(f"  [경고] {m} 패키지가 없어 제외합니다.")
    if not models:
        raise StageError("사용 가능한 wrapper 모델이 없습니다.")

    dev, _ = load_split(ctx)
    y = dev[cfg.target].astype(int)
    groups = make_groups(dev, cfg)
    X = dev[used_features(ctx)]
    n_splits = int(ex["cv"].get("n_splits", 5))
    n_repeats = int(ex["cv"].get("n_repeats", 1))
    folds = repeated_group_folds(y, groups, n_splits, n_repeats, cfg.seed)
    ctx.write_csv(STAGE, "cv_folds", fold_table(dev.index, groups, folds, n_repeats))

    primary_ind = bool((ex.get("features") or {}).get("missing_indicators", True))
    variants = [primary_ind] + ([not primary_ind] if wc.get("compare_missing_indicators") else [])

    # fold·조건별 순위 (모든 모델 공유) + dev 전체 순위
    rankings, final_rank = {}, {}
    for ind in variants:
        rankings[ind] = [rank_fdr_candidates(X.iloc[tr], y.iloc[tr], alpha, ind)[0]["feature"].tolist()
                         for _, _, tr, _ in folds]
        final_rank[ind] = rank_fdr_candidates(X, y, alpha, ind)[0]
        fr = pd.DataFrame({f"r{rr}_fold{f}": pd.Series(r) for (rr, f, _, _), r in zip(folds, rankings[ind])})
        fr.insert(0, "rank", np.arange(1, len(fr) + 1))
        ctx.write_csv(STAGE, f"fold_rankings_{VARIANT[ind]}", fr)

    rows, curves, chosen_k, feature_sets = [], {}, {}, {}
    for m in models:
        p = {**(params.get(m) or {}), **(overrides.get(m) or {})}
        for ind in variants:
            t0 = time.time()
            var = VARIANT[ind]
            k_max = min(map(len, rankings[ind]))
            if wc.get("max_features"):
                k_max = min(k_max, int(wc["max_features"]))
            if k_max == 0:
                raise StageError(f"[{var}] FDR 기준을 통과한 특징이 없는 fold가 있습니다.")
            ckpt = ctx.stage_dir(STAGE) / f"_ckpt_curve_{m}_{var}.csv"
            recs = []
            grid = [] if ckpt.exists() else parse_k_grid(wc.get("k_grid"), k_max)
            for k in grid:
                per = []
                for (_, _, tr, va), ranked in zip(folds, rankings[ind]):
                    s, pr, _ = fit_predict(m, p, X.iloc[tr], y.iloc[tr], X.iloc[va], ranked[:k], ind, cfg.seed)
                    per.append(metrics(y.iloc[va], s, pr))
                f = pd.DataFrame(per)
                recs.append({"n_features": k,
                             "PR_AUC_mean": f["PR_AUC"].mean(), "PR_AUC_std": f["PR_AUC"].std(ddof=1),
                             "ROC_AUC_mean": f["ROC_AUC"].mean(),
                             "Recall_mean": f["Recall"].mean(), "Recall_std": f["Recall"].std(ddof=1),
                             "Precision_mean": f["Precision"].mean(), "F1_mean": f["F1"].mean()})
            if ckpt.exists():   # 중단 후 --resume: 이미 계산한 곡선 재사용 (같은 실행 = 같은 설정·코드)
                curve = pd.read_csv(ckpt, encoding="utf-8-sig")
            else:
                curve = pd.DataFrame(recs)
                curve.to_csv(ckpt, index=False, encoding="utf-8-sig")
            k, best, best_at = choose_k(curve, rule, tol, wc.get("min_recall"), n_splits)
            curve["selected"] = curve["n_features"] == k
            ctx.write_csv(STAGE, f"curve_{m}_{var}", curve)
            fr = final_rank[ind]
            if len(fr) < k:
                raise StageError(f"[{m}/{var}] dev 전체의 FDR 후보({len(fr)})가 k={k}보다 적습니다.")
            sel = fr.head(k)[["rank", "feature", "best_component", "F_score"]]
            ctx.write_csv(STAGE, f"selected_features_{m}_{var}", sel)
            at = curve.loc[curve["n_features"] == k].iloc[0]
            if ind == primary_ind:
                ctx.write_csv(STAGE, f"selected_features_{m}", sel)
                chosen_k[m] = k
                feature_sets[m] = {"k": k, "features": sel["feature"].tolist(),
                                   "PR_AUC_at_k": float(at["PR_AUC_mean"]),
                                   "PR_AUC_std_at_k": float(at["PR_AUC_std"])}
            curves[(m, var)] = curve
            rows.append({"model": m, "model_name": DISPLAY[m], "variant": var, "primary": ind == primary_ind,
                         "k_evaluated": int(curve["n_features"].max()), "n_grid": len(curve),
                         "best_PR_AUC": best, "best_at_k": best_at, "selected_k": k,
                         "PR_AUC_at_k": float(at["PR_AUC_mean"]), "PR_AUC_std_at_k": float(at["PR_AUC_std"]),
                         "Recall_at_k": float(at["Recall_mean"]),
                         "n_selected_via_indicator": int(sel["best_component"].str.endswith("__missing").sum()),
                         "seconds": round(time.time() - t0, 1)})
            print(f"   {'*' if ind == primary_ind else ' '}{DISPLAY[m]:20s} [{var:14s}] k={k:2d} "
                  f"PR-AUC {at['PR_AUC_mean']:.4f}±{at['PR_AUC_std']:.4f} "
                  f"(최고 {best:.4f} @k={best_at})  {time.time() - t0:.0f}s", flush=True)

    comp = pd.DataFrame(rows)
    ctx.write_csv(STAGE, "wrapper_comparison", comp)

    # 모델별 선택 특징 겹침표
    order = final_rank[primary_ind]["feature"].tolist()
    union = [f for f in order if any(f in s["features"] for s in feature_sets.values())]
    ov = pd.DataFrame({"rank": [order.index(f) + 1 for f in union], "feature": union})
    for m in models:
        ov[m] = ov["feature"].isin(feature_sets[m]["features"])
    ov["n_models"] = ov[models].sum(axis=1)
    ctx.write_csv(STAGE, "feature_overlap", ov)
    ctx.write_json(STAGE, "model_feature_sets", feature_sets)

    fig = _plot(ctx, curves, chosen_k, models, VARIANT[primary_ind])
    ctx.write_json(STAGE, "wrapper_summary", {
        "estimators": models, "skipped": skipped, "cv": f"StratifiedGroupKFold({n_splits}) × {n_repeats}회 반복", "n_splits": n_splits, "n_repeats": n_repeats,
        "dev_rows": len(dev), "dev_groups": int(groups.nunique()), "fdr_alpha": alpha,
        "k_rule": rule, "tolerance": tol, "min_recall": wc.get("min_recall"), "k_grid": wc.get("k_grid"),
        "param_overrides": overrides,
        "primary_variant": VARIANT[primary_ind], "primary_missing_indicators": primary_ind,
        "selected_k": {m: s["k"] for m, s in feature_sets.items()},
        "common_features": ov.loc[ov["n_models"] == len(models), "feature"].tolist(),
        "figure": fig,
    })
    print("  * = 다음 단계로 넘기는 조건 (features.missing_indicators)")
