"""06 explain — 선택 모델의 주요 영향변수·정상범위 + MVP용 모델 패키지.

- 영향변수: holdout permutation importance (PR-AUC 감소량). 원본 특징을 섞으면 결측 indicator도 함께 바뀐다.
- 영향 방향: dev에서 특징값과 CV 예측 위험도(OOF)의 Spearman 상관 부호.
- 정상범위: dev의 정상(0) 샘플 기준 백분위 구간. 그룹 키 일부(실험일)별 구간도 함께.
- 패키지: dev 전체로 학습한 선택 모델(model.joblib) + model_card.json(특징·위험 구간 기준).
"""
from __future__ import annotations

import json
import platform
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from src.evaluate import build_pipeline, rank_metrics, scores_of
from src.groups import group_parts
from src.pipeline import load_split

STAGE = "06_explain"


def _plot(ctx, imp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    d = imp.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.5))
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], capsize=3)
    ax.set_xlabel("Holdout PR-AUC decrease when permuted")
    ax.set_title("Permutation importance (selected model)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = ctx.stage_dir(STAGE) / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def run(ctx):
    cfg = ctx.cfg
    ex = cfg.experiment
    ec = ex.get("explain") or {}
    th = ctx.read_json("05_model", "thresholds")
    m, feats, ind = th["model"], th["features"], bool(th["missing_indicators"])
    params = (ex["model"].get("params") or {}).get(m)
    lo, hi = ec.get("normal_range_percentiles", [5, 95])

    dev, hold = load_split(ctx)
    y, yh = dev[cfg.target].astype(int), hold[cfg.target].astype(int)

    # 최종 모델: dev 전체로 학습 (05와 같은 설정)
    model = build_pipeline(m, params, y, cfg.seed, ind, th.get("calibration"))
    model.fit(dev[feats], y)
    hold_metrics = rank_metrics(yh, scores_of(model, hold[feats]))

    # 1) Permutation importance (holdout)
    pi = permutation_importance(model, hold[feats], yh, scoring="average_precision",
                                n_repeats=int(ec.get("permutation_repeats", 30)),
                                random_state=cfg.seed)
    imp = pd.DataFrame({"feature": feats, "importance_mean": pi.importances_mean,
                        "importance_std": pi.importances_std})

    # 2) 영향 방향 (dev OOF 위험도와의 순위 상관)
    oof = ctx.read_csv("05_model", "oof_predictions").set_index("row_index").loc[dev.index, m]
    dirs = []
    for f in feats:
        ok = dev[f].notna()
        rho = spearmanr(dev.loc[ok, f], oof[ok]).statistic if ok.sum() > 2 and dev.loc[ok, f].nunique() > 1 else np.nan
        dirs.append(rho)
    imp["spearman_with_risk"] = dirs
    imp["direction"] = np.where(np.isnan(dirs), "-", np.where(np.array(dirs) > 0, "높을수록 위험↑", "낮을수록 위험↑"))
    imp = imp.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    imp.insert(0, "rank", np.arange(1, len(imp) + 1))
    ctx.write_csv(STAGE, "feature_importance", imp)

    # 3) 정상범위 (dev 정상 샘플)
    ok_rows, nok_rows = dev[y == 0], dev[y == 1]
    rows = []
    for f in imp["feature"]:
        a, b = np.nanpercentile(ok_rows[f], [lo, hi]) if ok_rows[f].notna().any() else (np.nan, np.nan)
        nv = nok_rows[f].dropna()
        rows.append({"feature": f, f"ok_p{lo}": a, "ok_median": ok_rows[f].median(), f"ok_p{hi}": b,
                     "nok_median": nv.median() if len(nv) else np.nan,
                     "nok_outside_rate": float(((nv < a) | (nv > b)).mean()) if len(nv) else np.nan,
                     "missing_rate": float(dev[f].isna().mean())})
    ranges = pd.DataFrame(rows)
    ctx.write_csv(STAGE, "normal_ranges", ranges)

    lodo_by = ex["model"].get("leave_one_out_by")
    if lodo_by:
        part = group_parts(dev, cfg)[lodo_by]
        brow = []
        for v in sorted(part.unique()):
            sub = dev[(part == v) & (y == 0)]
            r = {lodo_by: v, "n_ok": len(sub)}
            for f in imp["feature"]:
                vals = sub[f].dropna()
                r[f] = (f"{np.percentile(vals, lo):.4g} ~ {np.percentile(vals, hi):.4g}" if len(vals) else "결측")
            brow.append(r)
        ctx.write_csv(STAGE, f"normal_ranges_by_{lodo_by}", pd.DataFrame(brow))

    # 4) 모델 패키지
    import joblib
    import sklearn
    joblib.dump(model, ctx.stage_dir(STAGE) / "model.joblib")
    card = {
        "created": datetime.now().isoformat(timespec="seconds"), "run": ctx.run_dir.name,
        "model": m, "model_name": th["model_name"], "features": feats, "missing_indicators": ind,
        "target": cfg.target, "risk_score": "보정된 불량 확률 (0~1)" if th.get("calibration") else "모델 점수",
        "decision_threshold": th["decision_threshold"], "target_recall": th["target_recall"],
        "bands": th["bands"], "trained_on": {"rows": len(dev), "split": "dev"},
        "holdout_check": hold_metrics,
        "normal_ranges": {r["feature"]: [r[f"ok_p{lo}"], r[f"ok_p{hi}"]] for r in rows},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }
    (ctx.stage_dir(STAGE) / "model_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    fig = _plot(ctx, imp)
    ctx.write_json(STAGE, "explain_summary", {"model": m, "model_name": th["model_name"],
                                              "n_features": len(feats), "percentiles": [lo, hi],
                                              "holdout_check": hold_metrics, "figure": fig,
                                              "group_part": lodo_by})

    print(f"  모델: {th['model_name']} (특징 {len(feats)}개), holdout PR-AUC {hold_metrics['PR_AUC']:.3f}")
    for r in imp.merge(ranges, on="feature").itertuples():
        print(f"   {r.rank}. {r.feature:28s} 중요도 {r.importance_mean:+.3f}  {r.direction:10s} "
              f"정상범위 {getattr(r, f'ok_p{lo}'):.4g} ~ {getattr(r, f'ok_p{hi}'):.4g} "
              f"(불량 중 범위 밖 {r.nok_outside_rate:.0%})")
    print("  패키지: 06_explain/model.joblib, model_card.json")
