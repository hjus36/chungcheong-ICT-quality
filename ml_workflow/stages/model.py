"""05 model — 모델별 자기 특징(04_wrapper)으로 비교 → 선택 규칙 → 위험 구간 → holdout.

1) 그룹 CV: 04와 같은 fold. 매 fold 학습 부분으로 순위를 다시 계산해 모델별 k개 사용.
   검증 fold 예측(OOF)을 모아 dev 전체의 '학습에 쓰지 않은 예측'을 만든다.
2) OOF로 모델별 판정 기준(Recall ≥ 목표)·그때 Precision·Brier 계산.
3) 실험일 제외 진단을 모든 모델에 적용 (선택 규칙의 동점 처리 기준).
4) 선택 규칙: gate → 주 지표(PR-AUC) → 1-SE 동점 집합 → 실험일 제외 → Brier → 해석 가능성.
5) 선택 모델 OOF로 위험 구간 기준(정상/품질주의/불량 위험)을 정하고 holdout에 고정 적용.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.design import rank_fdr_candidates
from src.evaluate import fit_predict, rank_metrics, threshold_at_recall, threshold_metrics
from src.groups import group_parts
from src.models import DISPLAY, is_available
from src.pipeline import load_split, used_features
from src.runio import StageError
from src.selection import select_model

STAGE = "05_model"
BANDS = ["정상", "품질주의", "불량 위험"]


def band_of(scores, thr_caution, thr_defect):
    s = np.asarray(scores)
    return np.where(s >= thr_defect, BANDS[2], np.where(s >= thr_caution, BANDS[1], BANDS[0]))


def _plot(ctx, trace):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 5))
    names = [f"{DISPLAY[m]}\n(k={k})" for m, k in zip(trace["model"], trace["k"])]
    x = np.arange(len(names))
    ax.bar(x - 0.27, trace["PR_AUC_mean"], 0.27, yerr=trace["PR_AUC_std"], capsize=3, label="CV PR-AUC")
    ax.bar(x, trace["Precision_at_target"], 0.27, label="Precision @ target Recall (OOF)")
    ax.bar(x + 0.27, trace["lodo_PR_AUC"], 0.27, label="Leave-one-day-out PR-AUC")
    ax.axhline(trace["tie_floor"].iloc[0], ls="--", lw=1, color="gray", label="1-SE tie floor")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.08)
    ax.set_title("Model selection criteria (dev, group CV)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    path = ctx.stage_dir(STAGE) / "model_selection.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def run(ctx):
    cfg = ctx.cfg
    ex = cfg.experiment
    mc = ex["model"]
    params = mc.get("params") or {}
    calib = (mc.get("calibration") or {"method": "sigmoid", "cv": 3}) if mc.get("calibrate") else None
    alpha = float(ex["filter"].get("fdr_alpha", 0.05))
    sel_cfg = mc.get("selection") or {}
    target = float((sel_cfg.get("gate") or {}).get("target_recall", 0.9))
    bands_cfg = mc.get("risk_bands") or {}

    sets = ctx.read_json("04_wrapper", "model_feature_sets")
    w = ctx.read_json("04_wrapper", "wrapper_summary")
    ind = bool(w["primary_missing_indicators"])
    candidates = mc.get("candidates") or list(sets)
    models = [m for m in candidates if is_available(m) and m in sets]
    skipped = [m for m in candidates if m not in models]
    for m in skipped:
        print(f"  [경고] {m}: 패키지가 없거나 04_wrapper 결과가 없어 제외합니다.")
    if not models:
        raise StageError("비교할 모델이 없습니다.")

    dev, hold = load_split(ctx)
    y, yh = dev[cfg.target].astype(int), hold[cfg.target].astype(int)
    feats = used_features(ctx)
    X, Xh = dev[feats], hold[feats]
    fold_ids = ctx.read_csv("04_wrapper", "cv_folds").set_index("row_index").loc[dev.index, "fold"].to_numpy()
    n_folds = len(np.unique(fold_ids))

    # 1) 그룹 CV + OOF
    oof = {m: np.full(len(dev), np.nan) for m in models}
    rows = []
    for f in sorted(np.unique(fold_ids)):
        tr, va = np.flatnonzero(fold_ids != f), np.flatnonzero(fold_ids == f)
        ranked = rank_fdr_candidates(X.iloc[tr], y.iloc[tr], alpha, ind)[0]["feature"].tolist()
        for m in models:
            k = sets[m]["k"]
            if len(ranked) < k:
                raise StageError(f"fold {f}의 FDR 후보({len(ranked)})가 {m}의 k={k}보다 적습니다.")
            s, _, _ = fit_predict(m, params.get(m), X.iloc[tr], y.iloc[tr], X.iloc[va], ranked[:k], ind,
                                  cfg.seed, calib)
            oof[m][va] = s
            rows.append({"model": m, "fold": int(f), **rank_metrics(y.iloc[va], s)})
    folds = pd.DataFrame(rows)
    ctx.write_csv(STAGE, "cv_fold_metrics", folds)
    oof_tab = pd.DataFrame({"row_index": dev.index, "fold": fold_ids, cfg.target: y.values,
                            **{m: oof[m] for m in models}})
    ctx.write_csv(STAGE, "oof_predictions", oof_tab)

    table = []
    for m in models:
        f = folds[folds["model"] == m]
        thr, p_at, r_at = threshold_at_recall(y, oof[m], target)
        tm = threshold_metrics(y, oof[m], thr)
        om = rank_metrics(y, oof[m])
        table.append({"model": m, "model_name": DISPLAY[m], "k": sets[m]["k"],
                      "PR_AUC_mean": f["PR_AUC"].mean(), "PR_AUC_std": f["PR_AUC"].std(ddof=1),
                      "ROC_AUC_mean": f["ROC_AUC"].mean(), "Brier": om["Brier"], "PR_AUC_oof": om["PR_AUC"],
                      "threshold_at_target": thr, "Recall_at_target": tm["Recall"],
                      "Precision_at_target": tm["Precision"], "F1_at_target": tm["F1"],
                      "FP_at_target": tm["FP"], "FN_at_target": tm["FN"]})
    table = pd.DataFrame(table)

    # 2) 실험일(그룹 키 일부) 제외 진단 — 모든 모델
    lodo_by = mc.get("leave_one_out_by")
    lodo = pd.DataFrame()
    if lodo_by:
        part = group_parts(dev, cfg)[lodo_by].to_numpy()
        lrows = []
        for v in sorted(np.unique(part)):
            te = part == v
            if y[~te].nunique() < 2:
                continue
            ranked = rank_fdr_candidates(X[~te], y[~te], alpha, ind)[0]["feature"].tolist()
            for m in models:
                sel = ranked[:sets[m]["k"]]
                s, _, _ = fit_predict(m, params.get(m), X[~te], y[~te], X[te], sel, ind, cfg.seed, calib)
                thr = float(table.loc[table["model"] == m, "threshold_at_target"].iloc[0])
                lrows.append({"model": m, "model_name": DISPLAY[m], "held_out": v, "n_test": int(te.sum()),
                              "nok_rate_test": float(y[te].mean()), **rank_metrics(y[te], s),
                              **{f"{k}_at_cv_threshold": val for k, val in
                                 threshold_metrics(y[te], s, thr).items() if k in ("Recall", "Precision")}})
        lodo = pd.DataFrame(lrows)
        ctx.write_csv(STAGE, "leave_one_out", lodo)
        table = table.merge(lodo.groupby("model")["PR_AUC"].mean().rename("lodo_PR_AUC"),
                            left_on="model", right_index=True, how="left")
    else:
        table["lodo_PR_AUC"] = np.nan

    # 3) 선택 규칙
    trace, chosen, reason = select_model(table, sel_cfg, n_folds)
    ctx.write_csv(STAGE, "selection_trace", trace)

    # 4) 위험 구간 기준 (선택 모델 OOF)
    thr_def = threshold_at_recall(y, oof[chosen], float(bands_cfg.get("defect_recall", target)))[0]
    thr_cau = min(threshold_at_recall(y, oof[chosen], float(bands_cfg.get("caution_recall", 0.98)))[0], thr_def)
    oof_band = pd.DataFrame({"band": band_of(oof[chosen], thr_cau, thr_def), "y": y.values}) \
        .groupby("band")["y"].agg(n="size", n_nok="sum", nok_rate="mean").reindex(BANDS).reset_index()
    ctx.write_csv(STAGE, "oof_bands", oof_band)

    # 5) Holdout — dev 전체로 학습, 모델별 자기 특징, CV에서 정한 기준 고정 적용
    hrows = []
    for m in models:
        s, _, _ = fit_predict(m, params.get(m), X, y, Xh, sets[m]["features"], ind, cfg.seed, calib)
        thr = float(table.loc[table["model"] == m, "threshold_at_target"].iloc[0])
        hrows.append({"model": m, "model_name": DISPLAY[m], "selected": m == chosen, "k": sets[m]["k"],
                      "threshold": thr, **rank_metrics(yh, s), **threshold_metrics(yh, s, thr)})
        if m == chosen:
            hold_scores = s
    holdout = pd.DataFrame(hrows)
    ctx.write_csv(STAGE, "holdout_metrics", holdout)
    hb = pd.DataFrame({"row_index": hold.index, cfg.target: yh.values, "risk_score": hold_scores,
                       "band": band_of(hold_scores, thr_cau, thr_def)})
    ctx.write_csv(STAGE, "holdout_predictions", hb)
    hold_band = hb.groupby("band")[cfg.target].agg(n="size", n_nok="sum", nok_rate="mean") \
        .reindex(BANDS).reset_index()
    ctx.write_csv(STAGE, "holdout_bands", hold_band)

    thresholds = {
        "model": chosen, "model_name": DISPLAY[chosen], "features": sets[chosen]["features"],
        "missing_indicators": ind, "calibration": calib,
        "target_recall": target, "decision_threshold": float(table.loc[table.model == chosen, "threshold_at_target"].iloc[0]),
        "bands": {"caution": thr_cau, "defect": thr_def,
                  "caution_recall": float(bands_cfg.get("caution_recall", 0.98)),
                  "defect_recall": float(bands_cfg.get("defect_recall", target)),
                  "labels": BANDS},
    }
    ctx.write_json(STAGE, "thresholds", thresholds)
    fig = _plot(ctx, trace)
    hs = holdout[holdout["selected"]].iloc[0]
    ctx.write_json(STAGE, "model_summary", {
        "models": models, "skipped": skipped, "k": {m: sets[m]["k"] for m in models},
        "missing_indicators": ind, "calibration": calib, "n_folds": n_folds,
        "selected_model": chosen, "selected_model_name": DISPLAY[chosen], "selection_reason": reason,
        "selection_config": sel_cfg, "risk_bands": thresholds["bands"],
        "holdout_rows": len(hold), "holdout_selected": {k: hs[k] for k in
            ["PR_AUC", "ROC_AUC", "Brier", "Recall", "Precision", "F1", "TP", "FN", "FP", "TN", "threshold"]},
        "leave_one_out_by": lodo_by, "figure": fig,
    })

    print(f"  indicator={'on' if ind else 'off'}, 보정={'on' if calib else 'off'}, 목표 Recall {target}")
    for r in trace.itertuples():
        print(f"   {r.final_rank}. {r.model_name:20s} k={r.k:2d} PR-AUC {r.PR_AUC_mean:.3f}±{r.PR_AUC_std:.3f} "
              f"P@R{target:.2f} {r.Precision_at_target:.3f} 실험일제외 {r.lodo_PR_AUC:.3f} "
              f"Brier {r.Brier:.3f} → {r.result}")
    print(f"  선택: {DISPLAY[chosen]} — {reason}")
    print(f"  위험 구간 기준: 품질주의 ≥ {thr_cau:.3f}, 불량 위험 ≥ {thr_def:.3f}")
    print(f"  Holdout({len(hold)}행, 기준 {hs['threshold']:.3f} 고정): PR-AUC {hs['PR_AUC']:.3f}, "
          f"Recall {hs['Recall']:.3f}, Precision {hs['Precision']:.3f} (TP {hs['TP']} FN {hs['FN']} FP {hs['FP']})")
    for r in hold_band.itertuples():
        print(f"   {r.band:6s}: {int(r.n) if pd.notna(r.n) else 0:3d}행, 실제 불량률 "
              f"{(r.nok_rate if pd.notna(r.nok_rate) else 0):.1%}")
