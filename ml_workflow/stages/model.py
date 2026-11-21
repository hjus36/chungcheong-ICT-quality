"""05 model — 모델 비교 → 선택 규칙 → 위험 구간 → 기준선·holdout 평가.

1) 반복 그룹 CV: 04와 같은 fold(반복 포함). 매 fold 학습 부분으로 순위를 다시 계산해 모델별 k개 사용.
   검증 fold 예측(OOF)을 반복마다 모은다.
2) OOF로 모델별 판정 기준(Recall ≥ 목표)·그때 Precision·Brier 계산.
3) 실험일 제외 진단(모든 모델): 실험일 하나를 통째로 빼고 학습 → CV 판정 기준 그대로 적용.
4) 선택 규칙: Gate 1(불량 탐지) + Gate 2(실험일 일반화) → PR-AUC → 1-SE 동점 → tie 순서.
5) 선택 모델 OOF로 위험 구간 기준을 정하고 holdout에 고정 적용. 기준선과 함께 보고.
6) holdout 평가 성격(첫 평가 / 개발 참고 평가)을 기록한다.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data import file_sha256
from src.design import rank_fdr_candidates
from src.evaluate import fit_predict, rank_metrics, threshold_at_recall, threshold_metrics
from src.groups import group_parts
from src.models import DISPLAY, is_available
from src.pipeline import folds_from_table, load_split, used_features
from src.runio import StageError
from src.selection import select_model

STAGE = "05_model"
BANDS = ["정상", "품질주의", "불량 위험"]
DEV_REF = "previously_seen_holdout_development_reference"
FIRST = "first_holdout_evaluation"


def band_of(scores, thr_caution, thr_defect):
    s = np.asarray(scores)
    return np.where(s >= thr_defect, BANDS[2], np.where(s >= thr_caution, BANDS[1], BANDS[0]))


def _pooled(y, oof_mat):
    """반복 OOF (n × R) → 전부 이어 붙인 (y, score)."""
    R = oof_mat.shape[1]
    return np.tile(np.asarray(y), R), oof_mat.T.reshape(-1)


def _holdout_status(ctx, hold_index, ev):
    data_sha = file_sha256(ctx.data_file)
    sig = hashlib.sha256((data_sha + "," + ",".join(map(str, sorted(hold_index)))).encode()).hexdigest()
    ledger_path = ctx.run_dir.parent / "holdout_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    previous = [r for r in ledger.get(sig, []) if r != ctx.run_dir.name]
    mode = (ev or {}).get("holdout_status", "auto")
    status = DEV_REF if (mode == "development_reference" or previous) else FIRST
    ledger.setdefault(sig, [])
    if ctx.run_dir.name not in ledger[sig]:
        ledger[sig].append(ctx.run_dir.name)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    note = (ev or {}).get("holdout_status_note") if mode == "development_reference" else \
        (f"같은 holdout을 이전 실행이 이미 평가했다: {previous}" if previous else "이 분할의 첫 holdout 평가")
    return {"status": status, "split_signature": sig, "previous_runs": previous, "mode": mode, "note": note}


def _plot(ctx, trace, target):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [f"{DISPLAY[m]}\n(k={k})" for m, k in zip(trace["model"], trace["k"])]
    x = np.arange(len(names))
    w = 0.2
    ax.bar(x - 1.5 * w, trace["PR_AUC_mean"], w, yerr=trace["PR_AUC_std"], capsize=3, label="CV PR-AUC")
    ax.bar(x - 0.5 * w, trace["Precision_at_target"], w, label=f"Precision @ Recall {target} (OOF) — Gate 1")
    ax.bar(x + 0.5 * w, trace["lodo_recall_min"], w, label="Min recall, leave-one-day-out — Gate 2")
    ax.bar(x + 1.5 * w, trace["lodo_PR_AUC"], w, label="Leave-one-day-out PR-AUC")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model selection criteria (dev, repeated group CV)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left")
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
    n_splits = int(ex["cv"].get("n_splits", 5))

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
    fold_tab = ctx.read_csv("04_wrapper", "cv_folds").set_index("row_index").loc[dev.index].reset_index()
    folds = folds_from_table(fold_tab)
    n_repeats = len({r for r, *_ in folds})

    # 1) 반복 그룹 CV + OOF
    rankings = [rank_fdr_candidates(X.iloc[tr], y.iloc[tr], alpha, ind)[0]["feature"].tolist()
                for _, _, tr, _ in folds]
    oof, rows = {}, []
    for m in models:
        ckpt = ctx.stage_dir(STAGE) / f"_ckpt_cv_{m}.npz"   # 중단 후 --resume 시 재사용
        if ckpt.exists():
            z = np.load(ckpt, allow_pickle=False)
            oof[m] = z["oof"]
            rows += pd.read_json(ckpt.with_suffix(".json")).to_dict("records")
            print(f"   {DISPLAY[m]}: 저장된 CV 결과 사용")
            continue
        oof[m] = np.full((len(dev), n_repeats), np.nan)
        mrows = []
        for (r, f, tr, va), ranked in zip(folds, rankings):
            k = sets[m]["k"]
            if len(ranked) < k:
                raise StageError(f"반복 {r} fold {f}의 FDR 후보({len(ranked)})가 {m}의 k={k}보다 적습니다.")
            s, _, _ = fit_predict(m, params.get(m), X.iloc[tr], y.iloc[tr], X.iloc[va], ranked[:k], ind,
                                  cfg.seed, calib)
            oof[m][va, r - 1] = s
            mrows.append({"model": m, "repeat": r, "fold": f, **rank_metrics(y.iloc[va], s)})
        np.savez(ckpt, oof=oof[m])
        pd.DataFrame(mrows).to_json(ckpt.with_suffix(".json"), orient="records")
        rows += mrows
        print(f"   {DISPLAY[m]}: CV 완료", flush=True)
    fold_metrics = pd.DataFrame(rows)
    ctx.write_csv(STAGE, "cv_fold_metrics", fold_metrics)
    oof_tab = pd.DataFrame({"row_index": dev.index, cfg.target: y.values})
    for m in models:
        for r in range(n_repeats):
            oof_tab[f"{m}_r{r + 1}"] = oof[m][:, r]
        oof_tab[m] = oof[m].mean(axis=1)
    ctx.write_csv(STAGE, "oof_predictions", oof_tab)

    table = []
    for m in models:
        f = fold_metrics[fold_metrics["model"] == m]
        yp, sp = _pooled(y, oof[m])
        thr, _, _ = threshold_at_recall(yp, sp, target)
        tm = threshold_metrics(yp, sp, thr)
        om = rank_metrics(yp, sp)
        table.append({"model": m, "model_name": DISPLAY[m], "k": sets[m]["k"],
                      "PR_AUC_mean": f["PR_AUC"].mean(), "PR_AUC_std": f["PR_AUC"].std(ddof=1),
                      "ROC_AUC_mean": f["ROC_AUC"].mean(), "Brier": om["Brier"], "PR_AUC_oof": om["PR_AUC"],
                      "threshold_at_target": thr, "Recall_at_target": tm["Recall"],
                      "Precision_at_target": tm["Precision"], "F2_at_target": tm["F2"],
                      "Alert_rate_at_target": tm["Alert_rate"]})
    table = pd.DataFrame(table)
    thr_of = dict(zip(table["model"], table["threshold_at_target"]))

    # 2) 실험일 제외 진단 — 모든 모델
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
                s, _, _ = fit_predict(m, params.get(m), X[~te], y[~te], X[te], ranked[:sets[m]["k"]], ind,
                                      cfg.seed, calib)
                tm = threshold_metrics(y[te], s, thr_of[m])
                lrows.append({"model": m, "model_name": DISPLAY[m], "held_out": v, "n_test": int(te.sum()),
                              "n_nok_test": int(y[te].sum()), "nok_rate_test": float(y[te].mean()),
                              **rank_metrics(y[te], s), "Recall_at_cv_threshold": tm["Recall"],
                              "Precision_at_cv_threshold": tm["Precision"], "TP": tm["TP"], "FN": tm["FN"],
                              "FP": tm["FP"]})
        lodo = pd.DataFrame(lrows)
        ctx.write_csv(STAGE, "leave_one_out", lodo)
        agg = lodo.groupby("model").agg(lodo_PR_AUC=("PR_AUC", "mean"),
                                        lodo_recall_min=("Recall_at_cv_threshold", "min"),
                                        lodo_recall_mean=("Recall_at_cv_threshold", "mean"))
        table = table.merge(agg, left_on="model", right_index=True, how="left")
    else:
        table["lodo_PR_AUC"] = table["lodo_recall_min"] = table["lodo_recall_mean"] = np.nan

    # 3) 선택 규칙
    trace, chosen, reason = select_model(table, sel_cfg, n_splits)
    ctx.write_csv(STAGE, "selection_trace", trace)
    chosen_row = trace[trace["model"] == chosen].iloc[0]

    # 4) 위험 구간 기준 (선택 모델 OOF, 반복 전체)
    yp, sp = _pooled(y, oof[chosen])
    thr_def = threshold_at_recall(yp, sp, float(bands_cfg.get("defect_recall", target)))[0]
    thr_cau = min(threshold_at_recall(yp, sp, float(bands_cfg.get("caution_recall", 0.98)))[0], thr_def)
    oof_band = pd.DataFrame({"band": band_of(oof[chosen].mean(axis=1), thr_cau, thr_def), "y": y.values}) \
        .groupby("band")["y"].agg(n="size", n_nok="sum", nok_rate="mean").reindex(BANDS).reset_index()
    ctx.write_csv(STAGE, "oof_bands", oof_band)

    # 5) Holdout — 상태 기록 후 1회 평가
    hstat = _holdout_status(ctx, hold.index, ex.get("evaluation"))
    ctx.write_json(STAGE, "holdout_lock", hstat)
    hrows = []
    for m in models:
        s, _, _ = fit_predict(m, params.get(m), X, y, Xh, sets[m]["features"], ind, cfg.seed, calib)
        hrows.append({"model": m, "model_name": DISPLAY[m], "selected": m == chosen, "k": sets[m]["k"],
                      "threshold": thr_of[m], **rank_metrics(yh, s), **threshold_metrics(yh, s, thr_of[m])})
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

    # 6) 기준선
    bc = ex.get("baselines") or {}
    brows = []
    prev = float(yh.mean())
    if bc.get("all_risk", True):
        brows.append({"baseline": "all_risk", "name": "전부 우선 검토", "threshold": 0.0,
                      **rank_metrics(yh, np.ones(len(yh))), **threshold_metrics(yh, np.ones(len(yh)), 0.5)})
    if bc.get("all_normal", True):
        brows.append({"baseline": "all_normal", "name": "전부 통상 검토", "threshold": 1.0,
                      **rank_metrics(yh, np.zeros(len(yh))), **threshold_metrics(yh, np.zeros(len(yh)), 0.5)})
    if bc.get("single_feature", True):
        rank = ctx.read_csv("03_filter", "filter_ranking")
        top = rank.loc[~rank["best_component"].str.endswith("__missing"), "feature"].iloc[0]
        med = X[top].median()
        xs = X[top].fillna(med)
        sign = 1.0 if spearmanr(xs, y).statistic >= 0 else -1.0
        thr, _, _ = threshold_at_recall(y, sign * xs, target)
        hs = sign * Xh[top].fillna(med)
        brows.append({"baseline": "single_feature", "threshold": thr,
                      "name": f"단일 특징 규칙: {top} {'≥' if sign > 0 else '≤'} {sign * thr:.4g}",
                      **rank_metrics(yh, hs), **threshold_metrics(yh, hs, thr)})
    pre = bc.get("setpoint_only_prefix")
    if pre:
        sp_feats = [f for f in feats if f.startswith(f"{pre}_")]
        if sp_feats:
            r1 = [(tr, va) for r, f, tr, va in folds if r == 1]
            so = np.full(len(dev), np.nan)
            for tr, va in r1:
                so[va] = fit_predict("logistic_regression", params.get("logistic_regression"), X.iloc[tr],
                                     y.iloc[tr], X.iloc[va], sp_feats, ind, cfg.seed, calib)[0]
            thr, _, _ = threshold_at_recall(y, so, target)
            s, _, _ = fit_predict("logistic_regression", params.get("logistic_regression"), X, y, Xh, sp_feats,
                                  ind, cfg.seed, calib)
            brows.append({"baseline": "setpoint_only", "threshold": thr,
                          "name": f"설정값만 사용한 LR ({pre}_* {len(sp_feats)}개)",
                          **rank_metrics(yh, s), **threshold_metrics(yh, s, thr)})
    baselines = pd.DataFrame(brows)
    ctx.write_csv(STAGE, "baselines_holdout", baselines)

    thresholds = {
        "model": chosen, "model_name": DISPLAY[chosen], "features": sets[chosen]["features"],
        "missing_indicators": ind, "calibration": calib, "target_recall": target,
        "decision_threshold": float(thr_of[chosen]),
        "bands": {"caution": thr_cau, "defect": thr_def,
                  "caution_recall": float(bands_cfg.get("caution_recall", 0.98)),
                  "defect_recall": float(bands_cfg.get("defect_recall", target)), "labels": BANDS},
        "gates": {"gate1_pass": bool(chosen_row["gate1_pass"]), "gate2_pass": bool(chosen_row["gate2_pass"])},
    }
    ctx.write_json(STAGE, "thresholds", thresholds)
    fig = _plot(ctx, trace, target)
    hs_ = holdout[holdout["selected"]].iloc[0]
    ctx.write_json(STAGE, "model_summary", {
        "models": models, "skipped": skipped, "k": {m: sets[m]["k"] for m in models},
        "missing_indicators": ind, "calibration": calib, "n_folds": n_splits, "n_repeats": n_repeats,
        "selected_model": chosen, "selected_model_name": DISPLAY[chosen], "selection_reason": reason,
        "selection_config": sel_cfg, "selected_gates": thresholds["gates"],
        "any_model_passed_gates": bool(trace["gate_pass"].any()), "risk_bands": thresholds["bands"],
        "holdout_rows": len(hold), "holdout_prevalence": prev, "holdout_status": hstat,
        "holdout_selected": {k: hs_[k] for k in ["PR_AUC", "ROC_AUC", "Brier", "Recall", "Precision", "F1", "F2",
                                                 "Alert_rate", "TP", "FN", "FP", "TN", "threshold"]},
        "leave_one_out_by": lodo_by, "figure": fig,
    })

    print(f"  반복 그룹 CV {n_splits}-fold × {n_repeats}, indicator={'on' if ind else 'off'}, 목표 Recall {target}")
    for r in trace.itertuples():
        print(f"   {r.final_rank}. {r.model_name:20s} k={r.k:2d} PR-AUC {r.PR_AUC_mean:.3f}±{r.PR_AUC_std:.3f} "
              f"P@R {r.Precision_at_target:.3f} 실험일 최소Recall {r.lodo_recall_min:.2f} → {r.result}")
    print(f"  선택: {DISPLAY[chosen]} — {reason}")
    print(f"  위험 구간 기준: 품질주의 ≥ {thr_cau:.3f}, 불량 위험(우선 검토) ≥ {thr_def:.3f}")
    print(f"  Holdout [{'개발 참고 평가' if hstat['status'] == DEV_REF else '첫 평가'}] ({len(hold)}행, 기준 고정): "
          f"PR-AUC {hs_['PR_AUC']:.3f}, Recall {hs_['Recall']:.3f}, Precision {hs_['Precision']:.3f}, "
          f"F2 {hs_['F2']:.3f}, 우선 검토 비율 {hs_['Alert_rate']:.0%}")
    for b in baselines.itertuples():
        print(f"   기준선 {b.name}: PR-AUC {b.PR_AUC:.3f}, Recall {b.Recall:.3f}, Precision {b.Precision:.3f}, "
              f"F2 {b.F2:.3f}")
