"""07 release — 서비스 계약 파일을 만들고 승격 조건을 판정한다.

만드는 파일 (runs/<이름>/07_release/)
- model.joblib           서비스 번들 (파이프라인 + 특징 + 판정 기준 + 참고 구간 + 결측 정책)
- benchmark_public.json  화면·API용 집계 지표 (행 단위 정보 없음)
- data_dictionary.json   업로드 입력 계약 (메타·특징·단위·범위·필수 여부)
- release_check.json     승격 조건 판정 결과 (promote.py가 확인)
"""
from __future__ import annotations

import hashlib
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr

from src.contract import feature_info, lower_metrics
from src.models import DISPLAY
from src.pipeline import load_split
from src.runio import StageError

STAGE = "07_release"


def _histogram(v, bins):
    v = v.dropna().to_numpy(dtype=float)
    if len(v) == 0:
        return []
    counts, edges = np.histogram(v, bins=bins)
    return [{"low": float(a), "high": float(b), "count": int(c)} for a, b, c in zip(edges[:-1], edges[1:], counts)]


def _relation(v, y, min_group):
    """값 순서로 정렬해 min_group개 이상씩 묶은 평균 (개별 부품 값은 공개하지 않음)."""
    d = pd.DataFrame({"v": v, "y": y}).dropna().sort_values("v")
    n_bins = min(10, len(d) // min_group)
    if n_bins < 2:
        return []
    out = []
    for idx in np.array_split(np.arange(len(d)), n_bins):
        chunk = d.iloc[idx]
        out.append({"value": float(chunk["v"].mean()), "defect_rate_percent": float(chunk["y"].mean() * 100),
                    "count": int(len(chunk))})
    return out


def run(ctx):
    cfg = ctx.cfg
    rel, svc = cfg.release, cfg.columns.get("service") or {}
    ms = ctx.read_json("05_model", "model_summary")
    th = ctx.read_json("05_model", "thresholds")
    trace = ctx.read_csv("05_model", "selection_trace")
    hold = ctx.read_csv("05_model", "holdout_metrics")
    base = ctx.read_csv("05_model", "baselines_holdout")
    imp = ctx.read_csv("06_explain", "feature_importance").set_index("feature")
    ranking = ctx.read_csv("03_filter", "filter_ranking")
    lodo = ctx.read_csv("05_model", "leave_one_out") if ms.get("leave_one_out_by") else pd.DataFrame()
    model_path = ctx.run_dir / "06_explain" / "model.joblib"
    if not model_path.exists():
        raise StageError("06_explain/model.joblib이 없습니다.")
    pipeline = joblib.load(model_path)
    feats = th["features"]
    version = f"{rel.get('version_prefix', 'model')}-{ctx.run_dir.name}"

    dev, _ = load_split(ctx)
    y = dev[cfg.target].astype(int)
    ok = dev[y == 0]
    lo, hi = (cfg.experiment.get("explain") or {}).get("normal_range_percentiles", [5, 95])

    # 결측 정책: dev에서 한 번도 비지 않은 특징 = 필수, 결측이 있던 특징 = 선택(파이프라인이 대체)
    required = [f for f in feats if dev[f].notna().all()]
    optional = [f for f in feats if f not in required]
    info = {f: feature_info(cfg, f) for f in feats}
    reference = {"q05": {}, "median": {}, "q95": {}, "mean": {}, "std": {}}
    for f in feats:
        reference["q05"][f] = float(np.nanpercentile(ok[f], lo))
        reference["median"][f] = float(ok[f].median())
        reference["q95"][f] = float(np.nanpercentile(ok[f], hi))
        reference["mean"][f] = float(ok[f].mean())
        reference["std"][f] = float(ok[f].std())
    reference["basis"] = f"dev 정상 부품 {lo}~{hi} 백분위"

    bundle = {
        "schema_version": 2, "version": version, "data_source": svc.get("data_source"),
        "selected_model": DISPLAY[th["model"]], "model_key": th["model"], "pipeline": pipeline,
        "features": feats, "alert_threshold": float(th["bands"]["defect"]),
        "caution_threshold": float(th["bands"]["caution"]), "risk_score_kind": "calibrated_probability",
        "target_recall": th["target_recall"], "prediction_time": cfg.columns.get("prediction_time"),
        "label": {"column": cfg.target, "definition": "LBL_SinkMarks 또는 LBL_Underfilled가 1이면 불량(1)"},
        "reference": reference,
        "missing_policy": {"required": required, "optional": optional,
                           "imputation": "학습(dev) 중앙값으로 대체하고 결측 여부를 indicator로 함께 입력"},
        "units": {f: info[f]["unit"] for f in feats},
        "ranges": {f: {"min": info[f]["min"], "max": info[f]["max"]} for f in feats},
    }
    out_dir = ctx.stage_dir(STAGE)
    joblib.dump(bundle, out_dir / "model.joblib")
    model_sha = hashlib.sha256((out_dir / "model.joblib").read_bytes()).hexdigest()

    # 데이터 사전
    meta = svc.get("meta") or {}
    dictionary = {
        "version": version, "data_source": svc.get("data_source"), "synthetic_source": svc.get("synthetic_source"),
        "meta": [
            {"name": "part_id", "required": True, "source_column": meta.get("part_id"),
             "description": "부품(사출 사이클) 식별자. 파일 안에서 중복될 수 없다."},
            {"name": "cycle_start_at", "required": True, "source_column": meta.get("cycle_start_at"),
             "description": "사이클 시작 시각 (YYYY-MM-DD HH:MM:SS)"},
            {"name": "cycle_end_at", "required": True, "source_column": meta.get("cycle_end_at"),
             "description": "사이클 종료 시각 (YYYY-MM-DD HH:MM:SS)"},
            {"name": "data_source", "required": True, "source_column": None,
             "description": f"{svc.get('data_source')}(실측) 또는 {svc.get('synthetic_source')}(합성). 한 파일에 섞을 수 없다."},
        ],
        "optional": [
            {"name": "material", "source_column": meta.get("material"), "description": "재료 (예: PP, ABS)"},
            {"name": svc.get("label_column", "actual_label"), "source_column": cfg.target,
             "description": "실제 판정 0(정상)/1(불량). 예측에 쓰지 않고 사후 비교에만 사용"},
        ],
        "features": [{"name": f, "unit": info[f]["unit"], "description": info[f]["description"],
                      "min": info[f]["min"], "max": info[f]["max"], "required": f in required,
                      "reference_q05": reference["q05"][f], "reference_median": reference["median"][f],
                      "reference_q95": reference["q95"][f]} for f in feats],
        "notes": ["단위 변환은 하지 않는다. 표시된 단위로 입력한다.",
                  "min/max는 물리적으로 불가능한 값을 거르는 범위이며 공정 규격이 아니다.",
                  "required=false인 특징은 비어 있어도 학습 중앙값으로 대체해 예측한다."],
    }
    ctx.write_json(STAGE, "data_dictionary", dictionary)

    # 화면용 공정변수 분석: 모델 특징 + Filter 상위 특징
    an = rel.get("analysis") or {}
    extra = [f for f in ranking.loc[~ranking["best_component"].str.endswith("__missing"), "feature"]
             if f not in feats][: int(an.get("extra_top_features", 10))]
    fa = []
    for f in feats + extra:
        okv = ok[f]
        rho = spearmanr(dev[f], y, nan_policy="omit").statistic if dev[f].nunique() > 1 else np.nan
        fi = feature_info(cfg, f)
        fa.append({"feature": f, "in_model": f in feats, "unit": fi["unit"], "description": fi["description"],
                   "q05": float(np.nanpercentile(okv, lo)), "median": float(okv.median()),
                   "q95": float(np.nanpercentile(okv, hi)),
                   "model_weight": float(imp.loc[f, "importance_mean"]) if f in imp.index else 0.0,
                   "permutation_ap_drop": float(imp.loc[f, "importance_mean"]) if f in imp.index else None,
                   "permutation_ap_std": float(imp.loc[f, "importance_std"]) if f in imp.index else None,
                   "direction": imp.loc[f, "direction"] if f in imp.index else None,
                   "spearman": None if np.isnan(rho) else float(rho), "missing_rate": float(dev[f].isna().mean()),
                   "histogram": _histogram(dev[f], int(an.get("histogram_bins", 12))),
                   "quality_relation": _relation(dev[f], y, int(an.get("relation_min_group", 5)))})

    hsel = hold[hold["selected"].astype(bool)].iloc[0].to_dict()
    blist = base.to_dict("records")
    by_name = {b["baseline"]: b for b in blist}
    cv_rows = []
    for r in trace.to_dict("records"):
        cv_rows.append({"model": r["model_name"], "model_key": r["model"], "selected": bool(r["selected"]),
                        "features": int(r["k"]), "alert_threshold": r["threshold_at_target"],
                        "average_precision": r["PR_AUC_mean"], "average_precision_std": r["PR_AUC_std"],
                        "roc_auc": r["ROC_AUC_mean"], "brier": r["Brier"],
                        "recall": r["Recall_at_target"], "precision": r["Precision_at_target"],
                        "f2": r["F2_at_target"], "alert_rate": r["Alert_rate_at_target"],
                        "gate1_pass": bool(r["gate1_pass"]), "gate2_pass": bool(r["gate2_pass"]),
                        "lodo_recall_min": r.get("lodo_recall_min"), "lodo_average_precision": r.get("lodo_PR_AUC"),
                        "in_tie_set": bool(r["in_tie_set"]), "result": r["result"]})
    hs = ms["holdout_status"]
    man = ctx.run_dir / "manifest.json"
    import json
    manifest = json.loads(man.read_text(encoding="utf-8")) if man.exists() else {}
    gate_cfg = (cfg.experiment["model"].get("selection") or {}).get("gate") or {}
    benchmark = {
        "version": version, "selected_model": DISPLAY[th["model"]], "model_key": th["model"],
        "alert_threshold": float(th["bands"]["defect"]), "caution_threshold": float(th["bands"]["caution"]),
        "risk_score_kind": "calibrated_probability", "model_sha256": model_sha,
        "data_sha256": manifest.get("data_sha256"), "seed": cfg.seed,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__},
        "label": bundle["label"], "prediction_time": bundle["prediction_time"],
        "selection_rule": (f"반복 그룹 CV({ms['n_folds']}-fold × {ms['n_repeats']}) · Gate 1: Recall "
                           f"{gate_cfg.get('target_recall')} 지점 Precision ≥ {gate_cfg.get('min_precision')} · "
                           f"Gate 2: 실험일 제외 Recall ≥ {gate_cfg.get('lodo_min_recall')} · PR-AUC → 1-SE 동점 → "
                           "실험일 제외 PR-AUC → Brier → 해석 가능성"),
        "selection_reason": ms["selection_reason"],
        "validation_comparison": cv_rows,
        "test": lower_metrics(hsel), "test_status": hs["status"], "test_status_note": hs.get("note"),
        "test_all_risk": lower_metrics(by_name.get("all_risk", {})),
        "test_all_normal": lower_metrics(by_name.get("all_normal", {})),
        "baselines": [{"baseline": b["baseline"], "name": b["name"], **lower_metrics(b)} for b in blist],
        "holdout_models": [{"model": r["model_name"], "selected": bool(r["selected"]), **lower_metrics(r)}
                           for r in hold.to_dict("records")],
        "leave_one_out": [lower_metrics(r) for r in lodo.to_dict("records")],
        "leave_one_out_by": ms.get("leave_one_out_by"),
        "risk_bands": th["bands"], "split": {"dev_rows": len(dev), "holdout_rows": ms["holdout_rows"],
                                             "holdout_prevalence": ms["holdout_prevalence"]},
        "feature_analysis": fa, "feature_count": len(feats), "reference_basis": reference["basis"],
        "quality_relation_mode": f"group_means_min{int(an.get('relation_min_group', 5))}",
        "distribution_notice": "집계 통계만 포함한다. 개별 부품의 측정값·식별자는 포함하지 않는다.",
        "source": rel.get("source"), "source_url": (rel.get("source") or {}).get("url"),
        "limitations": rel.get("limitations", []),
    }
    ctx.write_json(STAGE, "benchmark_public", benchmark)

    # 승격 조건
    g = rel.get("gate") or {}
    checks = []
    if g.get("require_selection_gates", True):
        checks.append({"check": "선택 모델 Gate 1·2 통과", "passed": bool(th["gates"]["gate1_pass"] and th["gates"]["gate2_pass"]),
                       "value": th["gates"]})
    if g.get("holdout_min_recall") is not None:
        checks.append({"check": f"holdout Recall ≥ {g['holdout_min_recall']}",
                       "passed": bool(hsel["Recall"] >= float(g["holdout_min_recall"])), "value": hsel["Recall"]})
    bname = g.get("holdout_beats_baseline")
    if bname:
        b = by_name.get(bname)
        if b is None:
            checks.append({"check": f"기준선 {bname} 존재", "passed": False, "value": None})
        else:
            checks.append({"check": f"holdout PR-AUC > 기준선({b['name']})", "value": [hsel["PR_AUC"], b["PR_AUC"]],
                           "passed": bool(hsel["PR_AUC"] > b["PR_AUC"])})
            checks.append({"check": f"holdout F2 ≥ 기준선({b['name']})", "value": [hsel["F2"], b["F2"]],
                           "passed": bool(hsel["F2"] >= b["F2"])})
    passed = all(c["passed"] for c in checks)
    ctx.write_json(STAGE, "release_check", {"version": version, "passed": passed, "checks": checks,
                                            "model_sha256": model_sha, "test_status": hs["status"]})
    print(f"  번들 {version}: {DISPLAY[th['model']]}, 특징 {len(feats)}개 (필수 {len(required)}, 선택 {len(optional)}), "
          f"우선 검토 기준 {bundle['alert_threshold']:.3f}")
    for c in checks:
        print(f"   {'✓' if c['passed'] else '✗'} {c['check']}: {c['value']}")
    print(f"  승격 조건: {'통과' if passed else '미통과'} → promote.py --run {ctx.run_dir.name}")
