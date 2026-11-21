"""99 report — 모든 단계 산출물로 report.md를 생성한다.

숫자·표는 전부 자동으로 채우고, 해석은 '> 해석:' 자리에 사람이 작성한다.
선행 단계 결과가 없으면 해당 섹션만 건너뛴다.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

import shutil

from src.runio import PRIVATE_FILES, StageError

STAGE = "99_report"
SCI = {"p_value", "FDR_q_value", "mwu_p", "mwu_q", "q_value"}
INT_LIKE = {"n", "n_pos", "TN", "FP", "FN", "TP", "rank", "cv_rank", "n_features", "k"}


def _fmt(col, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "✓" if v else ""
    if isinstance(v, (int, np.integer)):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    if isinstance(v, (float, np.floating)):
        if col in SCI:
            return f"{v:.3e}"
        if "rate" in col.lower():
            return f"{v * 100:.2f}%"
        if col in INT_LIKE or float(v).is_integer() and abs(v) >= 1000:
            return f"{v:.0f}"
        return f"{v:.4f}"
    s = str(v).replace("|", "\\|")
    return s if col == "features" or len(s) <= 300 else s[:300] + "…"


def md_table(df: pd.DataFrame, columns: dict | None = None, code_cols=()) -> str:
    if df is None or len(df) == 0:
        return "_(없음)_\n"
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    names = [(columns or {}).get(c, c) for c in df.columns]   # 서식은 원래 컬럼명 기준
    head = "| " + " | ".join(map(str, names)) + " |"
    sep = "| " + " | ".join("---" for _ in names) + " |"
    body = []
    for row in df.itertuples(index=False):
        cells = []
        for c, v in zip(df.columns, row):
            t = _fmt(c, v)
            cells.append(f"`{t}`" if c in code_cols and t else t)
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body]) + "\n"


def _codes(items, limit=None):
    items = list(items)
    shown = items if limit is None else items[:limit]
    s = ", ".join(f"`{x}`" for x in shown)
    return s + (f" 외 {len(items) - len(shown)}개" if limit and len(items) > limit else "")


NOTE = "\n> **해석:** _(작성)_\n"


def design(anchor, label="설계 근거"):
    return f"\n<sub>{label}: [WORKFLOW.md](../../WORKFLOW.md#{anchor})</sub>\n"


class Doc:
    def __init__(self, ctx):
        self.ctx, self.parts = ctx, []

    def add(self, *xs):
        self.parts.extend(xs)

    def csv(self, stage, name):
        try:
            return self.ctx.read_csv(stage, name)
        except StageError:
            return None

    def json(self, stage, name):
        try:
            return self.ctx.read_json(stage, name)
        except StageError:
            return None


def sec_run(d, cfg):
    man = d.ctx.run_dir / "manifest.json"
    import json
    m = json.loads(man.read_text()) if man.exists() else {}
    c = cfg.columns
    split = d.json("02_split", "split_summary") or {}
    d.add("# 제조 공정 품질위험 예측 — 자동 생성 리포트\n",
          f"> 이 문서의 표와 수치는 `ml_workflow/run.py`가 생성했다. 해석 문단(`> 해석:`)만 직접 작성한다.  \n"
          f"> 생성: {datetime.now():%Y-%m-%d %H:%M} · 실행 폴더: `{d.ctx.run_dir.name}`  \n"
          "> 워크플로우 설계: [WORKFLOW.md](../../WORKFLOW.md)\n",
          "## 실행 정보\n",
          md_table(pd.DataFrame([
              {"항목": "데이터 파일", "값": f"`{m.get('data_file', '')}`".replace(str(d.ctx.run_dir.parents[2]) + "/", "")},
              {"항목": "데이터 SHA-256", "값": f"`{str(m.get('data_sha256', ''))[:16]}…`"},
              {"항목": "설정 SHA-256", "값": f"`{str(m.get('config_sha256', ''))[:16]}…`"},
              {"항목": "타깃", "값": f"`{cfg.target}`"},
              {"항목": "예측 시점", "값": f"`{c['prediction_time']}`"},
              {"항목": "그룹 키", "값": " + ".join(f"`{g}`" for g in c["group"]["columns"])},
              {"항목": "Holdout 분할", "값": f"{split.get('strategy', '')} (holdout {split.get('holdout_size', '')}, seed {split.get('seed', '')})"},
              {"항목": "Python / sklearn", "값": f"{m.get('versions', {}).get('python')} / {m.get('versions', {}).get('scikit-learn')}"},
          ])))


def sec_data(d, cfg):
    eda = d.json("01_eda", "eda_summary")
    if not eda:
        return
    t = eda["target"]
    d.add("\n---\n\n# 데이터 탐색(EDA)\n", design("s00", "컬럼 역할 판정") + design("s01", "EDA 설계"), "## 데이터 샘플 분포\n",
          md_table(pd.DataFrame([
              {"구분": "정상(0)", "개수": t["n_ok"], "비율": f"{t['n_ok'] / eda['rows'] * 100:.2f}%"},
              {"구분": "불량(1)", "개수": t["n_nok"], "비율": f"{t['n_nok'] / eda['rows'] * 100:.2f}%"},
              {"구분": "**전체**", "개수": eda["rows"], "비율": "100%"}])))
    dt = d.csv("01_eda", "dtype_counts")
    d.add("\n## 컬럼 자료형\n", md_table(dt, {"dtype": "자료형", "n": "개수"}))

    ex = d.csv("01_eda", "excluded_summary")
    if ex is not None:
        ex = ex.copy()
        ex["컬럼"] = [c if n <= 8 else f"{n}개 (`00_validate/column_roles.csv` 참고)"
                     for c, n in zip(ex["columns"], ex["n"])]
        ex["컬럼"] = [", ".join(f"`{x.strip()}`" for x in c.split(",")) if "참고" not in c else c
                     for c in ex["컬럼"]]
        d.add("\n## 제외 컬럼\n",
              "제외 기준은 `config/columns.yaml`(역할, 접두사별 측정 단계, 예측 시점)에서 자동 적용된다.\n\n",
              md_table(ex, {"reason": "제외 사유", "prefix": "접두사", "n": "개수", "컬럼": "컬럼"}),
              f"\n**합계:** {int(ex['n'].sum())}개 제외 → 모델 입력 후보 **{eda['n_candidates']}개**\n")
    byp = pd.Series(eda.get("candidates_by_prefix", {})).rename_axis("접두사").reset_index(name="개수")
    d.add("\n## 입력 후보 구성 (접두사별)\n", md_table(byp))
    cand = d.csv("01_eda", "candidate_features")
    if cand is not None:
        d.add("\n<details><summary>입력 후보 전체 목록</summary>\n\n",
              md_table(cand, {"column": "컬럼", "prefix": "접두사", "stage": "측정 단계", "dtype": "자료형"},
                       code_cols=["column"]), "\n</details>\n")
    d.add(NOTE)

    lab = d.csv("01_eda", "label_distribution")
    d.add("\n## 라벨 분포\n",
          md_table(lab, {"label": "라벨", "n_0": "0", "n_1": "1", "n_missing": "결측",
                         "rate_1_observed": "1 비율(관측 기준)"}, code_cols=["label"]))

    g = eda.get("groups", {})
    comp = None
    for f in sorted((d.ctx.run_dir / "01_eda").glob("composition_by_*.csv")):
        comp = pd.read_csv(f, encoding="utf-8-sig")
    d.add("\n## 실험 그룹 구성\n",
          f"그룹 키로 묶은 그룹 {g.get('n_groups')}개 중 정상 고정 {g.get('pure_ok')}개, "
          f"불량 고정 {g.get('pure_nok')}개, 정상·불량 혼합 {g.get('mixed')}개.\n")
    if g.get("warning"):
        d.add(f"\n> ⚠️ {g['warning']}\n")
    if comp is not None:
        d.add("\n", md_table(comp.rename(columns={"n": "행 수", "n_groups": "그룹 수"})))
    d.add(NOTE)


def sec_missing(d):
    eda = d.json("01_eda", "eda_summary")
    if not eda:
        return
    mr = d.csv("01_eda", "missing_rates")
    d.add("\n---\n\n# 결측치 분석\n", design("s01"), "## 결측치 현황\n",
          md_table(pd.DataFrame([
              {"항목": "입력 후보 특징", "개수": eda["n_candidates"]},
              {"항목": "결측이 있는 특징", "개수": eda["n_features_with_missing"]},
              {"항목": "결측이 없는 특징", "개수": eda["n_candidates"] - eda["n_features_with_missing"]},
              {"항목": "전체 결측 셀 수", "개수": int(mr["n_missing"].sum()) if mr is not None else ""}])))
    d.add("\n## 결측률 분포\n", md_table(d.csv("01_eda", "missing_bins"),
                                     {"missing_rate_bin": "결측률", "n_features": "특징 수"}))
    if mr is not None:
        d.add("\n## 변수별 결측률 (결측 있는 특징)\n",
              md_table(mr[mr["n_missing"] > 0], {"feature": "특징", "n_missing": "결측 수",
                                                "missing_rate": "결측률"}, code_cols=["feature"]))
    pat = d.csv("01_eda", "missing_patterns")
    if pat is not None:
        pat = pat.copy()
        pat["missing_features"] = pat["missing_features"].fillna("").map(
            lambda s: _codes([x.strip() for x in s.split(",") if x.strip()]))
        d.add("\n## 결측 패턴\n", md_table(pat, {"pattern": "패턴", "n_rows": "행 수",
                                               "n_missing_features": "행별 결측 특징 수",
                                               "nok_rate": "NOK rate", "missing_features": "결측 특징"}))
        for f in sorted((d.ctx.run_dir / "01_eda").glob("missing_patterns_by_*.csv")):
            d.add(f"\n**패턴 × {f.stem.replace('missing_patterns_by_', '')}**\n\n",
                  md_table(pd.read_csv(f, encoding="utf-8-sig")))
    mt = d.csv("01_eda", "missingness_vs_target")
    if mt is not None:
        mt = mt.copy()
        mt["features"] = mt["features"].map(lambda s: _codes([x.strip() for x in s.split(",")], 6))
        d.add("\n## 결측 여부와 NOK의 관계\n",
              md_table(mt, {"features": "특징(결측 위치가 같은 묶음)", "n_features": "특징 수",
                            "missing_rate": "결측률", "nok_observed": "관측 행 NOK", "nok_missing": "결측 행 NOK",
                            "diff_pp": "차이(%p)"}))
    d.add(NOTE)


def sec_quality(d):
    eda = d.json("01_eda", "eda_summary")
    if not eda:
        return
    d.add("\n---\n\n# 변수 품질 분석\n",
          "전체 데이터 기준 보고용 표다. 실제 제거는 Filter 단계에서 dev 데이터로 다시 판정한다.\n",
          "\n## 상수 변수\n", md_table(d.csv("01_eda", "constant_features"), {"feature": "특징"}, ["feature"]),
          "\n## 완전 중복 변수\n", md_table(d.csv("01_eda", "exact_duplicates"),
                                       {"kept": "유지", "dropped": "제거"}, ["kept", "dropped"]),
          "\n## 근사 중복 쌍 (|Spearman ρ| ≥ 기준)\n",
          md_table(d.csv("01_eda", "near_duplicates"), {"a": "특징 A", "b": "특징 B",
                                                        "spearman_rho": "ρ", "n_obs": "동시 관측 수"}, ["a", "b"]),
          NOTE)


def sec_univariate(d, cfg):
    uni = d.csv("01_eda", "univariate")
    eda = d.json("01_eda", "eda_summary")
    if uni is None:
        return
    n = int(cfg.experiment.get("report", {}).get("top_n_univariate", 30))
    u = eda.get("univariate", {})
    d.add("\n---\n\n# 타깃 연관성 탐색\n",
          "결측은 대체하지 않고 특징값과 타깃이 모두 있는 행만 사용했다. "
          "AUC 구분력 = max(AUC, 1 − AUC). Mann–Whitney U의 p-value에 BH-FDR 보정을 적용했다.\n",
          f"\n- 검정 특징 {u.get('n_tested')}개 중 FDR 보정 p < {u.get('fdr_alpha')}: **{u.get('n_fdr_significant')}개**\n",
          f"\n## 단일 변수 AUC · Mann–Whitney U (상위 {n}개)\n",
          md_table(uni.head(n), {"feature": "특징", "n_obs": "관측 수", "missing_rate": "결측률",
                                 "auc_strength": "AUC 구분력", "nok_tendency": "NOK 경향",
                                 "median_ok": "정상 중앙값", "median_nok": "NOK 중앙값",
                                 "mwu_p": "p-value", "mwu_q": "FDR q"}, ["feature"]),
          f"\n전체 표: `01_eda/univariate.csv`\n", NOTE)


def sec_filter(d, cfg):
    fs = d.json("03_filter", "filter_summary")
    if not fs:
        return
    n = int(cfg.experiment.get("report", {}).get("top_n_ranking", 30))
    fset = d.csv("03_filter", "feature_set")
    dropped = fset[~fset["used"].astype(bool)] if fset is not None else None
    rk = d.csv("03_filter", "filter_ranking")
    d.add("\n---\n\n# 특징 선택(Feature selection)\n", "## Filter — ANOVA F-test + BH-FDR\n", design("s03"),
          f"Holdout을 제외한 **dev {fs['dev_rows']}행**만 사용했다. 결측은 dev 중앙값으로 대체했고, "
          f"결측 indicator는 **{'사용' if fs['missing_indicators'] else '미사용'}**. "
          "값 열과 indicator를 각각 검정하고, 둘 중 하나라도 FDR 기준을 통과하면 원본 특징을 후보로 두며 "
          "통과 항목 중 가장 큰 F 점수로 순위를 매긴다.\n\n",
          md_table(pd.DataFrame([
              {"단계": "입력 후보", "개수": fs["n_candidates"]},
              {"단계": "품질 제거 후 사용 특징", "개수": fs["n_used"]},
              {"단계": "평가 항목 (값 + 결측 indicator)",
               "개수": f"{fs['n_components']} ({fs['n_components_value']} + {fs['n_components_missing']})"},
              {"단계": f"FDR q < {fs['fdr_alpha']} 통과 항목", "개수": fs["n_components_passed"]},
              {"단계": "순위에 오른 특징", "개수": fs["n_features_ranked"]}])))
    if dropped is not None and len(dropped):
        d.add("\n**품질 기준으로 제거된 특징**\n\n",
              md_table(dropped, {"feature": "특징", "reason": "사유", "partner": "유지된 대응 특징"},
                       ["feature", "partner"]))
    if rk is not None:
        rk = rk.copy()
        rk["via"] = np.where(rk["best_component"].str.endswith("__missing"), "결측 indicator", "값")
        d.add(f"\n**특징 순위 (상위 {n}개)**\n\n",
              md_table(rk.head(n), {"rank": "순위", "feature": "특징", "via": "통과 근거",
                                    "F_score": "F 점수", "FDR_q_value": "FDR q"}, ["feature"]),
              f"\n전체: `03_filter/filter_ranking.csv`, 항목별 점수: `03_filter/filter_scores.csv`\n")
        via = rk[rk["via"] == "결측 indicator"]["feature"].tolist()
        if via:
            d.add(f"\n> ⚠️ 결측 indicator로만 순위에 든 특징: {_codes(via)}\n")
    d.add(NOTE)


def sec_wrapper(d):
    w = d.json("04_wrapper", "wrapper_summary")
    if not w:
        return
    comp = d.csv("04_wrapper", "wrapper_comparison")
    ov = d.csv("04_wrapper", "feature_overlap")
    rule = ("1-SE 규칙(최고 PR-AUC − 그 지점의 표준오차 이상인 가장 적은 k)"
            if w.get("k_rule") == "one_se" else
            f"최고 PR-AUC의 {float(w.get('tolerance', 0.01)) * 100:.0f}% 이내에서 가장 적은 k")
    if w.get("min_recall") is not None:
        rule += f", 최소 Recall {w['min_recall']}"
    d.add("\n## Wrapper — 모델별 특징 수 탐색\n", design("s04"),
          f"dev {w['dev_rows']}행({w['dev_groups']}개 그룹)에서 **{w['cv']}**로 평가했다. "
          "매 fold 학습 부분에서 F-test/FDR 순위를 다시 계산했고, **특징을 넣는 순서는 모든 모델이 공유**하되 "
          f"**몇 개를 쓸지(k)는 모델마다** 따로 골랐다. k 선택 규칙: {rule}. "
          f"탐색한 k: `{w.get('k_grid') or '전체'}`.\n")
    if w.get("param_overrides"):
        d.add(f"\n탐색 속도를 위해 Wrapper에서만 바꾼 설정: `{w['param_overrides']}` (모델 단계는 원래 설정 사용)\n")
    cols = {"model_name": "모델", "variant": "결측 indicator", "primary": "채택", "selected_k": "선택 k",
            "PR_AUC_at_k": "선택 k PR-AUC", "PR_AUC_std_at_k": "표준편차", "best_PR_AUC": "최고 PR-AUC",
            "best_at_k": "최고 지점 k", "Recall_at_k": "Recall", "n_selected_via_indicator": "indicator 근거 특징"}
    d.add("\n", md_table(comp, cols))
    if w.get("figure"):
        d.add(f"\n![feature count curves](04_wrapper/{w['figure']})\n")
    if ov is not None:
        models = [c for c in ov.columns if c not in ("rank", "feature", "n_models")]
        from src.models import DISPLAY
        d.add("\n**모델별 선택 특징 (✓ = 사용)**\n\n",
              md_table(ov, {"rank": "순위", "feature": "특징", **{m: DISPLAY.get(m, m) for m in models},
                            "n_models": "사용 모델 수"}, ["feature"]))
    if w.get("common_features"):
        d.add(f"\n모든 모델이 공통으로 쓴 특징: {_codes(w['common_features'])}\n")
    d.add(NOTE)


def sec_model(d):
    ms = d.json("05_model", "model_summary")
    if not ms:
        return
    tr = d.csv("05_model", "selection_trace")
    ho = d.csv("05_model", "holdout_metrics")
    sc = ms.get("selection_config") or {}
    gate, tie = sc.get("gate") or {}, sc.get("tie") or {}
    tgt = gate.get("target_recall")
    from src.models import DISPLAY
    d.add("\n---\n\n# 모델 비교 및 선정 (벤치마크)\n",
          f"비교 모델: {', '.join(DISPLAY[m] for m in ms['models'])}. 각 모델은 Wrapper에서 고른 자기 특징 수"
          f"({', '.join(f'{DISPLAY[m]} {k}개' for m, k in ms['k'].items())})를 사용했다. "
          f"평가는 모두 dev의 그룹 {ms['n_folds']}-fold CV이며, 결측 indicator "
          f"{'사용' if ms['missing_indicators'] else '미사용'}, 확률 보정 "
          f"{'사용(' + ms['calibration']['method'] + ')' if ms.get('calibration') else '미사용'}.\n")
    if ms.get("skipped"):
        d.add(f"\n> 제외된 모델: {', '.join(ms['skipped'])}\n")
    d.add("\n## 선택 규칙\n", design("selection"),
          md_table(pd.DataFrame([
              {"단계": "1. Gate 1 — 불량 탐지",
               "내용": f"CV 예측에서 Recall ≥ {tgt}를 만족하는 판정 기준을 모델마다 찾고, 그때 Precision ≥ {gate.get('min_precision')}"},
              {"단계": "1. Gate 2 — 실험일 일반화",
               "내용": f"실험일을 하나씩 빼고 학습·검증했을 때 CV 판정 기준에서의 Recall ≥ {gate.get('lodo_min_recall')}"
                       + (" (모든 실험일)" if gate.get('lodo_all_parts', True) else " (평균)")},
              {"단계": "2. 주 지표", "내용": f"{sc.get('primary', 'PR_AUC')} fold 평균 (높을수록 좋음)"},
              {"단계": "3. 동점 처리",
               "내용": f"1위와의 차이가 표준오차 이내({tie.get('rule')})면 동점 → "
                       + " → ".join({"lodo_PR_AUC": "실험일 제외 PR-AUC", "Brier": "Brier score",
                                     "interpretability": "해석 가능성"}.get(k, k) for k in tie.get("order", []))
                       + f" (각 기준에서 {tie.get('tolerance')} 이내 차이는 같다고 봄)"},
              {"단계": "해석 가능성 순서", "내용": " > ".join(DISPLAY.get(m, m) for m in sc.get("interpretability", []))},
          ])))
    d.add("\n## 선택 기록표\n",
          md_table(tr, {"final_rank": "순위", "model_name": "모델", "k": "특징 수",
                        "PR_AUC_mean": "PR-AUC 평균", "PR_AUC_std": "표준편차",
                        "threshold_at_target": "판정 기준", "Precision_at_target": f"Precision@R{tgt}",
                        "gate1_pass": "Gate 1", "lodo_recall_min": "실험일 최소 Recall", "gate2_pass": "Gate 2",
                        "in_tie_set": "동점 집합", "lodo_PR_AUC": "실험일 제외 PR-AUC",
                        "Brier": "Brier", "interpretability_rank": "해석 순서", "result": "결과"}),
          f"\n**선택 모델: {ms['selected_model_name']}** — {ms['selection_reason']}\n")
    if ms.get("figure"):
        d.add(f"\n![model selection](05_model/{ms['figure']})\n")

    b = ms["risk_bands"]
    ob = d.csv("05_model", "oof_bands")
    d.add("\n## 품질위험도 구간\n",
          f"선택 모델의 CV 예측(OOF)으로 기준을 정했다. **품질주의**: 위험도 ≥ {b['caution']:.3f} "
          f"(Recall {b['caution_recall']} 지점), **불량 위험**: 위험도 ≥ {b['defect']:.3f} (Recall {b['defect_recall']} 지점).\n\n",
          "**CV(dev) 구간별 실제 불량률**\n\n",
          md_table(ob, {"band": "구간", "n": "행 수", "n_nok": "실제 불량", "nok_rate": "불량 rate"}))

    hs = ms.get("holdout_status") or {}
    dev_ref = hs.get("status") == "previously_seen_holdout_development_reference"
    d.add("\n## Holdout 평가" + (" — 개발 참고 평가" if dev_ref else "") + "\n",
          (f"\n> ⚠️ **개발 참고 평가:** {hs.get('note')} 결과를 모델 성능의 독립 검증으로 해석하지 않는다.\n\n"
           if dev_ref else ""),
          f"Holdout {ms['holdout_rows']}행은 그룹 단위로 분리되어 특징·모델·기준 선택에 쓰지 않았다. "
          "각 모델에 **CV에서 정한 판정 기준을 고정 적용**했다. 모든 모델을 기록하지만 선택은 CV 결과로만 했다.\n\n",
          md_table(ho, {"model_name": "모델", "selected": "선택", "k": "특징 수", "threshold": "판정 기준",
                        "PR_AUC": "PR-AUC", "ROC_AUC": "ROC-AUC", "Brier": "Brier", "Recall": "Recall",
                        "Precision": "Precision", "F2": "F2", "Alert_rate": "우선 검토 rate",
                        "TP": "TP", "FN": "FN", "FP": "FP", "TN": "TN"}))
    bl = d.csv("05_model", "baselines_holdout")
    if bl is not None:
        d.add("\n**기준선 비교 (Holdout)**\n\n",
              md_table(bl, {"name": "기준선", "PR_AUC": "PR-AUC", "Recall": "Recall", "Precision": "Precision",
                            "F2": "F2", "Alert_rate": "우선 검토 rate", "TP": "TP", "FN": "FN", "FP": "FP", "TN": "TN"}))
    h = ms["holdout_selected"]
    hb = d.csv("05_model", "holdout_bands")
    d.add("\n**선택 모델 혼동행렬 (Holdout, 판정 기준 고정)**\n\n",
          md_table(pd.DataFrame([{"": "실제 불량", "예측 불량": h["TP"], "예측 정상": h["FN"]},
                                 {"": "실제 정상", "예측 불량": h["FP"], "예측 정상": h["TN"]}])),
          "\n**Holdout 구간별 실제 불량률**\n\n",
          md_table(hb, {"band": "구간", "n": "행 수", "n_nok": "실제 불량", "nok_rate": "불량 rate"}))
    if ms.get("leave_one_out_by"):
        lo = d.csv("05_model", "leave_one_out")
        d.add(f"\n## 일반화 진단 — `{ms['leave_one_out_by']}` 하나씩 제외\n",
              f"dev 안에서 `{ms['leave_one_out_by']}`의 한 값을 통째로 검증용으로 빼고 나머지로 순위 계산·학습했다. "
              "Recall·Precision은 CV에서 정한 판정 기준을 그대로 적용한 값이다.\n\n",
              md_table(lo, {"model_name": "모델", "held_out": "제외한 값", "n_test": "검증 행",
                            "nok_rate_test": "검증 NOK rate", "PR_AUC": "PR-AUC", "ROC_AUC": "ROC-AUC",
                            "Recall_at_cv_threshold": "Recall", "Precision_at_cv_threshold": "Precision"}))
    d.add(NOTE)


def sec_explain(d):
    es = d.json("06_explain", "explain_summary")
    if not es:
        return
    imp = d.csv("06_explain", "feature_importance")
    rg = d.csv("06_explain", "normal_ranges")
    lo, hi = es["percentiles"]
    d.add("\n---\n\n# 주요 영향변수 및 정상범위\n", design("s06"),
          f"선택 모델({es['model_name']})을 dev 전체로 다시 학습했다(holdout PR-AUC {es['holdout_check']['PR_AUC']:.3f}). "
          "중요도는 holdout에서 특징을 무작위로 섞었을 때 PR-AUC가 줄어드는 정도(permutation importance)이고, "
          "방향은 dev에서 특징값과 CV 예측 위험도의 순위 상관 부호다.\n\n",
          md_table(imp, {"rank": "순위", "feature": "특징", "importance_mean": "중요도",
                         "importance_std": "표준편차", "direction": "방향",
                         "spearman_with_risk": "위험도와 상관"}, ["feature"]))
    if es.get("figure"):
        d.add(f"\n![feature importance](06_explain/{es['figure']})\n")
    d.add(f"\n**정상범위 (dev 정상 샘플 {lo}~{hi} 백분위)**\n\n",
          md_table(rg, {"feature": "특징", f"ok_p{lo}": f"하한(p{lo})", "ok_median": "정상 중앙값",
                        f"ok_p{hi}": f"상한(p{hi})", "nok_median": "불량 중앙값",
                        "nok_outside_rate": "불량 중 범위 밖 rate", "missing_rate": "결측 rate"}, ["feature"]))
    gp = es.get("group_part")
    if gp:
        bp = d.csv("06_explain", f"normal_ranges_by_{gp}")
        if bp is not None:
            d.add(f"\n**`{gp}`별 정상범위**\n\n", md_table(bp))
    d.add("\nMVP용 패키지: `06_explain/model.joblib`, `06_explain/model_card.json` "
          "(예측: `python predict.py <실행 폴더> <입력 파일>`)\n", NOTE)


def publish(run_dir) -> int:
    """행 단위 산출물과 코드 사본을 뺀 집계 결과를 run_dir/public/에 복사한다."""
    pub = run_dir / "public"
    if pub.exists():
        shutil.rmtree(pub)
    n = 0
    for f in sorted(run_dir.rglob("*")):
        rel = f.relative_to(run_dir).as_posix()
        if (not f.is_file() or rel.split("/")[0] in ("public", "code") or rel in PRIVATE_FILES
                or f.name.startswith("_ckpt_")):
            continue
        dst = pub / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        n += 1
    return n


def run(ctx):
    cfg = ctx.cfg
    d = Doc(ctx)
    sec_run(d, cfg)
    sec_data(d, cfg)
    sec_missing(d)
    sec_quality(d)
    sec_univariate(d, cfg)
    sec_filter(d, cfg)
    sec_wrapper(d)
    sec_model(d)
    sec_explain(d)
    d.add("\n---\n\n# 결과 해석 및 한계\n", design("limitations", "알려진 한계"), "\n_(작성)_\n")
    path = ctx.run_dir / "report.md"
    path.write_text("\n".join(d.parts), encoding="utf-8")
    n_public = publish(ctx.run_dir)
    print(f"  공개용 사본: public/ ({n_public}개 파일, 행 단위 산출물 제외)")
    print(f"  {path.relative_to(ctx.run_dir.parent.parent)} ({len(path.read_text(encoding='utf-8').splitlines())}줄)")
