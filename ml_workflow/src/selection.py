"""모델 선택 규칙: gate → primary → 1-SE 동점 집합 → tie 순서."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import DISPLAY
from .runio import StageError

DIRECTION = {"lodo_PR_AUC": "max", "Brier": "min", "PR_AUC_mean": "max", "ROC_AUC_mean": "max",
             "Precision_at_target": "max"}
KEY_LABEL = {"lodo_PR_AUC": "실험일 제외 PR-AUC", "Brier": "Brier score",
             "interpretability": "해석 가능성 순서", "primary": "주 지표"}


def select_model(table: pd.DataFrame, sel: dict, n_folds: int):
    """table: 모델별 한 행 (model, PR_AUC_mean, PR_AUC_std, Precision_at_target, lodo_PR_AUC, Brier ...)

    반환: (선택 기록표, 선택 모델, 선택 이유 문장)
    """
    t = table.copy().reset_index(drop=True)
    gate = sel.get("gate") or {}
    min_p = gate.get("min_precision")
    t["gate_pass"] = (t["Precision_at_target"] >= float(min_p)) if min_p is not None else True
    gate_skipped = False
    if not t["gate_pass"].any():
        if gate.get("if_none_pass", "skip_gate") == "stop":
            raise StageError("gate를 통과한 모델이 없습니다 (if_none_pass=stop).")
        gate_skipped = True
    eligible = t["gate_pass"] | gate_skipped

    prim = f"{sel.get('primary', 'PR_AUC')}_mean"
    std = prim.replace("_mean", "_std")
    pool = t[eligible]
    best = pool.loc[pool[prim].idxmax()]
    tie = sel.get("tie") or {}
    floor = (best[prim] - best[std] / np.sqrt(n_folds)) if tie.get("rule", "one_se") == "one_se" \
        else best[prim] - 1e-12
    t["tie_floor"] = floor
    t["in_tie_set"] = eligible & (t[prim] >= floor - 1e-12)

    order = sel.get("interpretability") or t["model"].tolist()
    t["interpretability_rank"] = t["model"].map(lambda m: order.index(m) + 1 if m in order else len(order) + 1)

    cands = t[t["in_tie_set"]].copy()
    tol = float(tie.get("tolerance", 0.0))
    eliminated = {}          # 모델 → 밀려난 기준
    decided_by = "primary" if len(cands) == 1 else None
    if decided_by is None:
        for key in (tie.get("order") or []) + ["interpretability"]:
            if key == "interpretability":
                keep = cands.nsmallest(1, "interpretability_rank")
            else:
                vals = cands[key]
                if vals.isna().all():
                    continue
                if DIRECTION.get(key, "max") == "max":
                    keep = cands[vals >= vals.max() - tol]
                else:
                    keep = cands[vals <= vals.min() + tol]
            for m in set(cands["model"]) - set(keep["model"]):
                eliminated[m] = key
            cands = keep
            if len(cands) == 1:
                decided_by = key
                break
    chosen = cands.iloc[0]["model"]

    def note(r):
        if r["model"] == chosen:
            return "선택"
        if not r["gate_pass"] and not gate_skipped:
            return f"gate 탈락 (Precision@Recall {r['Precision_at_target']:.2f} < {min_p})"
        if not r["in_tie_set"]:
            return f"{sel.get('primary', 'PR_AUC')} 평균이 동점 기준({floor:.4f}) 미만"
        return f"동점 집합 내, {KEY_LABEL.get(eliminated.get(r['model']), eliminated.get(r['model']))}에서 밀림"

    t["result"] = t.apply(note, axis=1)
    t["selected"] = t["model"] == chosen
    t = pd.concat([t[t["selected"]], t[~t["selected"]].sort_values(prim, ascending=False)])
    t.insert(0, "final_rank", np.arange(1, len(t) + 1))

    tied = t[t["in_tie_set"]]["model"].map(DISPLAY).tolist()
    parts = []
    if gate_skipped:
        parts.append("gate를 통과한 모델이 없어 gate를 생략")
    else:
        failed = t[~t["gate_pass"]]["model"].map(DISPLAY).tolist()
        parts.append(f"gate 통과 {int(t['gate_pass'].sum())}/{len(t)}" + (f" (탈락: {', '.join(failed)})" if failed else ""))
    if len(tied) == 1:
        parts.append(f"{sel.get('primary', 'PR_AUC')} 평균 1위, 동점 모델 없음")
    else:
        parts.append(f"{sel.get('primary', 'PR_AUC')} 동점 집합(1-SE): {', '.join(tied)}")
        parts.append(f"{KEY_LABEL.get(decided_by, decided_by)} 기준으로 결정")
    return t.reset_index(drop=True), chosen, " → ".join(parts)
