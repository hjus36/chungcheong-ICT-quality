"""학습·예측·지표 계산.

모든 모델은 [중앙값 대체(+결측 indicator) → (스케일링) → 모델] 파이프라인으로 만든다.
calibration을 주면 학습 데이터 안에서 CalibratedClassifierCV로 확률을 보정한다.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import SCALED, make_estimator


def build_pipeline(name, params, y_train, seed, missing_indicators, calibration=None):
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=missing_indicators))]
    if name in SCALED:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", make_estimator(name, params, y_train, seed)))
    pipe = Pipeline(steps)
    if calibration:
        return CalibratedClassifierCV(pipe, method=calibration.get("method", "sigmoid"),
                                      cv=int(calibration.get("cv", 3)))
    return pipe


def scores_of(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def fit_predict(name, params, X_train, y_train, X_valid, features, missing_indicators, seed,
                calibration=None):
    """features만 사용해 학습 → (검증 점수, 기본 판정, 학습된 모델)."""
    model = build_pipeline(name, params, y_train, seed, missing_indicators, calibration)
    model.fit(X_train[features], y_train)
    Xv = X_valid[features]
    return scores_of(model, Xv), model.predict(Xv), model


def _is_prob(scores):
    s = np.asarray(scores, dtype=float)
    return bool(np.all((s >= 0) & (s <= 1)))


def rank_metrics(y, scores) -> dict:
    """판정 기준과 무관한 지표: PR-AUC, ROC-AUC, Brier(확률일 때)."""
    y = np.asarray(y).astype(int)
    has_pos, both = y.sum() > 0, len(np.unique(y)) == 2
    return {
        "n": len(y), "n_pos": int(y.sum()),
        "PR_AUC": float(average_precision_score(y, scores)) if has_pos else np.nan,
        "ROC_AUC": float(roc_auc_score(y, scores)) if both else np.nan,
        "Brier": float(brier_score_loss(y, scores)) if _is_prob(scores) else np.nan,
    }


def threshold_metrics(y, scores, threshold) -> dict:
    """점수 ≥ threshold 를 불량으로 판정했을 때의 지표."""
    y = np.asarray(y).astype(int)
    preds = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    return {
        "Recall": float(recall_score(y, preds, zero_division=0)) if y.sum() > 0 else np.nan,
        "Precision": float(precision_score(y, preds, zero_division=0)),
        "F1": float(f1_score(y, preds, zero_division=0)),
        "Balanced_Acc": float(balanced_accuracy_score(y, preds)) if len(np.unique(y)) == 2 else np.nan,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def metrics(y, scores, preds) -> dict:
    """순위 지표 + 모델 기본 판정(predict) 기준 지표."""
    y = np.asarray(y).astype(int)
    preds = np.asarray(preds).astype(int)
    out = rank_metrics(y, scores)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    out.update({
        "Recall": float(recall_score(y, preds, zero_division=0)) if y.sum() > 0 else np.nan,
        "Precision": float(precision_score(y, preds, zero_division=0)),
        "F1": float(f1_score(y, preds, zero_division=0)),
        "Balanced_Acc": float(balanced_accuracy_score(y, preds)) if len(np.unique(y)) == 2 else np.nan,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    })
    return out


def threshold_at_recall(y, scores, target):
    """Recall ≥ target 을 만족하는 가장 높은 기준 → (기준, 그때 Precision, 그때 Recall)."""
    y = np.asarray(y).astype(int)
    p, r, thr = precision_recall_curve(y, scores)
    ok = np.flatnonzero(r[:-1] >= target)
    i = ok[-1] if len(ok) else 0
    return float(thr[i]), float(p[i]), float(r[i])
