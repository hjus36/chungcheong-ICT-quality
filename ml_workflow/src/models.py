"""모델 생성. 이름 → 추정기. 하이퍼파라미터는 experiment.yaml의 model.params."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

DISPLAY = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
           "xgboost": "XGBoost", "svm": "SVM"}
SCALED = {"logistic_regression", "svm"}   # 스케일링이 필요한 모델


def is_available(name: str) -> bool:
    if name == "xgboost":
        try:
            import xgboost  # noqa: F401
            return True
        except ImportError:
            return False
    return name in DISPLAY


def make_estimator(name: str, params: dict | None, y_train, seed: int):
    p = dict(params or {})
    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, **p)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **p)
    if name == "svm":
        return SVC(random_state=seed, **p)
    if name == "xgboost":
        from xgboost import XGBClassifier
        pos, neg = int((y_train == 1).sum()), int((y_train == 0).sum())
        p.setdefault("scale_pos_weight", neg / max(pos, 1))
        return XGBClassifier(random_state=seed, **p)
    raise ValueError(f"알 수 없는 모델 이름: {name}")
