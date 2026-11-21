"""컬럼 역할 판정 — columns.yaml 규칙을 실제 데이터 컬럼에 적용한다."""
from __future__ import annotations

import pandas as pd


def prefix_of(column: str) -> str | None:
    return column.split("_", 1)[0] if "_" in column else None


def allowed_stages(cfg) -> set[str]:
    c = cfg.columns
    order = c["stage_order"]
    point = c["prediction_time"]
    if point not in order:
        raise ValueError(f"prediction_time {point!r}이 stage_order에 없습니다.")
    allowed = set(order[: order.index(point) + 1])
    if c.get("allow_offline"):
        allowed.add("offline")
    return allowed


def resolve_columns(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """모든 컬럼의 역할·측정 단계·포함 여부·사유를 표로 만든다."""
    c = cfg.columns
    roles = c.get("roles") or {}
    target = c["target"]
    ids = set(roles.get("id") or [])
    times = set(roles.get("time") or [])
    meta = set(roles.get("meta") or [])
    label_prefixes = tuple(roles.get("label_prefixes") or [])
    prefix_stage = c.get("prefix_stage") or {}
    overrides = c.get("column_overrides") or {}
    manual = {d["column"]: d.get("reason", "") for d in (c.get("manual_drop") or [])}
    allowed = allowed_stages(cfg)

    rows = []
    for col in df.columns:
        prefix = prefix_of(col)
        override = overrides.get(col) or {}
        stage = override.get("stage", prefix_stage.get(prefix))
        numeric = pd.api.types.is_numeric_dtype(df[col])
        status, role = "excluded", "feature"

        if col == target:
            role, reason = "target", "타깃"
        elif col in ids:
            role, reason = "id", "식별자"
        elif col in times:
            role, reason = "time", "시간 정보"
        elif label_prefixes and col.startswith(label_prefixes):
            role, reason = "label", "사후 결과 라벨"
        elif col in meta:
            role, reason = "meta", "메타(그룹·해석용)"
        elif stage is None:
            role, reason = "unclassified", "측정 단계 미분류"
        elif col in manual:
            reason = f"수동 제외: {manual[col]}"
        elif stage not in allowed:
            reason = ("offline 단계 비허용" if stage == "offline"
                      else f"예측 시점 이후 측정({stage})")
        elif not numeric:
            reason = "비숫자형(시계열 배열·문자열)"
        else:
            status, reason = "candidate", "입력 후보"
        if override.get("reason"):
            reason = f"{reason} / override: {override['reason']}"

        rows.append({
            "column": col, "prefix": prefix, "dtype": str(df[col].dtype),
            "numeric": numeric, "role": role, "stage": stage,
            "status": status, "reason": reason,
        })
    return pd.DataFrame(rows)


def candidate_columns(roles: pd.DataFrame) -> list[str]:
    return roles.loc[roles["status"] == "candidate", "column"].tolist()
