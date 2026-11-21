"""00 validate — 스키마 검사. 실패하면 이후 단계를 실행하지 않는다."""
from __future__ import annotations

from src.columns import allowed_stages, resolve_columns
from src.groups import make_groups
from src.runio import StageError

STAGE = "00_validate"


def run(ctx):
    df, cfg = ctx.df, ctx.cfg
    c = cfg.columns
    errors, warnings = [], []

    # 단계 이름 검사
    known = set(c["stage_order"]) | {"offline"}
    bad = {p: s for p, s in (c.get("prefix_stage") or {}).items() if s not in known}
    for col, o in (c.get("column_overrides") or {}).items():
        if o.get("stage") and o["stage"] not in known:
            bad[col] = o["stage"]
    if bad:
        errors.append(f"정의되지 않은 측정 단계: {bad}")
    try:
        allowed = sorted(allowed_stages(cfg))
    except ValueError as e:
        errors.append(str(e))
        allowed = []

    # 타깃
    target = cfg.target
    if target not in df.columns:
        errors.append(f"타깃 컬럼 없음: {target}")
    else:
        values = set(df[target].dropna().unique().tolist())
        if not values <= {0, 1} or len(values) < 2:
            errors.append(f"타깃이 0/1 이진값이 아닙니다: {sorted(values)}")
        n_na = int(df[target].isna().sum())
        if n_na:
            warnings.append(f"타깃 결측 {n_na}행은 분석에서 제외됩니다.")

    # 그룹 키
    gcols = c["group"]["columns"]
    missing_g = [g for g in gcols if g not in df.columns]
    if missing_g:
        errors.append(f"그룹 키 컬럼 없음: {missing_g}")
    else:
        if df[gcols].isna().any().any():
            errors.append("그룹 키에 결측이 있습니다.")
        else:
            groups = make_groups(df, cfg)
            if groups.nunique() < 5:
                warnings.append(f"그룹 수가 적습니다: {groups.nunique()}개")

    # 설정에 적었지만 데이터에 없는 컬럼
    roles = c.get("roles") or {}
    listed = (roles.get("id") or []) + (roles.get("time") or []) + (roles.get("meta") or [])
    listed += list((c.get("column_overrides") or {}).keys())
    listed += [d["column"] for d in (c.get("manual_drop") or [])]
    absent = [x for x in listed if x not in df.columns]
    if absent:
        warnings.append(f"설정에 있으나 데이터에 없는 컬럼: {absent}")

    # 컬럼 역할 판정 — 미분류 컬럼은 오류
    table = resolve_columns(df, cfg)
    ctx.write_csv(STAGE, "column_roles", table)
    unclassified = table.loc[table["role"] == "unclassified", "column"].tolist()
    if unclassified:
        new_prefixes = sorted({str(x) for x in
                               table.loc[table["role"] == "unclassified", "prefix"]})
        errors.append(
            f"측정 단계가 정해지지 않은 컬럼 {len(unclassified)}개 "
            f"(접두사: {new_prefixes}). columns.yaml의 prefix_stage 또는 "
            f"column_overrides에 추가하세요. 목록: column_roles.csv")
    n_cand = int((table["status"] == "candidate").sum())
    if n_cand == 0:
        errors.append("입력 후보 특징이 0개입니다.")

    summary = {
        "rows": len(df), "columns": df.shape[1], "data_file": ctx.data_file.name,
        "prediction_time": c["prediction_time"], "allowed_stages": allowed,
        "n_candidates": n_cand,
        "by_status_reason": table.groupby(["status", "reason"]).size()
                                 .rename("n").reset_index().to_dict("records"),
        "errors": errors, "warnings": warnings,
    }
    ctx.write_json(STAGE, "validation", summary)

    print(f"  행 {len(df)}, 컬럼 {df.shape[1]}, 예측 시점 {c['prediction_time']} "
          f"→ 입력 후보 {n_cand}개")
    for w in warnings:
        print(f"  [경고] {w}")
    for e in errors:
        print(f"  [오류] {e}")
    if errors:
        raise StageError("검증 실패")
