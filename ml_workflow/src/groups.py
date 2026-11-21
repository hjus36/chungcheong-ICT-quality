"""그룹 키 생성."""
from __future__ import annotations

import pandas as pd


def group_parts(df: pd.DataFrame, cfg) -> pd.DataFrame:
    g = cfg.columns["group"]
    transforms = g.get("transforms") or {}
    parts = {}
    for col in g["columns"]:
        s = df[col]
        t = transforms.get(col)
        if t == "date":
            s = pd.to_datetime(s).dt.strftime("%Y-%m-%d")
        elif t is not None:
            raise ValueError(f"지원하지 않는 group transform: {t}")
        parts[col] = s.astype(str)
    return pd.DataFrame(parts, index=df.index)


def make_groups(df: pd.DataFrame, cfg) -> pd.Series:
    parts = group_parts(df, cfg)
    return parts.agg("|".join, axis=1).rename("group")
