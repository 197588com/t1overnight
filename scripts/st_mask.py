# -*- coding: utf-8 -*-
"""Point-in-time ST / delisting mask rebuilt from tushare ``namechange``.

Why not the current name: a stock that is *ST today may have been clean in 2022,
and a stock that was *ST in 2025 may have cleaned up since.  Judging history with
the current name introduces both false exclusions and false inclusions (measured
on the 1,161-day top1 sample: 39 days wrongly dropped, 37 days wrongly kept).

Why not ``st_proxy``: it infers the limit band from realised volatility, so it
cannot see an ST tag on ChiNext/STAR names (5% band vs 20%) until the stock has
already traded a few ST sessions.

``namechange`` gives (code, name, start_date, end_date) for every rename since
2015, which is enough to reconstruct the name in force on any date.  A stock with
no record at all is treated as never tagged.

usage
-----
    from st_mask import load_intervals, annotate
    iv = load_intervals()
    df = annotate(df, iv)          # adds is_st, is_delist (bool, point-in-time)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "namechange.parquet"
CACHE = ROOT / "outputs" / "st_intervals.parquet"


def _norm_code(ts_code: pd.Series) -> pd.Series:
    """002731.SZ -> 002731 (the dataset uses bare 6-digit codes)."""
    return ts_code.astype(str).str.split(".").str[0].str.zfill(6)


def _flags(name: pd.Series) -> pd.DataFrame:
    up = name.fillna("").str.upper().str.replace(" ", "", regex=False)
    # PT was the pre-2007 predecessor of ST; keep it for completeness.
    is_st = up.str.contains("ST") | up.str.startswith("PT")
    # 退市整理期 / 已摘牌: also unbuyable, tracked separately.
    is_delist = up.str.contains("退")
    return pd.DataFrame({"is_st": is_st.to_numpy(), "is_delist": is_delist.to_numpy()})


def build_intervals(src: Path = SRC) -> pd.DataFrame:
    nc = pd.read_parquet(src)
    nc = nc[nc["name"].notna()].copy()
    nc["code"] = _norm_code(nc["ts_code"])
    nc["start"] = pd.to_datetime(nc["start_date"], format="%Y%m%d", errors="coerce")
    nc["end"] = pd.to_datetime(nc["end_date"], format="%Y%m%d", errors="coerce")
    nc = nc[nc["start"].notna()]
    nc = nc.join(_flags(nc["name"]))
    nc = nc.sort_values(["code", "start"], kind="mergesort")

    # A rename record's end_date is often blank (still in force).  Close the gap
    # with the next record's start so the intervals tile the whole history.
    nxt = nc.groupby("code", sort=False)["start"].shift(-1)
    nc["end"] = nc["end"].fillna(nxt - pd.Timedelta(days=1))
    nc["end"] = nc["end"].fillna(pd.Timestamp("2100-01-01"))
    nc = nc[nc["end"] >= nc["start"]]

    iv = nc[["code", "start", "end", "is_st", "is_delist", "name"]].reset_index(drop=True)
    return iv


def load_intervals(rebuild: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)
    iv = build_intervals()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    iv.to_parquet(CACHE, index=False, compression="zstd")
    return iv


def _to_int_code(s: pd.Series) -> np.ndarray:
    """6-digit code -> int32, so the as-of join runs on a 4-byte key."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(-1).astype("int32").to_numpy()
    v = s.astype(str)
    if not v.str.len().eq(6).all():          # a CSV read can eat leading zeros
        v = v.str.zfill(6)
    return pd.to_numeric(v, errors="coerce").fillna(-1).astype("int32").to_numpy()


def annotate(df: pd.DataFrame, iv: pd.DataFrame | None = None,
             date_col: str = "date", code_col: str = "code",
             with_name: bool = False, copy: bool = False) -> pd.DataFrame:
    """Add point-in-time ``is_st`` / ``is_delist``.

    ``df`` must be sorted by (date, code) -- the as-of merge requires it and we
    avoid re-sorting an 8-million-row frame.  Writes into ``df`` unless
    ``copy=True``: at 8.7m rows a defensive copy costs several GB.
    """
    iv = load_intervals() if iv is None else iv
    out = df.copy() if copy else df
    dates = pd.to_datetime(out[date_col])

    left = pd.DataFrame({"_row": np.arange(len(out), dtype="int64"),
                         "date": dates.to_numpy(),
                         "code": _to_int_code(out[code_col])})
    if not left["date"].is_monotonic_increasing:
        raise ValueError("annotate() needs df sorted by (date, code)")

    right = pd.DataFrame({"date": iv["start"].to_numpy(),
                          "code": _to_int_code(iv["code"]),
                          "end": iv["end"].to_numpy(),
                          "is_st": iv["is_st"].to_numpy(),
                          "is_delist": iv["is_delist"].to_numpy()})
    if with_name:
        right["name"] = iv["name"].to_numpy()
    right = right.sort_values(["date", "code"], kind="mergesort")

    m = pd.merge_asof(left, right, on="date", by="code", direction="backward")
    hit = m["end"].notna().to_numpy() & (m["date"].to_numpy() <= m["end"].to_numpy())

    # cast hard to bool: merge_asof back-fills unmatched rows with NaN, which
    # silently turns the column into object dtype and breaks boolean indexing
    out["is_st"] = hit & m["is_st"].fillna(False).to_numpy(dtype=bool, na_value=False)
    out["is_delist"] = hit & m["is_delist"].fillna(False).to_numpy(dtype=bool, na_value=False)
    if with_name:
        out["name_pit"] = m["name"].where(hit).to_numpy()
    return out


def selftest() -> None:
    iv = load_intervals(rebuild=True)
    print(f"[intervals] {len(iv):,} rows, {iv.code.nunique():,} codes, "
          f"{iv.start.min().date()} -> {iv.end.max().date()}")
    print(f"[intervals] ST 期间 {int(iv.is_st.sum()):,} 段, "
          f"退市整理 {int(iv.is_delist.sum()):,} 段")

    probe = pd.DataFrame({"code": ["000004", "300742", "600165", "000001"],
                          "date": pd.to_datetime(["2023-06-15", "2024-06-15",
                                                  "2026-06-15", "2023-06-15"])})
    probe = probe.sort_values(["date", "code"]).reset_index(drop=True)
    print(annotate(probe, iv, copy=True, with_name=True).to_string(index=False))

    # agreement check against the current-name shortcut on the real top1 sample
    t1 = ROOT / "outputs" / "exit_mode" / "top1_with_st_flag.csv"
    if t1.exists():
        d = pd.read_csv(t1, parse_dates=["date"], dtype={"code": str})
        d = d.sort_values(["date", "code"]).reset_index(drop=True)
        cur = d["is_st"].to_numpy()                       # shortcut: today's name
        if "st_hist" in d:
            ref = (pd.to_numeric(d["st_hist"], errors="coerce").fillna(0) > 0).to_numpy()
        else:
            ref = cur
        a = annotate(d, iv, copy=True, with_name=True)
        new = a.is_st.to_numpy()
        print(f"[check] top1 {len(a):,} 天 | 当时名称 ST {int(new.sum())} 天 "
              f"({new.mean()*100:.2f}%) | 现存 st_hist 重建 {int(ref.sum())} 天 | "
              f"当前名称 {int(cur.sum())} 天")
        print(f"[check] 与 st_hist 重建不一致 {int((new != ref).sum())} 天；"
              f"与当前名称不一致 {int((new != cur).sum())} 天")
        bad = a[new != ref]
        if len(bad):
            print(bad[["date", "code", "name_pit", "st_hist"]].head(8).to_string(index=False))


if __name__ == "__main__":
    selftest()
