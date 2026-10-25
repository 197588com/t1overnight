# -*- coding: utf-8 -*-
"""v3 feature block — three new families, all strictly causal.

A. microstructure shape  (vwap deviation, shadows, drawdown, down-volume pressure)
B. industry relative strength  (industry momentum + stock-minus-industry)
C. market regime  (limit-up/down breadth, dispersion, new-high share, market trend)

Rules
-----
* Every column at date D is built from bars <= D only.
* Industry is the *first ever announced* classification for a code, so it cannot
  contain a later reclassification (no look-ahead, at the price of some staleness).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12

SHAPE = ["vwap_dev", "upper_shadow", "lower_shadow", "body",
         "dd_20", "close_pos_5", "down_vol_ratio", "turn_z", "ret_accel"]

IND = ["ind_ret_1", "ind_ret_5", "ind_ret_20", "ind_rel_5", "ind_rel_20",
       "ind_rank_5", "ind_strength"]

MKT = ["lu_ratio", "ld_ratio", "mkt_ret_5", "mkt_ret_20", "mkt_vol_20",
       "mkt_disp", "mkt_nh_ratio", "mkt_close_pos", "mkt_amt_z"]

V3 = SHAPE + IND + MKT


def _roll(x: np.ndarray, w: int, how: str, minp: int | None = None) -> np.ndarray:
    minp = minp if minp is not None else max(2, w // 2)
    s = pd.Series(x)
    r = getattr(s.rolling(w, min_periods=minp), how)()
    return r.to_numpy(dtype=np.float32)


def add_shape(df: pd.DataFrame) -> pd.DataFrame:
    """Price-action shape features. `df` gets sorted by (code, date) in place."""
    df = df.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)

    c = df["close"].to_numpy(np.float64)
    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    v = df["volume"].to_numpy(np.float64)
    amt = df["amount"].to_numpy(np.float64)
    turn = df["turn"].to_numpy(np.float64) if "turn" in df else df["turnover"].to_numpy(np.float64)
    r1 = df["ret_1"].to_numpy(np.float64)

    prev_c = c / (1.0 + r1)
    vwap = np.where(v > 0, amt / np.maximum(v, EPS), np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        df["vwap_dev"] = (c / vwap - 1.0).astype(np.float32)
        df["upper_shadow"] = ((h - np.maximum(o, c)) / prev_c).astype(np.float32)
        df["lower_shadow"] = ((np.minimum(o, c) - lo) / prev_c).astype(np.float32)
        df["body"] = ((c - o) / prev_c).astype(np.float32)

    n = len(df)
    out = {k: np.full(n, np.nan, dtype=np.float32)
           for k in ["dd_20", "close_pos_5", "down_vol_ratio", "turn_z", "ret_accel"]}

    code = df["code"].to_numpy()
    uniq, starts = np.unique(code, return_index=True)
    order = np.argsort(starts, kind="mergesort")
    starts = starts[order]
    ends = np.append(starts[1:], n)

    for s, e in zip(starts, ends):
        if e - s < 25:
            continue
        cc, vv, tt = c[s:e], v[s:e], turn[s:e]
        rr = r1[s:e]
        cp = df["close_pos"].to_numpy(np.float64)[s:e]

        rm = _roll(cc, 20, "max", 10)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["dd_20"][s:e] = cc / rm - 1.0
        out["close_pos_5"][s:e] = _roll(cp, 5, "mean", 3)

        # volume on down days vs up days, 20-bar average
        upv = np.where(rr > 0, vv, 0.0)
        dnv = np.where(rr < 0, vv, 0.0)
        upm = _roll(upv, 20, "mean", 10) * 20
        dnm = _roll(dnv, 20, "mean", 10) * 20
        with np.errstate(divide="ignore", invalid="ignore"):
            out["down_vol_ratio"][s:e] = dnm / (upm + dnm + EPS)

        tm = _roll(tt, 20, "mean", 10)
        ts = _roll(tt, 20, "std", 10)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["turn_z"][s:e] = (tt - tm) / (ts + EPS)

        r5 = df["ret_5"].to_numpy(np.float64)[s:e]
        out["ret_accel"][s:e] = (rr - r5 / 5.0).astype(np.float32)

    for k, val in out.items():
        df[k] = val
    return df


def attach_industry(df: pd.DataFrame, fin: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Map code -> first announced industry, then build industry aggregates per date."""
    f = fin[["code", "period", "announce_date", "industry"]].copy()
    f["code"] = f["code"].astype(str).str.zfill(6)
    f["announce_date"] = pd.to_datetime(f["announce_date"], errors="coerce")
    f = f.dropna(subset=["code", "announce_date"])
    f = f.sort_values(["code", "announce_date"])
    first = f.groupby("code", as_index=False).first()[["code", "industry"]]
    first["industry"] = first["industry"].fillna("NA").astype(str)

    mapping = {k: i for i, k in enumerate(sorted(first["industry"].unique()))}
    first["ind_code"] = first["industry"].map(mapping).astype(np.int16)

    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.merge(first[["code", "ind_code"]], on="code", how="left")

    g = df.groupby(["date", "ind_code"], observed=True)
    agg = g.agg(ind_ret_1=("ret_1", "mean"),
                ind_ret_5=("ret_5", "mean"),
                ind_ret_20=("ret_20", "mean"),
                ind_n=("ret_1", "size")).reset_index()
    agg["ind_rank_5"] = agg.groupby("date")["ind_ret_5"].rank(pct=True).astype(np.float32)
    agg["ind_strength"] = (agg["ind_ret_5"] - agg.groupby("date")["ind_ret_5"]
                           .transform("mean")).astype(np.float32)

    df = df.merge(agg.drop(columns=["ind_n"]), on=["date", "ind_code"], how="left")
    with np.errstate(invalid="ignore"):
        df["ind_rel_5"] = (df["ret_5"] - df["ind_ret_5"]).astype(np.float32)
        df["ind_rel_20"] = (df["ret_20"] - df["ind_ret_20"]).astype(np.float32)
    for c in IND:
        df[c] = df[c].astype(np.float32)
    return df, mapping


def add_market(df: pd.DataFrame) -> pd.DataFrame:
    """Market regime: one value per date, broadcast to all stocks."""
    d = df.groupby("date").agg(
        mkt_ret_1=("ret_1", "median"),
        mkt_disp=("ret_1", "std"),
        mkt_close_pos=("close_pos", "median"),
        mkt_amt=("amount", "sum"),
        mkt_ret_5=("ret_5", "median"),
        mkt_ret_20=("ret_20", "median"),
        nh=("dist_high_20", lambda s: float((s >= -0.002).mean())),
        lu=("ret_1", lambda s: float((s >= 0.095).mean())),
        ld=("ret_1", lambda s: float((s <= -0.095).mean())),
    ).sort_index()

    d["mkt_nh_ratio"] = d["nh"].astype(np.float32)
    d["lu_ratio"] = d["lu"].astype(np.float32)
    d["ld_ratio"] = d["ld"].astype(np.float32)
    d["mkt_vol_20"] = d["mkt_ret_1"].rolling(20, min_periods=10).std().astype(np.float32)
    d["mkt_amt_z"] = (d["mkt_amt"] / d["mkt_amt"].rolling(20, min_periods=10).mean()
                      ).astype(np.float32)

    d = d.reset_index()
    keep = ["date"] + MKT
    df = df.merge(d[keep], on="date", how="left")
    for c in MKT:
        df[c] = df[c].astype(np.float32)
    return df


def build_v3(df: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    df = add_shape(df)
    df, _ = attach_industry(df, fin)
    df = add_market(df)
    df = df.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)
    return df
