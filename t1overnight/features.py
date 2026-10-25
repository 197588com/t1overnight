# -*- coding: utf-8 -*-
"""T+1 overnight feature engineering (numpy, per-code loop).

Design rules
------------
1. Every feature at date D uses only bars <= D (no look-ahead).
2. Labels use bars D+1 (entry = close of D, exit = close of D+1).
3. Memory is controlled by computing per-code on contiguous numpy slices and
   writing into pre-allocated float32 arrays.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def board_code(c: np.ndarray) -> np.ndarray:
    """0=SH main, 1=SZ main, 2=GEM(300), 3=STAR(688), 4=BJ."""
    s = pd.Series(c).astype(str).str.zfill(6)
    out = np.full(len(c), 1, dtype=np.int8)
    out[s.str.startswith(("60", "90")).values] = 0
    out[s.str.startswith(("30",)).values] = 2
    out[s.str.startswith(("68",)).values] = 3
    out[s.str.startswith(("92", "43", "83", "87")).values] = 4
    return out


def band_of_board(b: np.ndarray) -> np.ndarray:
    """Daily price-limit band by board (ST not observable point-in-time)."""
    band = np.full(len(b), 0.10, dtype=np.float32)
    band[b == 2] = 0.20
    band[b == 3] = 0.20
    band[b == 4] = 0.30
    return band


def _roll_mean(x: np.ndarray, w: int, minp: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=minp).mean().to_numpy(dtype=np.float32)


def _roll_std(x: np.ndarray, w: int, minp: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=minp).std().to_numpy(dtype=np.float32)


def _roll_max(x: np.ndarray, w: int, minp: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=minp).max().to_numpy(dtype=np.float32)


def _roll_min(x: np.ndarray, w: int, minp: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=minp).min().to_numpy(dtype=np.float32)


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """Lag (k>0) or lead (k<0) by k observations, padding with NaN."""
    out = np.full(len(x), np.nan, dtype=np.float32)
    if k == 0:
        return x.astype(np.float32)
    if k > 0:
        if k < len(x):
            out[k:] = x[:-k]
    else:
        m = -k
        if m < len(x):
            out[:-m] = x[m:]
    return out


FEATURES = [
    # short-term price action
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
    "gap", "intraday", "close_pos", "amp",
    # risk
    "vol_5", "vol_20", "vol_ratio",
    # trend position
    "ma_dev_5", "ma_dev_20", "ma_dev_60", "dist_high_20", "dist_low_20", "dist_high_5",
    # liquidity
    "log_amt", "amt_ratio", "amt_trend", "turn", "turn_ratio", "log_mktcap",
    # limit-up pressure (point-in-time band proxy)
    "lu_cnt_5", "lu_cnt_20",
    # microstructure extras
    "z_ret", "streak_up", "days_listed",
    # quarterly fundamentals (point-in-time by announce_date)
    "roe", "net_profit_yoy", "revenue_yoy",
    # cross-sectional ranks within the day
    "cs_ret_1", "cs_ret_5", "cs_amt", "cs_turn", "cs_ma_dev_20",
    # market context
    "mkt_ret", "mkt_up_ratio", "mkt_amt_ratio",
    "board",
]

LABELS = ["fwd_cc", "fwd_co", "fwd_oc", "fwd_high", "fwd_low"]


def compute_base(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute per-code features + labels. `daily` is sorted by code, date."""
    df = daily.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    n_all = len(df)

    code = df["code"].to_numpy()
    date = df["date"].to_numpy()

    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    lo = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    v = df["volume"].to_numpy(dtype=np.float64)
    amt = df["amount"].to_numpy(dtype=np.float64) if "amount" in df else np.full(n_all, np.nan)
    so = df["outstanding_share"].to_numpy(dtype=np.float64) if "outstanding_share" in df else np.full(n_all, np.nan)
    turn = df["turnover"].to_numpy(dtype=np.float64) if "turnover" in df else np.full(n_all, np.nan)

    bcode = board_code(code)
    band = band_of_board(bcode)

    # output containers
    out: dict[str, np.ndarray] = {k: np.full(n_all, np.nan, dtype=np.float32) for k in FEATURES if k != "board"}
    out["st_proxy"] = np.full(n_all, np.nan, dtype=np.float32)
    for k in LABELS:
        out[k] = np.full(n_all, np.nan, dtype=np.float32)

    # group boundaries
    uniq, starts = np.unique(code, return_index=True)
    order = np.argsort(starts, kind="mergesort")
    starts = starts[order]
    ends = np.append(starts[1:], n_all)

    for s, e in zip(starts, ends):
        if e - s < 30:
            continue
        co, hh, ll, cc, vv = o[s:e], h[s:e], lo[s:e], c[s:e], v[s:e]
        aa, ss, tt, bb = amt[s:e], so[s:e], turn[s:e], band[s:e]
        n = e - s

        prev_c = _shift(cc.astype(np.float32), 1).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret1 = cc / prev_c - 1
        ret1 = np.where(np.isfinite(ret1), ret1, np.nan)

        for k in [1, 3, 5, 10, 20, 60]:
            with np.errstate(divide="ignore", invalid="ignore"):
                r = cc / _shift(cc.astype(np.float32), k).astype(np.float64) - 1
            out[f"ret_{k}"][s:e] = r

        with np.errstate(divide="ignore", invalid="ignore"):
            out["gap"][s:e] = co / prev_c - 1
            out["intraday"][s:e] = cc / co - 1
            rng = hh - ll
            out["close_pos"][s:e] = np.where(rng > EPS, (cc - ll) / rng, 0.5)
            out["amp"][s:e] = rng / prev_c
            out["log_mktcap"][s:e] = np.log(np.maximum(cc * ss, 1.0))
            out["log_amt"][s:e] = np.log1p(np.maximum(aa, 0))

        r1 = ret1.astype(np.float32)
        out["ret_1"][s:e] = r1
        vol5 = _roll_std(r1, 5, 4)
        vol20 = _roll_std(r1, 20, 15)
        out["vol_5"][s:e] = vol5
        out["vol_20"][s:e] = vol20
        with np.errstate(divide="ignore", invalid="ignore"):
            out["vol_ratio"][s:e] = vol5 / vol20

        for k, minp in [(5, 4), (20, 15), (60, 40)]:
            ma = _roll_mean(cc.astype(np.float32), k, minp)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f"ma_dev_{k}"][s:e] = cc / ma - 1

        rh20 = _roll_max(hh.astype(np.float32), 20, 15)
        rl20 = _roll_min(ll.astype(np.float32), 20, 15)
        rh5 = _roll_max(hh.astype(np.float32), 5, 4)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["dist_high_20"][s:e] = cc / rh20 - 1
            out["dist_low_20"][s:e] = cc / rl20 - 1
            out["dist_high_5"][s:e] = cc / rh5 - 1

        a5 = _roll_mean(aa.astype(np.float32), 5, 3)
        a20 = _roll_mean(aa.astype(np.float32), 20, 15)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["amt_ratio"][s:e] = aa / a20
            out["amt_trend"][s:e] = a5 / a20

        out["turn"][s:e] = tt
        t5 = _roll_mean(tt.astype(np.float32), 5, 3)
        t20 = _roll_mean(tt.astype(np.float32), 20, 15)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["turn_ratio"][s:e] = tt / t20

        # limit-up counts with board band (ST band handled by proxy filter later)
        lu = (ret1 >= (bb - 0.005) - 1e-9).astype(np.float32)
        out["lu_cnt_5"][s:e] = _roll_mean(lu, 5, 1) * 5
        out["lu_cnt_20"][s:e] = _roll_mean(lu, 20, 1) * 20

        with np.errstate(divide="ignore", invalid="ignore"):
            out["z_ret"][s:e] = r1 / vol20
        up = (r1 > 0).astype(np.float32)
        streak = np.zeros(n, dtype=np.float32)
        run = 0.0
        for i in range(n):
            run = run + 1.0 if up[i] == 1.0 else 0.0
            streak[i] = run
        out["streak_up"][s:e] = streak
        out["days_listed"][s:e] = np.arange(n, dtype=np.float32)

        # point-in-time ST proxy: 5% limit regime detected from trailing 60 bars
        mx60 = _roll_max(r1, 60, 30)
        mn60 = _roll_min(r1, 60, 30)
        stp = ((mx60 <= 0.0525) & (mn60 >= -0.0525) &
               ((mx60 >= 0.047) | (mn60 <= -0.047))).astype(np.float32)
        out["st_proxy"][s:e] = stp

        # labels: entry at close of D
        with np.errstate(divide="ignore", invalid="ignore"):
            nxt_c = _shift(cc.astype(np.float32), -1).astype(np.float64)
            out["fwd_cc"][s:e] = nxt_c / cc - 1
            nxt_o = _shift(co.astype(np.float32), -1).astype(np.float64)
            out["fwd_co"][s:e] = nxt_o / cc - 1
            out["fwd_oc"][s:e] = nxt_c / nxt_o - 1
            out["fwd_high"][s:e] = _shift(hh.astype(np.float32), -1).astype(np.float64) / cc - 1
            out["fwd_low"][s:e] = _shift(ll.astype(np.float32), -1).astype(np.float64) / cc - 1

    out["board"] = bcode.astype(np.float32)

    res = pd.DataFrame({"code": code, "date": date})
    for k, val in out.items():
        res[k] = val
    # keep raw fields needed by the backtest
    res["close"] = c.astype(np.float32)
    res["open"] = o.astype(np.float32)
    res["high"] = h.astype(np.float32)
    res["low"] = lo.astype(np.float32)
    res["volume"] = v.astype(np.float32)
    res["amount"] = amt.astype(np.float32)
    res["band"] = band
    return res


def add_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional percentile ranks + market context."""
    for col, name in [("ret_1", "cs_ret_1"), ("ret_5", "cs_ret_5"),
                      ("amount", "cs_amt"), ("turn", "cs_turn"),
                      ("ma_dev_20", "cs_ma_dev_20")]:
        df[name] = df.groupby("date")[col].rank(pct=True).astype(np.float32)

    g = df.groupby("date")
    df["mkt_ret"] = g["ret_1"].transform("median").astype(np.float32)
    df["mkt_up_ratio"] = g["ret_1"].transform(lambda s: (s > 0).mean()).astype(np.float32)
    df["mkt_amt_ratio"] = g["amt_ratio"].transform("median").astype(np.float32)
    return df


def merge_financials(df: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    """Latest announced fundamentals as of date (merge_asof on announce_date)."""
    f = fin.copy()
    f["announce_date"] = pd.to_datetime(f["announce_date"], errors="coerce")
    f["code"] = f["code"].astype(str).str.zfill(6)
    keep = ["code", "announce_date", "roe", "net_profit_yoy", "revenue_yoy"]
    f = f[[c for c in keep if c in f.columns]].dropna(subset=["code", "announce_date"])
    f = f.sort_values(["announce_date", "code"], kind="mergesort")

    left = df.copy()
    left["code"] = left["code"].astype(str).str.zfill(6)
    left["date"] = pd.to_datetime(left["date"])
    left = left.sort_values(["date", "code"], kind="mergesort")

    out = pd.merge_asof(left, f, left_on="date", right_on="announce_date",
                        by="code", direction="backward",
                        suffixes=("", "_fin"))
    out = out.drop(columns=["announce_date"], errors="ignore")
    for c in ["roe", "net_profit_yoy", "revenue_yoy"]:
        if f"{c}_fin" in out.columns:
            out[c] = out[f"{c}_fin"].astype(np.float32)
            out = out.drop(columns=[f"{c}_fin"])
    return out.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)


