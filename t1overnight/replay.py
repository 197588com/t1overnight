# -*- coding: utf-8 -*-
"""Replay the T+1 model at 14:36 instead of 15:00.

The training features are built from the *completed* daily bar. To score a stock
at 14:36 you only know the tape up to 14:36, so today's bar has to be truncated:

    close -> price @14:36           high  -> high so far
    low   -> low so far             volume/amount -> cumulative so far

Every downstream feature (ret_k, ma_dev, close_pos, amt_ratio, turn_ratio, ...)
is then recomputed from that truncated bar, exactly as it would be known live.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURES, add_cross_sectional, compute_base

BAR_COLS = ["code", "date", "open", "high", "low", "close", "volume",
            "amount", "outstanding_share", "turnover"]

# exactly what train_v2.py fed the booster (features.FEATURES minus the mkt_* block)
MODEL_FEATURES = [c for c in FEATURES
                  if c not in ("mkt_ret", "mkt_up_ratio", "mkt_amt_ratio")]

# intraday contract, one row per (code, date) at the cut-off
INTRA_COLS = ["code", "p_cut", "hi_cut", "lo_cut", "vol_cut", "amt_cut",
              "close_raw", "dayvol"]


def load_history(daily: pd.DataFrame, date: str, bars: int = 90) -> pd.DataFrame:
    """Last `bars` adjusted daily bars up to and including `date`, per code."""
    d = daily[daily["date"] <= pd.Timestamp(date)]
    d = d.sort_values(["code", "date"])
    return d.groupby("code", sort=False).tail(bars).reset_index(drop=True)


def truncate_bar(hist: pd.DataFrame, intr: pd.DataFrame) -> pd.DataFrame:
    """Replace today's bar with what is observable at the cut-off time."""
    h = hist.copy()
    h["date"] = pd.to_datetime(h["date"])
    last_idx = h.groupby("code", sort=False).tail(1).index

    m = h.loc[last_idx, ["code", "close", "turnover"]].merge(
        intr[INTRA_COLS], on="code", how="left")
    m.index = last_idx

    fac = (m["close"] / m["close_raw"]).to_numpy(np.float64)  # adj / raw
    h.loc[last_idx, "close"] = m["p_cut"].to_numpy() * fac
    h.loc[last_idx, "high"] = m["hi_cut"].to_numpy() * fac
    h.loc[last_idx, "low"] = m["lo_cut"].to_numpy() * fac
    h.loc[last_idx, "volume"] = m["vol_cut"].to_numpy()
    h.loc[last_idx, "amount"] = m["amt_cut"].to_numpy()
    # turnover = volume / float shares -> scale by the share of the day already traded
    h.loc[last_idx, "turnover"] = (
        m["turnover"].to_numpy() * (m["vol_cut"] / m["dayvol"]).to_numpy())
    return h


def rebuild(daily: pd.DataFrame, intr: pd.DataFrame, date: str,
            bars: int = 90) -> pd.DataFrame:
    """Feature rows for `date`, computed with the truncated bar."""
    hist = truncate_bar(load_history(daily, date, bars), intr)
    hist["date"] = pd.to_datetime(hist["date"])

    frames = []
    for _, g in hist.groupby("code", sort=False):
        if len(g) < 40:
            continue
        frames.append(compute_base(g).tail(1))
    out = pd.concat(frames, ignore_index=True)
    out["code"] = out["code"].astype(str).str.zfill(6)
    return out.sort_values("code").reset_index(drop=True)


def score_with(booster, df: pd.DataFrame) -> np.ndarray:
    x = df[MODEL_FEATURES].copy()
    x["board"] = x["board"].astype("int16")
    return booster.predict(x).astype(np.float32)
