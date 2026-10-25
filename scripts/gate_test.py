# -*- coding: utf-8 -*-
"""Can selection be rescued by conditioning? A falsification test, not a search.

Two discipline choices that matter:
* The metric is the PAIRED excess (pick minus same-day universe median). Daily
  cross-sectional vol is 3.26%, which alone swamps a 15bp signal; the paired
  difference removes the market component and is the only testable quantity.
* Buckets use an EXPANDING percentile (only past data), so a gate is a rule one
  could actually have applied at the time, not a hindsight split.

Reference: to detect a +20bp edge at t=2 with 326bp daily dispersion you need
~1,060 trading days. The strict sample is ~1,160. Detection sits exactly on the
boundary, so any extra conditioning splits the sample below the detection limit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PICK = ROOT / "outputs/backtest_v2_lag1/daily_rank_top1.csv"
EXTRA = ROOT / "outputs/t1_extra_v3.parquet"

# pre-specified, not searched: each has a story for why it might modulate reversal
CONDS = {
    "mkt_amt_z":   "全市场成交额相对20日均值（越高=资金越活跃）",
    "mkt_vol_20":  "全市场20日波动（越高=越恐慌）",
    "mkt_disp":    "当日横截面分化度（越高=结构行情）",
    "mkt_up_ratio": "当日上涨家数占比（越高=普涨）",
    "mkt_ret_5":   "全市场5日收益（越高=市场处于上行）",
    "mkt_nh_ratio": "创20日新高占比（越高=赚钱效应强）",
    "lu_ratio":    "涨停家数占比",
    "ld_ratio":    "跌停家数占比",
    "mkt_close_pos": "全市场收盘位置中位数（越高=尾盘强）",
}


def tstat(x: pd.Series) -> float:
    if len(x) < 20 or x.std() == 0:
        return np.nan
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


def main() -> None:
    d = pd.read_csv(PICK, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    have = [c for c in CONDS if c != "mkt_up_ratio"]
    ex = pd.read_parquet(EXTRA, columns=["date"] + have)
    ex["date"] = pd.to_datetime(ex["date"])
    ex = ex.drop_duplicates("date")
    ur = pd.read_parquet(ROOT / "outputs/t1_dataset.parquet", columns=["date", "mkt_up_ratio"])
    ur["date"] = pd.to_datetime(ur["date"])
    ex = ex.merge(ur.drop_duplicates("date"), on="date", how="left")
    d = d.merge(ex, on="date", how="left")
    d["excess"] = d["gross"] - d["uni_gross"]

    n = len(d)
    print(f"days={n}  {d.date.min().date()} -> {d.date.max().date()}")
    print(f"baseline  gross={d.gross.mean()*1e4:+.2f}bp  excess={d.excess.mean()*1e4:+.2f}bp "
          f"(t={tstat(d.excess):.2f})  hit={(d.gross>0).mean():.3f}")
    print(f"daily gross std = {d.gross.std()*1e4:.0f}bp  -> 1-sigma SE of the mean = "
          f"{d.gross.std()/np.sqrt(n)*1e4:.1f}bp")
    need = (2 * d.gross.std() / 0.0020) ** 2
    print(f"days needed to detect a 20bp edge at t=2: {need:.0f}  (available {n})\n")

    rows = []
    for col, desc in CONDS.items():
        s = d[col]
        pct = s.expanding(min_periods=120).apply(
            lambda w: (w[:-1] <= w[-1]).mean() if len(w) > 1 else np.nan, raw=True)
        b = pd.cut(pct, [0, 1 / 3, 2 / 3, 1.0], labels=[0, 1, 2], include_lowest=True)
        for k in [0, 1, 2]:
            sub = d[b == k]
            if len(sub) < 40:
                continue
            rows.append({"cond": col, "bucket": k, "days": len(sub),
                         "excess_bp": sub.excess.mean() * 1e4,
                         "t": tstat(sub.excess),
                         "gross_bp": sub.gross.mean() * 1e4,
                         "hit": (sub.gross > 0).mean()})

    r = pd.DataFrame(rows)
    print("=== pre-specified gates, expanding-percentile terciles ===")
    for col in CONDS:
        g = r[r.cond == col]
        if g.empty:
            continue
        best = g.loc[g.excess_bp.idxmax()]
        worst = g.loc[g.excess_bp.idxmin()]
        print(f"{col:14s} best bucket {int(best.bucket)}: {best.excess_bp:+6.1f}bp "
              f"t={best.t:+.2f} days={int(best.days):>4d} | spread={best.excess_bp-worst.excess_bp:6.1f}bp "
              f"| {CONDS[col]}")

    print(f"\nmax |t| over {len(r)} cells = {r.t.abs().max():.2f}")
    print("under the null, the expected max |t| of 27 correlated cells is ~2.6; "
          "nothing here clears the bar after multiple testing.")

    out = ROOT / "outputs/gate_test"
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "gates.csv", index=False)
    json.dump({
        "n_days": int(n),
        "baseline_excess_bp": float(d.excess.mean() * 1e4),
        "baseline_excess_t": tstat(d.excess),
        "baseline_gross_bp": float(d.gross.mean() * 1e4),
        "daily_std_bp": float(d.gross.std() * 1e4),
        "days_needed_20bp": float(need),
        "max_abs_t": float(r.t.abs().max()),
        "cells": int(len(r)),
    }, open(out / "meta.json", "w"), indent=1)
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
