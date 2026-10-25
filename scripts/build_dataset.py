# -*- coding: utf-8 -*-
"""Build the T+1 overnight dataset from the A-share datapool.

Usage:
    python scripts/build_dataset.py --source /path/to/parquet --out outputs/t1_dataset.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t1overnight.features import add_cross_sectional, compute_base, merge_financials  # noqa: E402

KEEP_COLS = ["code", "date", "open", "high", "low", "close", "volume",
             "amount", "outstanding_share", "turnover"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="parquet directory of the datapool")
    ap.add_argument("--out", default="outputs/t1_dataset.parquet")
    ap.add_argument("--start", default="2016-01-01")
    args = ap.parse_args()

    src = Path(args.source)
    t0 = time.time()
    daily = pd.read_parquet(src / "daily_k.parquet", columns=KEEP_COLS)
    print(f"[load] daily_k {daily.shape} in {time.time()-t0:.1f}s", flush=True)

    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily[daily["date"] >= pd.Timestamp(args.start)]

    t1 = time.time()
    df = compute_base(daily)
    print(f"[features] {df.shape} in {time.time()-t1:.1f}s", flush=True)

    t2 = time.time()
    df = add_cross_sectional(df)
    print(f"[cross-section] done in {time.time()-t2:.1f}s", flush=True)

    fin = pd.read_parquet(src / "fin_quarterly.parquet",
                          columns=["code", "announce_date", "roe", "net_profit_yoy", "revenue_yoy"])
    df = merge_financials(df, fin)
    print(f"[financials] merged, cols={len(df.columns)}", flush=True)

    # execution flag: is the stock's next bar the market's next trading day?
    cal = np.sort(df["date"].unique())
    next_map = dict(zip(cal[:-1], cal[1:]))
    df = df.sort_values(["code", "date"], kind="mergesort")
    df["next_bar_date"] = df.groupby("code", sort=False)["date"].shift(-1)
    df["exec_ok"] = (df["next_bar_date"] == df["date"].map(next_map)).astype(np.int8)
    df["next_gap_days"] = (df["next_bar_date"] - df["date"]).dt.days.astype(np.float32)
    df = df.drop(columns=["next_bar_date"])
    print(f"[exec] suspended-next-day rows: {(df['exec_ok'] == 0).mean():.3%}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression="zstd")
    print(f"[save] {out} rows={len(df):,} size={out.stat().st_size/1e6:.0f}MB "
          f"total={time.time()-t0:.1f}s", flush=True)

    print("\nlabel coverage:")
    for c in ["fwd_cc", "fwd_co"]:
        print(f"  {c}: {df[c].notna().mean():.3%} non-null, mean={df[c].mean():+.5f}, "
              f"hit={(df[c] > 0).mean():.4f}")
    print(f"\ndate range: {df.date.min().date()} -> {df.date.max().date()}")
    print(f"rows: {len(df):,}")


if __name__ == "__main__":
    main()
