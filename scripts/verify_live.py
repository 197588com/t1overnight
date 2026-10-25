# -*- coding: utf-8 -*-
"""Proof that the live path reproduces the training path.

Rebuilds one day's features from a *quote snapshot* plus local history and diffs
every column against the panel that training used. If the live score can be
trusted at 14:36, this is the test that has to pass: same numbers, same order.

Run on a closed day (the snapshot then carries that day's final values) so the
comparison is exact rather than approximately exact.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1overnight import live  # noqa: E402
from t1overnight.features import FEATURES  # noqa: E402
from t1overnight.features_v3 import IND, MKT, SHAPE  # noqa: E402

SRC = "/Users/yanglin/Desktop/未命名文件夹/datapool/parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-10-23")
    ap.add_argument("--dataset", default="outputs/t1_dataset.parquet")
    ap.add_argument("--extra", default="outputs/t1_extra_v3.parquet")
    args = ap.parse_args()

    day = pd.Timestamp(args.date)
    bars = live.load_bars(f"{SRC}/daily_k.parquet", args.date, 90)
    codes = sorted(bars["code"].unique().tolist())
    snap = live.bulk_snapshot(codes)
    snap = live.attach_shares(snap, bars)
    print(f"snapshot {len(snap):,} / {len(codes):,} coded names")

    fin = live.load_fin_first_industry(f"{SRC}/fin_quarterly.parquet")
    carry = pd.read_parquet(args.dataset,
                            columns=["code", "date"] + live.CARRY)
    carry["date"] = pd.to_datetime(carry["date"])
    carry = carry[carry["date"] < day].sort_values("date").groupby("code").last().reset_index()

    feat = live.build_live_features(bars, snap, day, fin, carry=carry)
    feat.to_parquet(f"/tmp/live_feat_{args.date}.parquet", index=False)
    print(f"rebuilt {feat.shape[0]:,} rows x {feat.shape[1]} cols")

    gap = feat["adj_gap"].astype(float)
    print(f"adjustment factor: median={gap.median():.6f} p99|dev|={gap.sub(1).abs().quantile(.99):.2e} "
          f"outside 1%: {int((gap.sub(1).abs() > 0.01).sum())}")

    cols = [c for c in FEATURES + SHAPE + IND + MKT if c != "board"]
    ref = pd.read_parquet(args.dataset, columns=["code", "date"] + [c for c in FEATURES if c != "board"])
    ref["date"] = pd.to_datetime(ref["date"])
    ref = ref[ref["date"] == day]
    ex = pd.read_parquet(args.extra, columns=["code", "date"] + SHAPE + IND + MKT)
    ex["date"] = pd.to_datetime(ex["date"])
    ref = ref.merge(ex[ex["date"] == day], on=["code", "date"], how="left")

    m = feat.merge(ref, on="code", suffixes=("", "_ref"))
    print(f"matched {len(m):,} / panel {len(ref):,}")

    rows = []
    for c in cols:
        a, b = m[c].to_numpy(float), m[f"{c}_ref"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 500:
            rows.append({"feature": c, "max_abs": np.nan, "med_abs": np.nan, "n": int(ok.sum())})
            continue
        d = np.abs(a[ok] - b[ok])
        rows.append({"feature": c, "max_abs": float(d.max()),
                     "med_abs": float(np.median(d)), "n": int(ok.sum())})
    r = pd.DataFrame(rows).sort_values("med_abs", ascending=False)

    print("\nworst 12 by MEDIAN deviation (what actually moves the score):")
    print(r.head(12).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    exact = r[r.med_abs < 1e-6]
    print(f"\nidentical (median dev < 1e-6): {len(exact)}/{len(r)} features")
    out = ROOT / "outputs/live_verify"
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / f"feature_diff_{args.date}.csv", index=False)
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
