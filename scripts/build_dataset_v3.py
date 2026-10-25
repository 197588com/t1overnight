# -*- coding: utf-8 -*-
"""Build the v3 feature block on top of the existing v2 dataset.

Output: outputs/t1_extra_v3.parquet  (code, date, ind_code, + 21 v3 features)
Keeping it separate keeps peak RAM low and lets training pick feature groups.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1overnight.features_v3 import V3, build_v3  # noqa: E402

DT = {"code": "string", "open": "float32", "high": "float32", "low": "float32",
      "close": "float32", "volume": "float32", "amount": "float32",
      "turn": "float32", "ret_1": "float32", "ret_5": "float32",
      "ret_20": "float32", "close_pos": "float32", "dist_high_20": "float32"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="outputs/t1_dataset.parquet")
    ap.add_argument("--fin", default="/Users/yanglin/Desktop/未命名文件夹/datapool/parquet/fin_quarterly.parquet")
    ap.add_argument("--out", default="outputs/t1_extra_v3.parquet")
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_parquet(args.dataset, columns=["date"] + list(DT))
    df["date"] = pd.to_datetime(df["date"])
    for k, v in DT.items():
        if k not in ("code",) and df[k].dtype != v:
            df[k] = df[k].astype(v)
    print(f"[load] {len(df):,} rows {time.time()-t0:.1f}s", flush=True)

    fin = pd.read_parquet(args.fin, columns=["code", "period", "announce_date", "industry"])
    out = build_v3(df, fin)
    print(f"[build] {out.shape} {time.time()-t0:.1f}s", flush=True)

    keep = ["code", "date", "ind_code"] + V3
    out = out[keep]
    for c in V3:
        out[c] = out[c].astype("float32")
    out["ind_code"] = out["ind_code"].fillna(-1).astype("int16")

    print("\nNaN rate:")
    print(out[V3].isna().mean().round(3).to_string())
    print(f"\nindustry coverage: {(out.ind_code >= 0).mean():.3%}, "
          f"n_industries={out.ind_code.max()+1}")

    out.to_parquet(args.out, index=False, compression="zstd")
    print(f"\n[done] -> {args.out}  ({out.memory_usage(deep=True).sum()/1e9:.2f} GB in RAM) "
          f"{time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
