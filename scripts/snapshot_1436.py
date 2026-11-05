# -*- coding: utf-8 -*-
"""14:36 前向记账 — capture a real snapshot, rebuild features, score, and keep the tape.

Historical replay of this is impossible (5,000 names x 1,160 days of minute data),
so the only honest way to know how much gross survives at 14:36 is to record it
forward, every session, from today.

Two slots per session, both cheap (~20 s pull, ~45 s rebuild):
    14:36  what the model would see if you had to decide now
    15:01  the settled close, used as the exit price for the previous entry

Usage
-----
    python scripts/snapshot_1436.py --slot 1436
    python scripts/snapshot_1436.py --slot 1501
    python scripts/snapshot_1436.py --slot 1436 --asof 2026-10-23   # backfill a closed day
"""
from __future__ import annotations

import os
import argparse
import glob
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1overnight import live  # noqa: E402

SRC = Path(os.environ.get("T1_DATAPOOL", os.path.expanduser("~/datapool/parquet")))


def newest(model_dir: str, config: str) -> Path:
    files = sorted(glob.glob(str(Path(model_dir) / f"{config}_*.txt")))
    if not files:
        raise FileNotFoundError(f"no {config}_*.txt under {model_dir}")
    return Path(files[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", choices=["1436", "1501"], default="1436")
    ap.add_argument("--asof", default=None, help="treat this date as 'today' (backfill)")
    ap.add_argument("--bars", default=str(SRC / "daily_k.parquet"))
    ap.add_argument("--fin", default=str(SRC / "fin_quarterly.parquet"))
    ap.add_argument("--panel", default=str(ROOT / "outputs/t1_dataset.parquet"))
    ap.add_argument("--triage-model", default=str(ROOT / "outputs/walkforward_v3/models"))
    ap.add_argument("--triage-config", default="tri_B")
    ap.add_argument("--rank-model", default=str(ROOT / "outputs/walkforward_v2/models"))
    ap.add_argument("--rank-config", default="rank")
    ap.add_argument("--out", default=str(ROOT / "outputs/forward"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    day = pd.Timestamp(args.asof) if args.asof else pd.Timestamp.today().normalize()
    day_s = day.strftime("%Y-%m-%d")

    t0 = time.time()
    bars = live.load_bars(args.bars, day_s, 90)
    codes = sorted(bars["code"].unique().tolist())
    print(f"[{day_s} {args.slot}] {len(codes):,} coded names, bars through "
          f"{bars.date.max().date()}", flush=True)

    snap = live.bulk_snapshot(codes)
    snap = live.attach_shares(snap, bars)
    snap.to_parquet(out / f"snap_{day_s}_{args.slot}.parquet", index=False)
    print(f"  snapshot {len(snap):,} rows in {time.time()-t0:.1f}s "
          f"(missing {len(snap.attrs.get('missing', []))})", flush=True)

    fin = live.load_fin_first_industry(args.fin)
    carry = pd.read_parquet(args.panel, columns=["code", "date"] + live.CARRY)
    carry["date"] = pd.to_datetime(carry["date"])
    carry = (carry[carry["date"] < day].sort_values("date")
             .groupby("code").last().reset_index())

    feat = live.build_live_features(bars, snap, day, fin, carry=carry)
    print(f"  features {feat.shape} in {time.time()-t0:.1f}s", flush=True)

    tmodel = newest(args.triage_model, args.triage_config)
    feat["bad_prob"] = live.score(lgb.Booster(model_file=str(tmodel)), feat)
    print(f"  triage model {tmodel.name} -> bad_prob mean={feat.bad_prob.mean():.4f}")

    try:
        rmodel = newest(args.rank_model, args.rank_config)
        feat["rank_score"] = live.score(lgb.Booster(model_file=str(rmodel)), feat)
        print(f"  rank model {rmodel.name} -> rank_score mean={feat.rank_score.mean():.4f}")
    except Exception as e:  # noqa: BLE001
        feat["rank_score"] = np.nan
        print(f"  rank model skipped: {e}")

    keep = ["code", "name", "date", "price_now", "ret_today", "bad_prob", "rank_score",
            "ret_1", "ret_5", "turn", "amt_ratio", "amount", "adj_gap"]
    keep = [c for c in keep if c in feat.columns]
    feat[keep].to_parquet(out / f"feat_{day_s}_{args.slot}.parquet", index=False)
    print(f"[done] -> {out}/feat_{day_s}_{args.slot}.parquet  "
          f"({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
