# -*- coding: utf-8 -*-
"""Replay one trading day as it looked at 14:36 and compare with the 15:00 score.

Only the pool that could plausibly occupy the top of the ranking is pulled
(top-N by the closing score + a random control), so the API cost stays bounded.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from probe_1436 import day_metrics, ts_code          # reuse the puller

ROOT = Path(__file__).resolve().parents[1]

CS_COLS = ["cs_ret_1", "cs_ret_5", "cs_amt", "cs_turn", "cs_ma_dev_20"]
FIN_COLS = ["roe", "net_profit_yoy", "revenue_yoy", "days_listed"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--fold", default=None, help="YYYYMM of the model to use")
    ap.add_argument("--top", type=int, default=600, help="pool by closing score")
    ap.add_argument("--random", type=int, default=200)
    ap.add_argument("--full", action="store_true",
                    help="pull the whole tradable universe (exact cross-section)")
    ap.add_argument("--out", default="outputs/replay1436")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(ROOT))
    from t1overnight.backtest import SelectConfig, tradable
    from t1overnight.features import FEATURES, add_cross_sectional
    from t1overnight.replay import rebuild, score_with

    day = pd.Timestamp(args.date)
    pred = pd.read_parquet(ROOT / "outputs/walkforward_v2/predictions.parquet")
    pred["date"] = pd.to_datetime(pred["date"])
    pred["code"] = pred["code"].astype(str).str.zfill(6)
    ds = pd.read_parquet(ROOT / "outputs/t1_dataset.parquet")
    ds["date"] = pd.to_datetime(ds["date"])
    ds["code"] = ds["code"].astype(str).str.zfill(6)

    ex = ds.loc[ds.date == day, ["code", "amt_ratio", "ret_20", "ma_dev_20"]]
    p = pred[pred.date == day].merge(ex, on="code", how="left")
    cfg = SelectConfig(top_n=1)
    p = p[tradable(p, cfg)].copy().sort_values("score", ascending=False)
    if p.empty:
        raise SystemExit(f"no tradable rows on {args.date}")

    rng = np.random.default_rng(11)
    if args.full:
        pool = p.copy()
    else:
        pool = pd.concat([p.head(args.top),
                          p.iloc[rng.choice(len(p), min(args.random, len(p)), replace=False)]])
        pool = pool.drop_duplicates("code")
    pool = pool.reset_index(drop=True)
    print(f"[pool] {len(pool)} names (tradable universe {len(p)})", flush=True)

    # ---- pull the tape up to the cut-off -----------------------------------
    cache = out / f"intra_{args.date}{'' if args.full else '_pool'}.parquet"
    if cache.exists():
        intr = pd.read_parquet(cache)
    else:
        rows = []
        t0 = time.time()
        for i, r in enumerate(pool.itertuples(), 1):
            m = day_metrics(r.code, args.date)
            if m is None:
                continue
            rows.append({"code": r.code, "close_raw": m["close_m"],
                         "p_cut": m["p1436"], "hi_cut": m["hi36"],
                         "lo_cut": m["lo36"], "vol_cut": m["cumvol"],
                         "amt_cut": m["cumamt"], "dayvol": m["dayvol"]})
            if i % 100 == 0:
                print(f"  {i}/{len(pool)} {time.time()-t0:.0f}s", flush=True)
        intr = pd.DataFrame(rows)
        intr.to_parquet(cache, index=False)
    print(f"[tape] {len(intr)} names with minute data", flush=True)

    # ---- rebuild truncated features ---------------------------------------
    daily = pd.read_parquet("/Users/yanglin/Desktop/未命名文件夹/datapool/parquet/daily_k.parquet",
                            columns=["code", "date", "open", "high", "low", "close",
                                     "volume", "amount", "outstanding_share", "turnover"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["date"] = pd.to_datetime(daily["date"])
    hist = daily[(daily.date <= day) & (daily.code.isin(set(intr.code)))]
    rec = rebuild(hist, intr, args.date)
    rec = rec.merge(intr[["code", "close_raw"]], on="code", how="left")

    ref = ds.loc[ds.date == day, ["code"] + CS_COLS + FIN_COLS]
    if args.full:
        # exact cross-section: recompute the percentile features from the truncated panel
        rec = add_cross_sectional(rec)
        ref = ref[["code"] + FIN_COLS]
    rec = rec.merge(ref, on="code", how="left", suffixes=("", "_ref"))
    for c in FIN_COLS:                       # listing age / fundamentals are carried, not rebuilt
        if f"{c}_ref" in rec.columns:
            rec[c] = rec[f"{c}_ref"].astype(float)
            rec = rec.drop(columns=[f"{c}_ref"])
    print(f"[feat] rebuilt {len(rec)} rows", flush=True)

    # ---- score both ways ---------------------------------------------------
    fold = args.fold or day.strftime("%Y%m")
    booster = lgb.Booster(model_file=str(ROOT / "outputs/walkforward_v2/models"
                                         / f"rank_{fold}.txt"))
    rec["score_1436"] = score_with(booster, rec)

    cmp = rec.merge(pool[["code", "score"]].rename(columns={"score": "score_1500"}),
                    on="code", how="inner")
    cmp["rank_1500"] = cmp["score_1500"].rank(ascending=False, method="first")
    cmp["rank_1436"] = cmp["score_1436"].rank(ascending=False, method="first")

    rho = cmp[["score_1436", "score_1500"]].corr(method="spearman").iloc[0, 1]
    print(f"\n=== {args.date}  fold {fold} ===")
    print(f"pool={len(cmp)}  spearman(14:36 score, 15:00 score) = {rho:.4f}")
    for k in [1, 5, 10, 20, 50]:
        a = set(cmp.nsmallest(k, "rank_1500").code)
        b = set(cmp.nsmallest(k, "rank_1436").code)
        print(f"  top{k:>3} 重合 {len(a & b)}/{k}  ({len(a & b)/k:.0%})")

    top1 = cmp.nsmallest(1, "rank_1500").iloc[0]
    row = cmp[cmp.code == top1.code].iloc[0]
    print(f"\n收盘口径 top1 = {top1.code}，14:36 重打后排名 {int(row.rank_1436)}/{len(cmp)}")
    print(cmp.nsmallest(8, "rank_1500")[["code", "rank_1500", "rank_1436",
                                         "score_1500", "score_1436", "close_pos",
                                         "turn", "ret_1"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n14:36 口径的 top5:")
    print(cmp.nsmallest(5, "rank_1436")[["code", "rank_1500", "rank_1436",
                                         "score_1500", "score_1436"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    cmp.to_csv(out / f"cmp_{args.date}.csv", index=False)
    print(f"\nsaved -> {out/f'cmp_{args.date}.csv'}")


if __name__ == "__main__":
    main()
