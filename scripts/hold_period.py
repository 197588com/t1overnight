# -*- coding: utf-8 -*-
"""Holding-period study: same ranking signal, different exit horizons.

For k = 1,2,3,5 trading days: enter every k-th session on the top-ranked name,
exit after k sessions, pay one round trip of costs per trade.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t1overnight.backtest import CostConfig, SelectConfig, performance, tradable  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="outputs/walkforward_v2/predictions.parquet")
    ap.add_argument("--dataset", default="outputs/t1_dataset.parquet")
    ap.add_argument("--out-dir", default="outputs/hold")
    ap.add_argument("--capital", type=float, default=100_000.0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cost = CostConfig().round_trip

    close = pd.read_parquet(args.dataset, columns=["code", "date", "close", "amount",
                                                   "volume", "band", "st_proxy", "ret_1",
                                                   "ret_20", "ma_dev_20", "amt_ratio"])
    close["date"] = pd.to_datetime(close["date"])
    close = close.sort_values(["code", "date"], kind="mergesort")
    for k in [1, 2, 3, 5]:
        close[f"c{k}"] = close.groupby("code", sort=False)["close"].shift(-k)

    pred = pd.read_parquet(args.pred)
    pred["date"] = pd.to_datetime(pred["date"])
    pred["code"] = pred["code"].astype(str).str.zfill(6)
    pred = pred.merge(close[["code", "date", "c1", "c2", "c3", "c5"]], on=["code", "date"], how="left")

    cfg = SelectConfig(top_n=1)
    # tradability needs the extra columns
    pred = pred.merge(close[["code", "date", "amount", "volume", "band", "st_proxy", "ret_1",
                             "ret_20", "ma_dev_20", "amt_ratio"]],
                      on=["code", "date"], how="left", suffixes=("", "_x"))
    for c in ["amount", "volume", "band", "st_proxy", "ret_1", "ret_20", "ma_dev_20", "amt_ratio"]:
        if f"{c}_x" in pred.columns:
            pred[c] = pred[f"{c}_x"]
    pred = pred.drop(columns=[c for c in pred.columns if c.endswith("_x")])

    uni = pred[tradable(pred, cfg)].copy()
    print(f"tradable rows {len(uni):,} over {uni.date.nunique()} dates")

    rows, curves = [], {}
    dates = np.sort(uni["date"].unique())
    for k in [1, 2, 3, 5, 10]:
        col = f"c{k}" if k <= 5 else None
        if col is None:
            continue
        trades = []
        for d in dates:                     # every date, overlapping (stable averages)
            g = uni[uni["date"] == d]
            if g.empty:
                continue
            top = g.nlargest(1, "score").iloc[0]
            v = top.get(col, np.nan)
            if not np.isfinite(v):
                continue
            trades.append({"date": d, "code": top["code"], "score": top["score"],
                           "gross": v / top["close"] - 1})
        tr = pd.DataFrame(trades)
        if tr.empty:
            continue
        tr["net"] = tr["gross"] - cost
        n = len(tr)
        mean_net = float(tr["net"].mean())
        ann_simple = (1 + mean_net) ** (244 / k) - 1
        # non-overlapping equity for a realistic compounded path
        tr_ns = tr.iloc[::k].copy()
        eq_ns = float((1 + tr_ns["net"]).cumprod().iloc[-1])
        ann_ns = eq_ns ** (244 / (len(tr_ns) * k)) - 1
        rows.append({"k": k, "trades": n, "hit": float((tr["gross"] > 0).mean()),
                     "gross_mean": float(tr["gross"].mean()), "net_mean": mean_net,
                     "t_stat": float(mean_net / (tr["net"].std() / np.sqrt(n))),
                     "ann_simple": float(ann_simple), "ann_compound": float(ann_ns),
                     "cost_per_year": float(cost * 244 / k)})
        curves[k] = (tr_ns.assign(equity=args.capital * (1 + tr_ns["net"]).cumprod())
                     .assign(date=tr_ns["date"].dt.strftime("%Y-%m-%d"))
                     [["date", "equity"]].round(0).to_dict(orient="list"))
        tr.to_csv(out / f"trades_k{k}.csv", index=False)

    hp = pd.DataFrame(rows)
    hp.to_csv(out / "hold_period.csv", index=False)
    print("\n=== holding-period comparison (same 1-day ranking signal, all entry dates) ===")
    print(hp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # how does gross alpha scale with horizon?
    print("\ninterpretation: gross_mean = average k-day return of the top-ranked name,")
    print("net_mean = after one round trip; ann_simple = (1+net_mean)^(244/k)-1 (no vol drag);")
    print("ann_compound = non-overlapping equity path, includes vol drag.")

    with open(out / "hold_period.json", "w", encoding="utf-8") as f:
        json.dump({"table": rows, "curves": {str(k): v for k, v in curves.items()},
                   "cost_rt": cost}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
