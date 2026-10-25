# -*- coding: utf-8 -*-
"""Backtest the T+1 overnight model predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t1overnight.backtest import CostConfig, SelectConfig, performance, pick, tradable  # noqa: E402


def daily_series(sel: pd.DataFrame, uni: pd.DataFrame, cost: CostConfig) -> pd.DataFrame:
    """Per-date returns of the selected names vs the tradable universe."""
    g_sel = sel.groupby("date").agg(
        n=("fwd_cc", "size"),
        gross=("fwd_cc", "mean"),
        hit=("fwd_cc", lambda s: float((s > 0).mean())),
        gap=("fwd_co", "mean"),
        prob=("prob", "mean"),
        price=("close", "mean"),
        amount=("amount", "median"),
        st=("st_proxy", "mean"),
        bj=("board", lambda s: float((s == 4).mean())) if "board" in sel else ("bj", "size"),
    )
    g_uni = uni.groupby("date").agg(uni_gross=("fwd_cc", "mean"),
                                    uni_hit=("fwd_cc", lambda s: float((s > 0).mean())),
                                    uni_n=("fwd_cc", "size"))
    d = g_sel.join(g_uni, how="left")
    d["net"] = d["gross"] - cost.round_trip
    d["gap_net"] = d["gap"] - cost.round_trip
    d["excess_gross"] = d["gross"] - d["uni_gross"]
    d["excess_net"] = d["net"] - d["uni_gross"]
    return d.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="outputs/walkforward/predictions.parquet")
    ap.add_argument("--dataset", default="outputs/t1_dataset.parquet")
    ap.add_argument("--out-dir", default="outputs/backtest")
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--universe", default="/Users/yanglin/Desktop/未命名文件夹/datapool/parquet/universe.parquet")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pred = pd.read_parquet(args.pred)
    pred["date"] = pd.to_datetime(pred["date"])
    # board + a few ranking features are not carried through predictions
    extra = pd.read_parquet(args.dataset, columns=["code", "date", "board", "amt_ratio",
                                                   "ret_20", "ma_dev_20"])
    extra["date"] = pd.to_datetime(extra["date"])
    pred = pred.merge(extra, on=["code", "date"], how="left")
    if args.universe and Path(args.universe).exists():
        uni_names = pd.read_parquet(args.universe, columns=["code", "name"])
        uni_names["code"] = uni_names["code"].astype(str).str.zfill(6)
        pred["code"] = pred["code"].astype(str).str.zfill(6)
        pred = pred.merge(uni_names, on="code", how="left")

    cost = CostConfig()
    scfg = SelectConfig(top_n=1)
    print(f"round-trip cost = {cost.round_trip:.4%}")
    print(f"pred rows={len(pred):,}, dates={pred.date.nunique()}")

    uni_mask = tradable(pred, scfg)
    uni = pred.loc[uni_mask].copy()
    print(f"tradable universe rows={len(uni):,} ({uni_mask.mean():.2%} of pred rows)")

    variants: dict[str, pd.DataFrame] = {}

    for n in [1, 3, 5, 10]:
        c = SelectConfig(top_n=n)
        variants[f"model_top{n}"] = pick(pred, "prob", c)

    # calibrated probabilities, if present
    if "prob_cal" in pred.columns:
        variants["model_top1_cal"] = pick(pred, "prob_cal", SelectConfig(top_n=1))
        variants["model_top5_cal"] = pick(pred, "prob_cal", SelectConfig(top_n=5))

    # baselines on the same tradable universe
    uni = uni.assign(_rnd=np.random.default_rng(42).random(len(uni)))
    variants["baseline_random"] = uni.sort_values(["date", "_rnd"]).groupby("date").head(1)

    for name, col, asc in [("baseline_reversal", "ret_1", True),
                           ("baseline_momentum", "ret_1", False),
                           ("baseline_amount", "amount", False),
                           ("baseline_amt_ratio", "amt_ratio", False)]:
        d = uni.sort_values(["date", col], ascending=[True, asc])
        variants[name] = d.groupby("date").head(1)

    rows = []
    curves = {}
    detail = {}
    for name, sel in variants.items():
        if sel.empty:
            continue
        s = daily_series(sel, uni, cost)
        detail[name] = s
        rows.append(performance(s["net"], name))
        curves[name] = s[["date", "net"]].assign(equity=args.capital * (1 + s["net"]).cumprod())

    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary_variants.csv", index=False)
    print("\n=== variant summary (net of costs) ===")
    cols = ["variant", "days", "hit_rate", "mean_daily", "total_return", "annual_return",
            "sharpe", "max_drawdown", "t_stat"]
    print(summ[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    for name, s in detail.items():
        s.to_csv(out / f"daily_{name}.csv", index=False)

    # yearly / monthly breakdown for the headline variant
    top = detail.get("model_top1")
    if top is not None:
        top = top.copy()
        top["year"] = top["date"].dt.year
        top["month"] = top["date"].dt.to_period("M").astype(str)
        yr = top.groupby("year").agg(
            days=("net", "size"), hit=("hit", "mean"),
            gross=("gross", "mean"), net=("net", "mean"),
            uni=("uni_gross", "mean"), excess=("excess_gross", "mean"),
            cum=("net", lambda s: float((1 + s).prod() - 1)),
        ).reset_index()
        yr.to_csv(out / "yearly_top1.csv", index=False)
        print("\n=== model_top1 by year ===")
        print(yr.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        mo = top.groupby("month").agg(days=("net", "size"), hit=("hit", "mean"),
                                      net=("net", "mean"), cum=("net", lambda s: float((1 + s).prod() - 1)))
        mo.to_csv(out / "monthly_top1.csv")
        print(f"monthly win rate (cum>0): {(mo['cum'] > 0).mean():.2%} over {len(mo)} months")

    # gross vs net sensitivity
    sens = []
    for c_rt in [0.0, 0.0010, 0.0015, 0.0020, 0.0025]:
        s = detail["model_top1"].copy()
        r = s["gross"] - c_rt
        sens.append({"cost_round_trip": c_rt, **{k: v for k, v in performance(r, "top1").items()
                                                if k in ("hit_rate", "mean_daily", "annual_return", "t_stat")}})
    pd.DataFrame(sens).to_csv(out / "cost_sensitivity.csv", index=False)
    print("\n=== cost sensitivity (model_top1) ===")
    print(pd.DataFrame(sens).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # latest snapshot
    pred_latest = pred[pred["date"] == pred["date"].max()]
    uni_latest = pred_latest[tradable(pred_latest, scfg)].sort_values("prob", ascending=False)
    uni_latest.head(20).to_csv(out / "latest_top20.csv", index=False)
    print(f"\nlatest date {pred.date.max().date()} top5:")
    print(uni_latest.head(5)[["code", "close", "ret_1", "amount", "prob", "st_proxy"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # report payload
    payload = {
        "cost_round_trip": cost.round_trip,
        "n_dates": int(pred.date.nunique()),
        "date_min": str(pred.date.min().date()),
        "date_max": str(pred.date.max().date()),
        "universe_rows": int(len(uni)),
        "summary": summ.to_dict(orient="records"),
        "curves": {k: {"date": v["date"].dt.strftime("%Y-%m-%d").tolist(),
                       "equity": v["equity"].round(0).tolist()} for k, v in curves.items()},
    }
    with open(out / "report_data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
