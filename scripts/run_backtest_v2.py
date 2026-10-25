# -*- coding: utf-8 -*-
"""Backtest v2: rank-head selection, selective trading, breakeven analysis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t1overnight.backtest import CostConfig, SelectConfig, performance, tradable  # noqa: E402


def daily_stats(sel: pd.DataFrame, uni: pd.DataFrame, cost_rt: float) -> pd.DataFrame:
    g = sel.groupby("date").agg(n=("fwd_cc", "size"), gross=("fwd_cc", "mean"),
                                hit=("fwd_cc", lambda s: float((s > 0).mean())),
                                gap=("fwd_co", "mean"), prob=("prob", "mean"),
                                score=("score", "mean"), price=("close", "mean"),
                                amount=("amount", "median"), st=("st_proxy", "mean"))
    u = uni.groupby("date").agg(uni_gross=("fwd_cc", "mean"),
                                uni_hit=("fwd_cc", lambda s: float((s > 0).mean())),
                                uni_n=("fwd_cc", "size"))
    d = g.join(u, how="left")
    d["net"] = d["gross"] - cost_rt
    d["excess_gross"] = d["gross"] - d["uni_gross"]
    return d.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="outputs/walkforward_v2/predictions.parquet")
    ap.add_argument("--dataset", default="outputs/t1_dataset.parquet")
    ap.add_argument("--out-dir", default="outputs/backtest_v2")
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--universe", default="/Users/yanglin/Desktop/未命名文件夹/datapool/parquet/universe.parquet")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cost = CostConfig()
    cfg = SelectConfig(top_n=1)

    pred = pd.read_parquet(args.pred)
    pred["date"] = pd.to_datetime(pred["date"])
    extra = pd.read_parquet(args.dataset, columns=["code", "date", "amt_ratio", "ret_20", "ma_dev_20"])
    extra["date"] = pd.to_datetime(extra["date"])
    pred = pred.merge(extra, on=["code", "date"], how="left")
    names = pd.read_parquet(args.universe, columns=["code", "name"])
    names["code"] = names["code"].astype(str).str.zfill(6)
    pred["code"] = pred["code"].astype(str).str.zfill(6)
    pred = pred.merge(names, on="code", how="left")

    uni = pred[tradable(pred, cfg)].copy()
    print(f"pred={len(pred):,} tradable={len(uni):,} ({len(uni)/len(pred):.1%}) "
          f"dates={pred.date.nunique()} cost={cost.round_trip:.4%}")

    variants: dict[str, pd.DataFrame] = {}
    for col, tag in [("score", "rank"), ("prob", "prob")]:
        for n in [1, 3, 5, 10]:
            d = uni.sort_values(["date", col], ascending=[True, False])
            variants[f"{tag}_top{n}"] = d.groupby("date").head(n)

    # selective trading: only take a trade when the hit head is confident
    for thr in [0.50, 0.52, 0.55]:
        s = uni[uni["prob"] >= thr]
        d = s.sort_values(["date", "score"], ascending=[True, False])
        v = d.groupby("date").head(1)
        variants[f"rank_top1_prob{int(thr*100)}"] = v

    # no-trade-day variant: skip when today's best score is weak (expanding 40th pct of daily max)
    dmax = variants["rank_top1"][["date", "score"]].rename(columns={"score": "day_max"})
    dmax["thr"] = dmax["day_max"].expanding(60).quantile(0.4).shift(1)
    keep = dmax.dropna()
    variants["rank_top1_timed"] = variants["rank_top1"].merge(
        keep[["date", "thr"]], on="date", how="left").query("score >= thr or thr != thr")

    uni = uni.assign(_rnd=np.random.default_rng(7).random(len(uni)))
    variants["baseline_random"] = uni.sort_values(["date", "_rnd"]).groupby("date").head(1)
    for name, col, asc in [("baseline_highest_amount", "amount", False),
                           ("baseline_weak_close", "close_pos", True) if "close_pos" in uni.columns
                           else ("baseline_amt_ratio", "amt_ratio", False)]:
        d = uni.sort_values(["date", col], ascending=[True, asc])
        variants[name] = d.groupby("date").head(1)

    rows, curves, detail = [], {}, {}
    for name, sel in variants.items():
        sel = sel.dropna(subset=["fwd_cc"])
        if sel.empty:
            continue
        s = daily_stats(sel, uni, cost.round_trip)
        detail[name] = s
        rows.append(performance(s["net"], name))
        curves[name] = pd.DataFrame({
            "date": s["date"].dt.strftime("%Y-%m-%d"),
            "equity": (args.capital * (1 + s["net"]).cumprod()).round(0)})

    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary_variants.csv", index=False)
    show = ["variant", "days", "hit_rate", "mean_daily", "annual_return", "sharpe",
            "max_drawdown", "t_stat"]
    print("\n=== v2 variants, net of 0.20% round trip ===")
    print(summ[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    for name, s in detail.items():
        s.to_csv(out / f"daily_{name}.csv", index=False)

    # breakeven round-trip cost per variant = mean gross daily return
    be = summ[["variant", "days", "hit_rate", "mean_daily"]].copy()
    be["gross_mean"] = be["variant"].map({k: v["gross"].mean() for k, v in detail.items()})
    be["excess_gross_mean"] = be["variant"].map({k: v["excess_gross"].mean() for k, v in detail.items()})
    be["breakeven_cost"] = be["gross_mean"]
    be.to_csv(out / "breakeven.csv", index=False)
    print("\n=== gross mean / breakeven round-trip cost ===")
    print(be[["variant", "gross_mean", "excess_gross_mean", "breakeven_cost"]]
          .to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # decile structure within tradable universe (score head)
    uni["decile"] = uni.groupby("date")["score"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    dec = uni.groupby("decile").agg(n=("fwd_cc", "size"), mean=("fwd_cc", "mean"),
                                    med=("fwd_cc", "median"),
                                    hit=("fwd_cc", lambda s: float((s > 0).mean())),
                                    std=("fwd_cc", "std"),
                                    p05=("fwd_cc", lambda s: s.quantile(0.05)),
                                    ret1=("ret_1", "mean"))
    dec.to_csv(out / "decile_by_score.csv")
    print("\n=== tradable deciles by rank score ===")
    print(dec.to_string(float_format=lambda x: f"{x:.4f}"))

    # daily rank IC of the score within the tradable universe
    ic = uni.groupby("date").apply(
        lambda g: g["score"].corr(g["fwd_cc"], method="spearman") if len(g) > 100 else np.nan,
        include_groups=False).dropna()
    ic_prob = uni.groupby("date").apply(
        lambda g: g["prob"].corr(g["fwd_cc"], method="spearman") if len(g) > 100 else np.nan,
        include_groups=False).dropna()
    ic.to_csv(out / "daily_ic.csv", header=["ic"])
    print(f"\ndaily rank IC(score) = {ic.mean():+.4f}  ICIR={ic.mean()/ic.std()*np.sqrt(244):.2f} "
          f"positive-day={((ic>0).mean()):.1%}")
    print(f"daily rank IC(prob)  = {ic_prob.mean():+.4f}  "
          f"ICIR={ic_prob.mean()/ic_prob.std()*np.sqrt(244):.2f}")

    # cost sensitivity for the headline variant
    head = detail.get("rank_top1")
    sens = []
    for c_rt in [0.0, 0.0005, 0.0007, 0.0010, 0.0015, 0.0020, 0.0025]:
        r = head["gross"] - c_rt
        sens.append({"cost_rt": c_rt, **{k: v for k, v in performance(r, "rank_top1").items()
                                         if k in ("hit_rate", "mean_daily", "annual_return", "t_stat")}})
    pd.DataFrame(sens).to_csv(out / "cost_sensitivity.csv", index=False)
    print("\n=== cost sensitivity (rank_top1) ===")
    print(pd.DataFrame(sens).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # latest snapshot
    last = pred[pred["date"] == pred["date"].max()].copy()
    last = last[tradable(last, cfg)].sort_values("score", ascending=False)
    last.head(20)[["code", "name", "close", "ret_1", "amount", "score", "prob", "st_proxy"]] \
        .to_csv(out / "latest_top20.csv", index=False)
    print(f"\nlatest {pred.date.max().date()} top8 by rank score:")
    print(last.head(8)[["code", "name", "close", "ret_1", "score", "prob"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    payload = {
        "cost_rt": cost.round_trip,
        "date_min": str(pred.date.min().date()),
        "date_max": str(pred.date.max().date()),
        "n_dates": int(pred.date.nunique()),
        "n_universe": int(len(uni)),
        "summary": summ.to_dict(orient="records"),
        "breakeven": be.to_dict(orient="records"),
        "decile": dec.reset_index().to_dict(orient="records"),
        "cost_sensitivity": sens,
        "ic": {"mean": float(ic.mean()), "icir": float(ic.mean() / ic.std() * np.sqrt(244)),
               "positive_share": float((ic > 0).mean()), "std": float(ic.std())},
        "curves": {k: {"date": v["date"].tolist(), "equity": v["equity"].tolist()}
                   for k, v in curves.items() if k in ("rank_top1", "rank_top5", "prob_top1", "baseline_random")},
    }
    with open(out / "report_data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
