# -*- coding: utf-8 -*-
"""Execution timing: which entry/exit clock actually pays, under T+1 legality.

Signal timing is fixed: features come from bars <= D-1 (the strict lag-1 model),
so the score attached to row (code, D) is knowable before D's opening auction.

Legality: A-shares are T+1, so a position bought on D cannot be sold before D+1.
That leaves exactly four legal clocks:

    D open  -> D+1 open      (hold one full session + one night)
    D open  -> D+1 close
    D close -> D+1 open      (current production rule)
    D close -> D+1 close

Two illegal clocks are printed for diagnosis only:
    D open -> D close        (same-day round trip, T+1 violation)
    D+1 open -> D+1 close    (T+1 violation)

All four legal clocks are built from the same rows, so any difference is purely
the clock, not the stock picking.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1overnight.backtest import SelectConfig, tradable  # noqa: E402

ANN = 244
COSTS = (10, 12, 15, 20, 25)


def summarise(r: pd.Series, label: str, legal: bool) -> dict:
    r = r.dropna()
    n = len(r)
    mean, std = r.mean(), r.std()
    geo = float(np.expm1(np.log1p(r.clip(-0.99, None)).mean()))
    return {
        "clock": label,
        "t1": "legal" if legal else "VIOLATION",
        "days": n,
        "mean_bp": mean * 1e4,
        "median_bp": r.median() * 1e4,
        "hit": float((r > 0).mean()),
        "std_pct": std * 100,
        "geo_bp": geo * 1e4,
        "vol_drag_bp": (mean - geo) * 1e4,
        "breakeven_bp": geo * 1e4,
        "t_stat": float(mean / (std / np.sqrt(n))),
        "monthly_win": float(
            pd.Series(r.values, index=r.index).groupby(
                pd.to_datetime(r.index).to_period("M")).mean().gt(0).mean()),
    }


def load() -> pd.DataFrame:
    extra = pd.read_parquet(
        ROOT / "outputs/t1_dataset.parquet",
        columns=["code", "date", "amt_ratio", "ret_20", "ma_dev_20"])
    extra["date"] = pd.to_datetime(extra["date"])
    extra["code"] = extra["code"].astype(str).str.zfill(6)
    p = pd.read_parquet(ROOT / "outputs/walkforward_v2_lag1/predictions.parquet")
    p["date"] = pd.to_datetime(p["date"])
    p["code"] = p["code"].astype(str).str.zfill(6)
    p = p.merge(extra, on=["code", "date"], how="left")
    p["d_oc"] = p["close"] / p["open"] - 1
    p["d_oo"] = (1 + p["fwd_co"]) * (1 + p["d_oc"]) - 1
    p["d_oc1"] = (1 + p["fwd_cc"]) * (1 + p["d_oc"]) - 1
    return p


def main() -> None:
    p = load()
    uni = p[tradable(p, SelectConfig(top_n=1))]
    top = (uni.sort_values(["date", "score"], ascending=[True, False])
           .groupby("date", sort=True).head(1).set_index("date").sort_index())
    print(f"[pool] tradable {len(uni):,}/{len(p):,}  top1 sessions={len(top)}  "
          f"{top.index.min().date()} -> {top.index.max().date()}")

    spec = [("d_oo", "D open -> D+1 open", True),
            ("d_oc1", "D open -> D+1 close", True),
            ("fwd_co", "D close -> D+1 open (current)", True),
            ("fwd_cc", "D close -> D+1 close", True),
            ("d_oc", "D open -> D close", False),
            ("fwd_oc", "D+1 open -> D+1 close", False)]
    rows = [summarise(top[c], lab, legal) for c, lab, legal in spec]
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\n=== execution clocks, strict lag-1 signal, gross of costs ===")
    print(d[["clock", "t1", "days", "mean_bp", "median_bp", "hit", "std_pct",
             "geo_bp", "vol_drag_bp", "breakeven_bp", "t_stat", "monthly_win"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== 100k capital, by round-trip cost (legal clocks only) ===")
    for c, lab, legal in spec[:4]:
        geo = float(np.expm1(np.log1p(top[c].dropna().clip(-0.99, None)).mean()))
        line = f"  {lab:<30}"
        for cost in COSTS:
            net = geo - cost / 1e4
            line += f" {cost}bp:{((1+net)**ANN-1)*100:+7.1f}%"
        print(line + "   (annualised)")

    print("\n=== year by year, gross mean bp ===")
    top["year"] = top.index.year
    hdr = "  year  n    " + "".join(f"{lab.split(' (')[0]:>22}" for _, lab, _ in spec[:4])
    print(hdr)
    for y, g in top.groupby("year"):
        line = f"  {y}  {len(g):>3}   "
        for c, _, _ in spec[:4]:
            line += f"{g[c].mean()*1e4:>+22.1f}"
        print(line)

    print("\n=== board split (legal best clock d_oo) ===")
    for b, g in top.groupby("board"):
        r = g["d_oo"].dropna()
        if len(r) < 30:
            continue
        t = r.mean() / (r.std() / np.sqrt(len(r)))
        print(f"  board {int(b)}: n={len(r):>3} mean={r.mean()*1e4:+7.1f}bp "
              f"hit={(r>0).mean():.3f} t={t:+.2f}")

    print("\n=== upside capture: what the current rule throws away ===")
    dc = top["d_oc"].mean() * 1e4
    co = top["fwd_co"].mean() * 1e4
    cc = top["fwd_cc"].mean() * 1e4
    print(f"  D intraday leg      (D open->D close) : {dc:+7.2f}bp  [T+1 illegal]")
    print(f"  overnight leg       (D close->D+1 open): {co:+7.2f}bp")
    print(f"  = current rule      (D close->D+1 close): {cc:+7.2f}bp")
    print(f"  D open -> D+1 open  (legal, best)     : {top['d_oo'].mean()*1e4:+7.2f}bp")

    out = ROOT / "outputs" / "exit_mode"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "clock_summary.csv", index=False)
    top.reset_index()[["date", "code", "open", "close", "d_oc", "d_oo", "d_oc1",
                       "fwd_cc", "fwd_co", "board"]].to_csv(out / "clocks.csv", index=False)
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
