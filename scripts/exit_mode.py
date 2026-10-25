# -*- coding: utf-8 -*-
"""Where does the overnight bet actually earn? Three exit variants, same entries.

    cc  buy close D, sell close D+1     (current production rule)
    co  buy close D, sell open D+1      (pure overnight gap, ~18 hours)
    oc  buy open D+1, sell close D+1    (skip the gap, intraday only)

Entries are identical in all three: the top tradable name by model score on day D.
The comparison isolates what the overnight gap contributes versus the next session's
intraday drift, and how much volatility drag each variant carries.

Also reports geometric mean (volatility drag = arithmetic - geometric) and the
break-even round-trip cost, because that is the number that decides deployment.
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


def stats(r: pd.Series, label: str, cost_bp: float = 0.0) -> dict:
    r = r.dropna()
    n = len(r)
    gross = r.mean()
    std = r.std()
    geo = float(np.expm1(np.log1p(r).mean())) if (r > -1).all() else np.nan
    net = r - cost_bp / 1e4
    geo = float(np.expm1(np.log1p(r).mean())) if (r > -1).all() else np.nan
    net_geo = float(np.expm1(np.log1p(net).mean())) if (net > -1).all() else np.nan
    eq = (1 + net).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return {
        "variant": label,
        "days": n,
        "gross_bp": gross * 1e4,
        "net_bp": net.mean() * 1e4,
        "hit": float((r > 0).mean()),
        "std_pct": std * 100,
        "geo_bp": geo * 1e4,
        "drag_bp": (gross - geo) * 1e4,
        "breakeven_bp": geo * 1e4,
        "ann_net_pct": (eq.iloc[-1] ** (ANN / n) - 1) * 100,
        "maxdd_pct": dd * 100,
        "worst_day_bp": float(r.min() * 1e4),
        "p05_bp": float(r.quantile(0.05) * 1e4),
        "t_stat": float(gross / (std / np.sqrt(n))),
    }


def main() -> None:
    pred = pd.read_parquet(ROOT / "outputs/walkforward_v2_lag1/predictions.parquet")
    pred["date"] = pd.to_datetime(pred["date"])
    pred["code"] = pred["code"].astype(str).str.zfill(6)

    cfg = SelectConfig(top_n=1)
    ok = tradable(pred, cfg)
    d = pred.loc[ok].sort_values(["date", "score"], ascending=[True, False])
    top = d.groupby("date", sort=True).head(1).copy()
    print(f"[entries] {len(top):,} sessions, {top.date.min().date()} -> {top.date.max().date()}")

    # sell-side feasibility: a limit-down open on D+1 cannot be sold at the open
    top["open_locked"] = top["fwd_co"] <= -(top["band"] - 0.005)
    print(f"[sell-side] limit-down open on {top['open_locked'].sum()} / {len(top)} sessions "
          f"({top['open_locked'].mean():.2%})")

    rows = []
    for col, label in [("fwd_cc", "close->close (current)"),
                       ("fwd_co", "close->open (pure overnight)"),
                       ("fwd_oc", "open->close (intraday only)")]:
        rows.append(stats(top[col], label))
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\n=== exit variants, identical entries, gross of costs ===")
    print(r[["variant", "days", "gross_bp", "hit", "std_pct", "geo_bp", "drag_bp",
             "breakeven_bp", "ann_net_pct", "maxdd_pct", "p05_bp", "t_stat"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== same, net of a realistic 20bp round trip ===")
    net_rows = [stats(top[c], l, cost_bp=20.0) for c, l in
                [("fwd_cc", "close->close"), ("fwd_co", "close->open"),
                 ("fwd_oc", "open->close")]]
    nr = pd.DataFrame(net_rows)
    print(nr[["variant", "net_bp", "ann_net_pct", "maxdd_pct"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # 10万本金下的最终数字
    print("\n=== 100k capital, full period ===")
    for c, l in [("fwd_cc", "close->close"), ("fwd_co", "close->open"), ("fwd_oc", "open->close")]:
        for cost in (10, 15, 20, 25):
            m = stats(top[c], l, cost_bp=cost)
            print(f"  {l:<14} cost {cost:>2}bp -> net {m['net_bp']:+6.2f}bp/day  "
                  f"ann {m['ann_net_pct']:+7.1f}%  final "
                  f"{100_000 * (1 + m['net_bp'] / 1e4) ** m['days']:>9,.0f}")

    # year-by-year for the best variant
    print("\n=== by year (close->open vs close->close, gross) ===")
    top["year"] = top.date.dt.year
    for y, g in top.groupby("year"):
        print(f"  {y}: days={len(g):>3}  cc={g.fwd_cc.mean()*1e4:+6.1f}bp  "
              f"co={g.fwd_co.mean()*1e4:+6.1f}bp  oc={g.fwd_oc.mean()*1e4:+6.1f}bp  "
              f"gap_share={g.fwd_co.mean()/g.fwd_cc.mean() if g.fwd_cc.mean() else float('nan'):.2f}")

    # per-day dispersion and correlation of the three legs
    print("\n=== leg diagnostics ===")
    print(f"  corr(cc, co) = {top.fwd_cc.corr(top.fwd_co):.3f}   "
          f"corr(cc, oc) = {top.fwd_cc.corr(top.fwd_oc):.3f}")
    print(f"  mean |co| = {top.fwd_co.abs().mean()*1e4:.0f}bp   "
          f"mean |oc| = {top.fwd_oc.abs().mean()*1e4:.0f}bp")

    out = ROOT / "outputs" / "exit_mode"
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "variants.csv", index=False)
    top[["date", "code", "close", "fwd_cc", "fwd_co", "fwd_oc", "band",
         "open_locked"]].to_csv(out / "entries.csv", index=False)
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
