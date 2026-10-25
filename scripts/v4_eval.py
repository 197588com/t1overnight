# -*- coding: utf-8 -*-
"""Evaluate v4 against the v2_lag1 baseline on identical folds.

Reads ``outputs/walkforward_v4/pred_*.parquet`` (one file per run), rebuilds the
four T+1-legal clocks, and reports:

  * top1 pick quality per run (gross, hit, t) with the ST share it swallowed
  * the select (2022-2024) / valid (2025-2026) split -- the split is fixed here,
    not fitted, so a v4 that only wins in one half is a red flag
  * cost sensitivity at 100k capital, geometric
  * a paired comparison on the *common* days only, which is the honest A/B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from exec_timing import summarise  # noqa: E402
from t1overnight.backtest import SelectConfig, tradable  # noqa: E402

ANN = 244
COSTS = (10, 13, 15, 18, 20, 25)
SPLIT = pd.Timestamp("2025-01-01")


def load_panel() -> pd.DataFrame:
    cols = ["code", "date", "amt_ratio", "ret_20", "ma_dev_20"]
    df = pd.read_parquet(ROOT / "outputs/t1_dataset.parquet", columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def top1_of(p: pd.DataFrame, panel: pd.DataFrame, drop_st: bool) -> pd.DataFrame:
    p = p.merge(panel, on=["code", "date"], how="left")
    p["d_oc"] = p["close"] / p["open"] - 1
    p["d_oo"] = (1 + p["fwd_co"]) * (1 + p["d_oc"]) - 1
    p["d_oc1"] = (1 + p["fwd_cc"]) * (1 + p["d_oc"]) - 1
    uni = p[tradable(p, SelectConfig(top_n=1))]
    if drop_st:
        uni = uni[~uni["is_st"].fillna(False)]
    top = (uni.sort_values(["date", "score"], ascending=[True, False])
           .groupby("date", sort=True).head(1)
           .set_index("date").sort_index())
    return top


def line_for(top: pd.DataFrame, label: str) -> dict:
    r = top["d_oo"].dropna()
    seg = pd.Series(np.where(r.index < SPLIT, "select", "valid"), index=r.index)
    out = {"run": label, "days": len(r),
           "st_share": float(top["is_st"].fillna(False).mean()),
           "mean_bp": r.mean() * 1e4,
           "geo_bp": float(np.expm1(np.log1p(r.clip(-0.99, None)).mean())) * 1e4,
           "hit": float((r > 0).mean()),
           "t": float(r.mean() / (r.std() / np.sqrt(len(r))))}
    for s in ("select", "valid"):
        rr = r[seg == s]
        out[f"{s}_days"] = len(rr)
        out[f"{s}_bp"] = rr.mean() * 1e4 if len(rr) else np.nan
        out[f"{s}_t"] = (rr.mean() / (rr.std() / np.sqrt(len(rr)))) if len(rr) > 5 else np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="outputs/walkforward_v4")
    ap.add_argument("--out", default="outputs/v4")
    args = ap.parse_args()

    pdir = ROOT / args.pred_dir
    files = sorted(pdir.glob("pred_*.parquet"))
    if not files:
        raise SystemExit(f"no predictions under {pdir}")

    panel = load_panel()
    rows, tops = [], {}
    for f in files:
        run = f.stem.replace("pred_", "")
        p = pd.read_parquet(f)
        p["date"] = pd.to_datetime(p["date"])
        p["code"] = p["code"].astype(str).str.zfill(6)
        drop_st = "drop" in run
        top = top1_of(p, panel, drop_st)
        tops[run] = top
        rows.append(line_for(top, run))
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("=== top1 under D open -> D+1 open (gross) ===")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

    ref = ROOT / "outputs/exit_mode/clock_summary.csv"
    if ref.exists():
        c = pd.read_csv(ref)
        r = c[c.clock.str.startswith("D open -> D+1 open")]
        if len(r):
            print(f"  [reference] v2_lag1 / 300 rounds, published: "
                  f"{float(r.mean_bp.iloc[0]):+.2f}bp t={float(r.t_stat.iloc[0]):+.2f} "
                  f"n={int(r.days.iloc[0])} -- different round count, compare pairs not levels")

    print("\n=== all four legal clocks, gross mean bp ===")
    for run, top in tops.items():
        cells = []
        for c, lab, legal in [("d_oo", "D开->D+1开", True), ("d_oc1", "D开->D+1收", True),
                              ("fwd_co", "D收->D+1开", True), ("fwd_cc", "D收->D+1收", True)]:
            d = summarise(top[c], lab, legal)
            cells.append(f"{lab} {d['mean_bp']:+7.2f}bp t={d['t_stat']:+5.2f}")
        print(f"  {run:<14} " + " | ".join(cells))

    print("\n=== 100k capital, annualised net by round-trip cost ===")
    hdr = f"  {'run':<14}" + "".join(f"{c}bp".rjust(9) for c in COSTS)
    print(hdr)
    for run, top in tops.items():
        geo = float(np.expm1(np.log1p(top["d_oo"].dropna().clip(-0.99, None)).mean()))
        line = f"  {run:<14}"
        for c in COSTS:
            line += f"{((1 + geo - c/1e4)**ANN - 1)*100:+8.1f}%"
        print(line)

    if len(tops) >= 2:
        runs = list(tops)
        a, b = runs[0], runs[1]
        common = tops[a].index.intersection(tops[b].index)
        x = tops[a].loc[common, "d_oo"].dropna()
        y = tops[b].loc[common, "d_oo"].dropna()
        idx = x.index.intersection(y.index)
        x, y = x.loc[idx], y.loc[idx]
        d = x - y
        t = d.mean() / (d.std() / np.sqrt(len(d)))
        print(f"\n=== paired A/B on {len(d)} common days: {a} vs {b} ===")
        print(f"  {a}: {x.mean()*1e4:+.2f}bp   {b}: {y.mean()*1e4:+.2f}bp   "
              f"diff: {d.mean()*1e4:+.2f}bp  t={t:+.2f}  win={(d>0).mean():.3f}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "eval_top1.csv", index=False)
    for run, top in tops.items():
        keep = [c for c in ("code", "is_st", "d_oc", "d_oo", "d_oc1", "fwd_cc", "fwd_co")
                if c in top.columns]
        top.reset_index()[["date"] + keep].to_csv(out / f"top1_{run}.csv", index=False)
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
