# -*- coding: utf-8 -*-
"""排雷日常脚本 — the deployable piece.

Timing convention (this is the part that is easy to get wrong):
    the strict model's features at row D are the panel's values of D-1,
    and its label is close(D) -> close(D+1).
So a row dated D-1 answers the question "I am about to buy on D and sell on
D+1 - is this name dangerous?". Running this script after the close of D-1 is
therefore enough; no intraday data is needed.

Usage
-----
    python scripts/triage_daily.py                  # latest bar, next session's list
    python scripts/triage_daily.py --asof 2026-10-23
    python scripts/triage_daily.py --check 600519 000001
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1overnight.features import FEATURES  # noqa: E402
from t1overnight.features_v3 import IND, MKT, SHAPE  # noqa: E402

CUT = 0.20          # size of the blacklist
PANEL_COLS = list(FEATURES) + ["close", "open", "high", "low",
                               "volume", "amount", "band",
                               "st_proxy", "exec_ok", "ret_20"]


def newest_model(model_dir: str, config: str) -> Path:
    files = sorted(glob.glob(str(Path(model_dir) / f"{config}_*.txt")))
    if not files:
        raise FileNotFoundError(f"no {config}_*.txt under {model_dir}")
    return Path(files[-1])


def load_day(dataset: str, extra: str, asof: str | None) -> tuple[pd.DataFrame, pd.Timestamp]:
    cols = list(dict.fromkeys(["code", "date"] + PANEL_COLS))
    d = pd.read_parquet(dataset, columns=cols)
    d["date"] = pd.to_datetime(d["date"])
    latest = d["date"].max() if asof is None else pd.Timestamp(asof)
    d = d[d["date"] == latest]
    if d.empty:
        raise SystemExit(f"no rows on {latest.date()}")

    ex = pd.read_parquet(extra, columns=["code", "date", "ind_code"] + SHAPE + IND + MKT)
    ex["date"] = pd.to_datetime(ex["date"])
    d = d.merge(ex[ex["date"] == latest], on=["code", "date"], how="left")
    d["ind_code"] = d["ind_code"].fillna(-1).astype("int16")
    d["board"] = d["board"].fillna(-1).astype("int16")
    return d.reset_index(drop=True), latest


def tradable(d: pd.DataFrame) -> pd.Series:
    """Same screen training used, applied with D-1 information only."""
    amt = d["amount"].where(d["amount"].notna(), d["close"] * d["volume"])
    m = (d["close"].between(2.0, 300.0) & (amt >= 20_000_000.0) & (d["volume"] > 0)
         & (d["st_proxy"].fillna(0) < 0.5) & (d["ret_1"] < (d["band"] - 0.005))
         & d["ret_20"].notna())
    return m.fillna(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "outputs/t1_dataset.parquet"))
    ap.add_argument("--extra", default=str(ROOT / "outputs/t1_extra_v3.parquet"))
    ap.add_argument("--model-dir", default=str(ROOT / "outputs/walkforward_v3/models"))
    ap.add_argument("--config", default="tri_B")
    ap.add_argument("--names", default="/Users/yanglin/Desktop/未命名文件夹/datapool/parquet/universe.parquet")
    ap.add_argument("--asof", default=None)
    ap.add_argument("--cut", type=float, default=CUT)
    ap.add_argument("--check", nargs="*", default=None, help="inspect specific codes")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/triage"))
    args = ap.parse_args()

    booster = lgb.Booster(model_file=str(newest_model(args.model_dir, args.config)))
    names = booster.feature_name()
    print(f"model {Path(newest_model(args.model_dir, args.config)).name}  "
          f"features={len(names)}")

    d, latest = load_day(args.dataset, args.extra, args.asof)
    print(f"as-of bar {latest.date()}  rows={len(d):,}")

    uni = d[tradable(d)].copy()
    missing = [c for c in names if c not in uni.columns]
    if missing:
        raise SystemExit(f"features missing from the frame: {missing}")
    x = uni[names].copy()
    for c in ("board", "ind_code"):
        if c in x.columns:
            x[c] = x[c].astype("int16")
    uni["bad_prob"] = booster.predict(x).astype(np.float32)
    uni["pct"] = uni["bad_prob"].rank(pct=True)
    uni["flagged"] = uni["pct"] >= 1.0 - args.cut

    try:
        nm = pd.read_parquet(args.names)
        nm["code"] = nm["code"].astype(str).str.zfill(6)
        uni = uni.merge(nm[["code", "name"]], on="code", how="left")
        uni["is_st_name"] = uni["name"].fillna("").str.contains("ST").astype(int)
    except Exception:  # noqa: BLE001
        uni["name"] = ""
        uni["is_st_name"] = 0

    uni = uni.sort_values("bad_prob", ascending=False)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = latest.strftime("%Y-%m-%d")
    cols = ["code", "name", "bad_prob", "pct", "flagged", "close", "ret_1", "ret_5",
            "turn", "amt_ratio", "amount", "is_st_name", "board"]
    cols = [c for c in cols if c in uni.columns]
    uni[cols].to_csv(out / f"scores_{tag}.csv", index=False, encoding="utf-8-sig")
    uni[uni["flagged"]][cols].to_csv(out / f"blacklist_{tag}.csv", index=False,
                                     encoding="utf-8-sig")

    n_flag = int(uni["flagged"].sum())
    print(f"\nscored {len(uni):,} tradable names, flagged {n_flag:,} "
          f"({n_flag/len(uni):.1%})  -> the next session's do-not-buy list")
    print("\nworst 15:")
    show = ["code", "name", "bad_prob", "close", "ret_1", "ret_5", "turn", "amount"]
    print(uni.head(15)[[c for c in show if c in uni.columns]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nnames the model likes most (least dangerous 8):")
    print(uni.tail(8)[[c for c in show if c in uni.columns]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.check:
        print("\n=== single-name check ===")
        for c in args.check:
            c = str(c).zfill(6)
            row = uni[uni["code"] == c]
            if row.empty:
                print(f"  {c}: not in the tradable universe on {tag}")
                continue
            r = row.iloc[0]
            verdict = "⛔ 黑名单" if r["flagged"] else "✅ 未触雷"
            print(f"  {c} {r.get('name','')}: bad_prob={r['bad_prob']:.4f} "
                  f"pct={r['pct']:.3f}  {verdict}")
    print(f"\n[done] -> {out}/blacklist_{tag}.csv")


if __name__ == "__main__":
    main()
