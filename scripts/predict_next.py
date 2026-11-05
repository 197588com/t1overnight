# -*- coding: utf-8 -*-
"""Score the next session with the latest fold of the strictly-lagged model.

Only the lag-1 model can produce a forward list: its features need nothing but
bars up to D-1, so scoring the newest bar gives a ranking usable at the open of D.
"""
from __future__ import annotations

import os
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_v2 import FEATURES  # noqa: E402
from t1overnight.backtest import SelectConfig, tradable  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="outputs/t1_dataset.parquet")
    ap.add_argument("--models", default="outputs/walkforward_v2_lag1/models")
    ap.add_argument("--universe",
                    default=os.environ.get("T1_UNIVERSE", os.path.expanduser("~/datapool/parquet") + "/universe.parquet"))
    ap.add_argument("--out-dir", default="outputs/live")
    args = ap.parse_args()

    db = Path(args.dataset)
    last_date = pd.read_parquet(db, columns=["date"])["date"].max()
    df = pd.read_parquet(db, columns=["code", "date"] + FEATURES +
                         ["close", "open", "high", "low", "volume", "amount",
                          "band", "st_proxy", "exec_ok"])
    df["date"] = pd.to_datetime(df["date"])
    df["board"] = df["board"].astype("int16")
    snap = df[df["date"] == last_date].copy()

    models = sorted(Path(args.models).glob("rank_*.txt"))
    if not models:
        raise FileNotFoundError(f"no rank model in {args.models}")
    booster = lgb.Booster(model_file=str(models[-1]))
    print(f"[model] {models[-1].name} | snapshot date {last_date.date()} | rows {len(snap):,}")

    snap["score"] = booster.predict(snap[FEATURES])
    cfg = SelectConfig(top_n=1)
    snap = snap[tradable(snap, cfg)].copy()
    snap["pct"] = snap["score"].rank(pct=True)

    names = pd.read_parquet(args.universe, columns=["code", "name"])
    names["code"] = names["code"].astype(str).str.zfill(6)
    names["is_st_now"] = names["name"].str.contains("ST").astype(int)
    snap["code"] = snap["code"].astype(str).str.zfill(6)
    snap = snap.merge(names, on="code", how="left")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["code", "name", "close", "ret_1", "score", "pct", "amount", "is_st_now"]
    snap.sort_values("score").to_csv(out / "scores_snapshot.csv", index=False, encoding="utf-8-sig")

    bl = snap[snap["pct"] < 0.2].sort_values("score")
    bl[cols].to_csv(out / "blacklist_next_session.csv", index=False, encoding="utf-8-sig")

    print(f"\nsnapshot {last_date.date()} -> next session ranking")
    print(f"scored universe: {len(snap):,} | blacklist (<20th pct): {len(bl):,}")
    print("\nbottom 10 (avoid):")
    print(bl.head(10)[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\ntop 10 (no positive alpha in strict mode - reference only):")
    print(snap.nlargest(10, "score")[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
