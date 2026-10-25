# -*- coding: utf-8 -*-
"""Fit the production model: same features/objective as the walk-forward, but
trained on every observation available instead of stopping three months short.

Walk-forward models exist to make the backtest honest. For live scoring you want
the newest information, so this writes `<config>_final.txt` next to them and the
daily scripts pick it up automatically (it sorts after the dated files).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_v3  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "outputs/t1_dataset.parquet"))
    ap.add_argument("--extra", default=str(ROOT / "outputs/t1_extra_v3.parquet"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/walkforward_v3/models"))
    ap.add_argument("--configs", default="tri_base,tri_A,tri_B,tri_C")
    ap.add_argument("--rounds", type=int, default=300)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = train_v3.load(args)
    print(f"[fit] all rows={len(df):,} up to {df.date.max().date()}", flush=True)

    for name in [c for c in args.configs.split(",") if c]:
        cfg = train_v3.CONFIGS[name]
        use_ind = any(c.startswith("ind_") for c in cfg["feats"])
        cols = cfg["feats"] + (["ind_code"] if use_ind else [])
        cat = [c for c in ("board", "ind_code") if c in cols]
        if cfg["head"] == "rank":
            mdl = lgb.LGBMRegressor(objective="regression_l2",
                                    **dict(train_v3.PARAMS, n_estimators=args.rounds))
            mdl.fit(df[cols], df["y_rank"], categorical_feature=cat)
        else:
            mdl = lgb.LGBMClassifier(objective="binary",
                                     **dict(train_v3.PARAMS, n_estimators=args.rounds))
            mdl.fit(df[cols], df["y_bad"], categorical_feature=cat)
        p = out / f"{name}_final.txt"
        mdl.booster_.save_model(str(p))
        imp = np.asarray(mdl.booster_.feature_importance("gain"), dtype=float)
        share = imp / imp.sum() * 100
        top = sorted(zip(cols, share), key=lambda x: -x[1])[:10]
        json.dump({"config": name, "rows": int(len(df)),
                   "trained_to": str(df.date.max().date()),
                   "top_gain": {k: round(float(v), 2) for k, v in top}},
                  open(out / f"meta_{name}_final.json", "w"), indent=1)
        print(f"[{name}] -> {p.name}  top gain: "
              + ", ".join(f"{k} {v:.1f}%" for k, v in top[:6]), flush=True)
    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
