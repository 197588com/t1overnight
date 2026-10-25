# -*- coding: utf-8 -*-
"""T+1 overnight model v2 — cross-sectional rank target, actionable universe only.

Two heads per fold:
  * rank head  : regression on within-day percentile rank of fwd_cc (selection score)
  * prob head  : binary on fwd_cc > 0 (hit probability, for calibration/reporting)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
    "gap", "intraday", "close_pos", "amp",
    "vol_5", "vol_20", "vol_ratio",
    "ma_dev_5", "ma_dev_20", "ma_dev_60", "dist_high_20", "dist_low_20", "dist_high_5",
    "log_amt", "amt_ratio", "amt_trend", "turn", "turn_ratio", "log_mktcap",
    "lu_cnt_5", "lu_cnt_20", "z_ret", "streak_up", "days_listed",
    "roe", "net_profit_yoy", "revenue_yoy",
    "cs_ret_1", "cs_ret_5", "cs_amt", "cs_turn", "cs_ma_dev_20",
    "board",
]

BASE_PARAMS = dict(
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=800,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    n_estimators=300,
    n_jobs=10,
    verbose=-1,
)


def make_folds(first_test: str, last_date: str, valid_months: int = 3) -> list[dict]:
    folds, cur, last = [], pd.Timestamp(first_test), pd.Timestamp(last_date)
    while cur <= last:
        te_end = min(cur + pd.DateOffset(months=3) - pd.Timedelta(days=1), last)
        v_end = cur - pd.Timedelta(days=1)
        v_start = v_end - pd.DateOffset(months=valid_months) + pd.Timedelta(days=1)
        folds.append({
            "train_end": (v_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "valid_start": v_start.strftime("%Y-%m-%d"),
            "valid_end": v_end.strftime("%Y-%m-%d"),
            "test_start": cur.strftime("%Y-%m-%d"),
            "test_end": te_end.strftime("%Y-%m-%d"),
        })
        cur = (cur + pd.DateOffset(months=3)).replace(day=1)
    return folds


def actionable(df: pd.DataFrame) -> pd.Series:
    """Rows a 10万 account could plausibly buy at the close (training universe)."""
    amt = df["amount"].where(df["amount"].notna(), df["close"] * df["volume"])
    m = df["close"].between(2.0, 300.0)
    m &= amt >= 20_000_000.0
    m &= df["volume"] > 0
    m &= df["st_proxy"].fillna(0) < 0.5
    m &= df["ret_1"] < (df["band"] - 0.005)
    m &= df["exec_ok"] == 1
    m &= df["ret_20"].notna()
    return m.fillna(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="outputs/t1_dataset.parquet")
    ap.add_argument("--out-dir", default="outputs/walkforward_v2")
    ap.add_argument("--first-test", default="2022-01-01")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--lag", type=int, default=0,
                    help="shift all features by N rows within code (1 = strictly as-of D-1)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cols = ["code", "date"] + FEATURES + ["close", "open", "high", "low", "volume",
                                          "amount", "band", "st_proxy", "exec_ok",
                                          "fwd_cc", "fwd_co", "fwd_oc"]
    df = pd.read_parquet(args.dataset, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df["board"] = df["board"].astype("int16")
    df = df[df["fwd_cc"].notna() & (df["exec_ok"] == 1)].reset_index(drop=True)
    mask = actionable(df)
    df = df[mask].reset_index(drop=True)
    if args.lag:
        # strictly point-in-time: features only from bars <= D-lag
        df = df.sort_values(["code", "date"], kind="mergesort")
        for c in FEATURES:
            df[c] = df.groupby("code", sort=False)[c].shift(args.lag).astype(np.float32)
        df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
        print(f"[lag] features shifted by {args.lag} bar(s) within code", flush=True)
    df["y_rank"] = df.groupby("date")["fwd_cc"].rank(pct=True).astype(np.float32)
    df["y_pb"] = (df["fwd_cc"] > 0).astype(np.int8)
    print(f"[data] actionable rows={len(df):,} dates={df.date.nunique()} "
          f"({time.time()-t0:.1f}s) {df.date.min().date()} -> {df.date.max().date()}", flush=True)
    print(f"[base] mean fwd_cc={df.fwd_cc.mean():+.5f} hit={df.y_pb.mean():.4f}", flush=True)

    params = dict(BASE_PARAMS, n_estimators=args.rounds)
    folds = make_folds(args.first_test, str(df.date.max().date()))
    print(f"[folds] {len(folds)}", flush=True)

    dates = df["date"].values
    preds, meta = [], []
    for i, f in enumerate(folds, 1):
        tf = time.time()
        tr = df[dates <= np.datetime64(f["train_end"])]
        te_mask = (dates > np.datetime64(f["valid_end"])) & (dates <= np.datetime64(f["test_end"]))
        te = df[te_mask]
        if tr.empty or te.empty:
            continue

        m_rank = lgb.LGBMRegressor(objective="regression_l2", **params)
        m_rank.fit(tr[FEATURES], tr["y_rank"], categorical_feature=["board"])

        m_prob = lgb.LGBMClassifier(objective="binary", **params)
        m_prob.fit(tr[FEATURES], tr["y_pb"], categorical_feature=["board"])

        tag = pd.Timestamp(f["test_start"]).strftime("%Y%m")
        m_rank.booster_.save_model(str(out / "models" / f"rank_{tag}.txt"))
        m_prob.booster_.save_model(str(out / "models" / f"prob_{tag}.txt"))

        part = df.loc[te_mask, ["code", "date", "close", "open", "high", "low", "volume",
                                "amount", "band", "st_proxy", "ret_1", "board",
                                "fwd_cc", "fwd_co", "fwd_oc", "y_rank"]].copy()
        part["score"] = m_rank.predict(te[FEATURES]).astype(np.float32)
        part["prob"] = m_prob.predict_proba(te[FEATURES])[:, 1].astype(np.float32)
        part["fold"] = tag
        preds.append(part)

        meta.append({"fold": tag, "train_rows": len(tr), "test_rows": len(te),
                     "base_hit": float(te["y_pb"].mean()) if "y_pb" in te else float((te.fwd_cc > 0).mean()),
                     "seconds": round(time.time() - tf, 1)})
        print(f"  fold {i:02d} {f['test_start']}..{f['test_end']} train={len(tr):>9,} "
              f"base_hit={meta[-1]['base_hit']:.4f} {meta[-1]['seconds']}s", flush=True)

    pred_all = pd.concat(preds, ignore_index=True)
    pred_all.to_parquet(out / "predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(meta).to_csv(out / "fold_metrics.csv", index=False)

    imp = pd.Series(m_rank.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    imp.to_csv(out / "feature_gain_rank.csv", header=["gain"])
    print("\nTop-12 gain (rank head, last fold):")
    print((imp / imp.sum() * 100).head(12).round(2).to_string())
    print(f"\n[done] preds={len(pred_all):,} {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
