# -*- coding: utf-8 -*-
"""Quarterly walk-forward training for the T+1 overnight model."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t1overnight.features import FEATURES  # noqa: E402

AUX = ["code", "name", "date", "close", "open", "high", "low", "volume", "amount",
       "band", "st_proxy", "ret_1", "fwd_cc", "fwd_co", "fwd_oc"]

PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=300,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l2=1.0,
    n_estimators=400,
    n_jobs=10,
    verbose=-1,
)

CAT_FEATURES = ["board"]


def quarter_starts(start: str, end: str) -> list[pd.Timestamp]:
    out = []
    cur = pd.Timestamp(start)
    last = pd.Timestamp(end)
    while cur <= last:
        out.append(cur)
        cur = cur + pd.DateOffset(months=3) - pd.Timedelta(days=0)
        cur = (cur.replace(day=1) + pd.DateOffset(months=0))
    return out


def make_folds(first_test: str, last_date: str, valid_months: int = 3) -> list[dict]:
    folds = []
    cur = pd.Timestamp(first_test)
    last = pd.Timestamp(last_date)
    while cur <= last:
        test_start = cur
        test_end = min(cur + pd.DateOffset(months=3) - pd.Timedelta(days=1), last)
        valid_end = test_start - pd.Timedelta(days=1)
        valid_start = valid_end - pd.DateOffset(months=valid_months) + pd.Timedelta(days=1)
        folds.append({
            "train_end": (valid_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "valid_start": valid_start.strftime("%Y-%m-%d"),
            "valid_end": valid_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
        })
        cur = (cur + pd.DateOffset(months=3)).replace(day=1)
    return folds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="outputs/t1_dataset.parquet")
    ap.add_argument("--out-dir", default="outputs/walkforward")
    ap.add_argument("--first-test", default="2022-01-01")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--min-child", type=int, default=300)
    ap.add_argument("--calibrate", action="store_true", default=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cols = ["code", "date"] + FEATURES + ["close", "open", "high", "low", "volume",
                                          "amount", "band", "st_proxy",
                                          "fwd_cc", "fwd_co", "fwd_oc"]
    df = pd.read_parquet(args.dataset, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df["board"] = df["board"].astype("int16")
    df = df[df["fwd_cc"].notna()].reset_index(drop=True)
    print(f"[load] {df.shape} in {time.time()-t0:.1f}s, "
          f"{df.date.min().date()} -> {df.date.max().date()}", flush=True)

    params = dict(PARAMS)
    params.update(n_estimators=args.rounds, learning_rate=args.lr,
                  num_leaves=args.leaves, min_child_samples=args.min_child)

    folds = make_folds(args.first_test, str(df.date.max().date()))
    print(f"[folds] {len(folds)} quarterly folds", flush=True)

    preds = []
    meta = []
    dates = df["date"].values
    for i, fold in enumerate(folds, 1):
        tf = time.time()
        tr = df[dates <= np.datetime64(fold["train_end"])]
        va = df[(dates >= np.datetime64(fold["valid_start"])) & (dates <= np.datetime64(fold["valid_end"]))]
        te_mask = (dates > np.datetime64(fold["valid_end"])) & (dates <= np.datetime64(fold["test_end"]))
        te = df[te_mask]
        if tr.empty or va.empty or te.empty:
            print(f"  fold {i} skipped (empty split)", flush=True)
            continue

        y_tr = (tr["fwd_cc"] > 0).astype(int).values
        y_va = (va["fwd_cc"] > 0).astype(int).values

        model = lgb.LGBMClassifier(**params)
        model.fit(
            tr[FEATURES], y_tr,
            eval_set=[(va[FEATURES], y_va)],
            eval_metric="binary_logloss",
            categorical_feature=CAT_FEATURES,
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )

        p_va = model.predict_proba(va[FEATURES])[:, 1]
        p_te = model.predict_proba(te[FEATURES])[:, 1]

        # isotonic calibration fitted on the validation quarter only
        p_cal = p_te
        if args.calibrate:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_va, y_va)
            p_cal = iso.predict(p_te)

        tag = pd.Timestamp(fold["test_start"]).strftime("%Y%m")
        model.booster_.save_model(str(out / "models" / f"lgbm_{tag}.txt"))

        part = df.loc[te_mask, ["code", "date", "close", "open", "high", "low",
                                "volume", "amount", "band", "st_proxy", "ret_1",
                                "fwd_cc", "fwd_co", "fwd_oc"]].copy()
        part["prob"] = p_te
        part["prob_cal"] = p_cal
        part["fold"] = tag
        preds.append(part)

        meta.append({
            "fold": tag,
            "train_rows": int(len(tr)),
            "valid_rows": int(len(va)),
            "test_rows": int(len(te)),
            "best_iter": int(getattr(model, "best_iteration_", 0) or 0),
            "valid_logloss": float(model.best_score_["valid_0"]["binary_logloss"]),
            "valid_hit": float((p_va > 0.5).mean() * 0 + (y_va.mean())),
            "test_base_hit": float((te["fwd_cc"] > 0).mean()),
            "seconds": round(time.time() - tf, 1),
        })
        print(f"  fold {i:02d} test={fold['test_start']}..{fold['test_end']} "
              f"train={len(tr):>9,} iter={meta[-1]['best_iter']:>3} "
              f"base_hit={meta[-1]['test_base_hit']:.4f} {meta[-1]['seconds']}s", flush=True)

    pred_all = pd.concat(preds, ignore_index=True)
    pred_all.to_parquet(out / "predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(meta).to_csv(out / "fold_metrics.csv", index=False)

    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    imp.to_csv(out / "feature_gain_lastfold.csv", header=["gain"])
    print("\nTop-12 gain (last fold):")
    print(imp.head(12).round(1).to_string())
    print(f"\n[done] predictions={len(pred_all):,} total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
