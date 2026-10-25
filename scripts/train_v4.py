# -*- coding: utf-8 -*-
"""v4 walk-forward: same folds, same features, ST removed from the universe.

The v2/v3 models learned on a universe where ST names were only caught by
``st_proxy`` (a volatility-based band guess) -- which misses ChiNext/STAR ST tags
entirely.  Measured cost: 91 of 1,161 top1 picks (7.84%) landed on a stock that
really was ST that day, and in the 2025-2026 validation window that share rose to
17.8% and produced ~42% of the segment's P&L.  That is not an edge we can or
should trade, so v4 removes those rows before the model ever sees them.

Runs (policy:head), all on identical folds:
    keep:rank   the v2_lag1 baseline, re-fitted here so the A/B shares folds
    drop:rank   ST excluded  -> the production head
    drop:triage ST excluded, bottom-20% binary head -> the risk gate
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import st_mask

ROOT = Path(__file__).resolve().parents[1]

BASE = [
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

PARAMS = dict(learning_rate=0.03, num_leaves=31, min_child_samples=600,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, n_jobs=10, verbose=-1)


def make_folds(first_test: str, last_date: str, valid_months: int = 3) -> list[dict]:
    folds, cur, last = [], pd.Timestamp(first_test), pd.Timestamp(last_date)
    while cur <= last:
        te_end = min(cur + pd.DateOffset(months=3) - pd.Timedelta(days=1), last)
        v_end = cur - pd.Timedelta(days=1)
        v_start = v_end - pd.DateOffset(months=valid_months) + pd.Timedelta(days=1)
        folds.append({"train_end": (v_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                      "valid_end": v_end.strftime("%Y-%m-%d"),
                      "test_start": cur.strftime("%Y-%m-%d"),
                      "test_end": te_end.strftime("%Y-%m-%d")})
        cur = (cur + pd.DateOffset(months=3)).replace(day=1)
    return folds


def load(args) -> pd.DataFrame:
    need = ["code", "date"] + BASE + ["close", "open", "volume", "amount", "band",
                                      "st_proxy", "exec_ok", "ret_20",
                                      "fwd_cc", "fwd_co", "fwd_oc"]
    seen, need_u = set(), []
    for c in need:
        if c not in seen:
            seen.add(c); need_u.append(c)
    df = pd.read_parquet(args.dataset, columns=need_u)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["fwd_cc"].notna() & (df["exec_ok"] == 1)]
    amt = df["amount"].where(df["amount"].notna(), df["close"] * df["volume"])
    m = (df["close"].between(2.0, 300.0) & (amt >= 20_000_000.0) & (df["volume"] > 0)
         & (df["st_proxy"].fillna(0) < 0.5) & (df["ret_1"] < (df["band"] - 0.005))
         & df["ret_20"].notna())
    df = df[m.fillna(False)].reset_index(drop=True)
    print(f"[universe] proxy-filtered rows={len(df):,}", flush=True)

    # strict point-in-time: every feature shifted one bar back within code.
    # NOTE the shift runs on the *full* proxy-filtered frame so that dropping ST
    # rows later does not hand a stock a feature from two bars ago.
    df = df.sort_values(["code", "date"], kind="mergesort")
    for c in BASE:
        df[c] = df.groupby("code", sort=False)[c].shift(1).astype(np.float32)
    df["board"] = df["board"].fillna(-1).astype("int16")
    df = df.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)

    t0 = time.time()
    st_mask.annotate(df)                       # in place: (date, code) already sorted
    print(f"[st] annotate {time.time()-t0:.0f}s | is_st={int(df.is_st.sum()):,} "
          f"({df.is_st.mean()*100:.2f}%), is_delist={int(df.is_delist.sum()):,}",
          flush=True)

    df["y_rank"] = df.groupby("date")["fwd_cc"].rank(pct=True).astype(np.float32)
    df["y_bad"] = (df["y_rank"] <= 0.2).astype(np.int8)
    print(f"[data] {len(df):,} rows, {df.date.nunique()} dates, "
          f"{df.date.min().date()} -> {df.date.max().date()}", flush=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="outputs/t1_dataset.parquet")
    ap.add_argument("--out-dir", default="outputs/walkforward_v4")
    ap.add_argument("--first-test", default="2022-01-01")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--runs", default="keep:rank,drop:rank,drop:triage")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = load(args)
    runs = [tuple(r.split(":")) for r in args.runs.split(",") if r]
    folds = make_folds(args.first_test, str(df.date.max().date()))
    dates = df["date"].values
    st_flag = (df["is_st"].to_numpy(dtype=bool)
               | df["is_delist"].to_numpy(dtype=bool))
    print(f"[plan] {len(folds)} folds x {len(runs)} runs | "
          f"universe keep={len(df):,} drop={int((~st_flag).sum()):,}", flush=True)

    frames: dict[str, list[pd.DataFrame]] = {f"{p}:{h}": [] for p, h in runs}
    detail = []

    for i, f in enumerate(folds, 1):
        tf = time.time()
        tr_mask = dates <= np.datetime64(f["train_end"])
        te_mask = ((dates > np.datetime64(f["valid_end"]))
                   & (dates <= np.datetime64(f["test_end"])))
        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            continue
        tag = pd.Timestamp(f["test_start"]).strftime("%Y%m")

        for pol, head in runs:
            key = f"{pol}:{head}"
            k = ~st_flag if pol == "drop" else np.ones(len(df), bool)
            tr = df[tr_mask & k]
            te = df[te_mask & k]
            if head == "rank":
                mdl = lgb.LGBMRegressor(objective="regression_l2",
                                        **dict(PARAMS, n_estimators=args.rounds))
                mdl.fit(tr[BASE], tr["y_rank"], categorical_feature=["board"])
                pred = mdl.predict(te[BASE])
                head_col = "score"
            else:
                mdl = lgb.LGBMClassifier(objective="binary",
                                         **dict(PARAMS, n_estimators=args.rounds))
                mdl.fit(tr[BASE], tr["y_bad"], categorical_feature=["board"])
                pred = mdl.predict_proba(te[BASE])[:, 1]
                head_col = "bad_prob"
            mdl.booster_.save_model(str(out / "models" / f"{pol}_{head}_{tag}.txt"))

            part = te[["code", "date", "open", "close", "fwd_cc", "fwd_co", "fwd_oc",
                       "ret_1", "amount", "volume", "band", "st_proxy", "board",
                       "y_rank", "is_st"]].copy()
            part[head_col] = pred.astype(np.float32)
            part["run"] = key
            frames[key].append(part)

            detail.append({"fold": tag, "run": key, "train_rows": int((tr_mask & k).sum()),
                           "test_rows": int((te_mask & k).sum())})
        print(f"  fold {i:02d} {f['test_start']}..{f['test_end']} "
              f"train={int(tr_mask.sum()):>8,} test={int(te_mask.sum()):>7,} "
              f"{time.time()-tf:.0f}s", flush=True)

    meta = []
    for key, parts in frames.items():
        if not parts:
            continue
        p = pd.concat(parts, ignore_index=True)
        p.to_parquet(out / f"pred_{key.replace(':','_')}.parquet", index=False,
                     compression="zstd")
        col = "score" if key.endswith("rank") else "bad_prob"
        sign = 1.0 if col == "score" else -1.0
        ic = (p.groupby("date").apply(
            lambda g: g[col].corr(g["fwd_cc"], method="spearman"), include_groups=False)
            .dropna())
        pct = p.groupby("date")[col].rank(pct=True)
        p["bl"] = (pct <= 0.2) if col == "score" else (pct >= 0.8)
        kept = p[~p["bl"]]
        top1 = p.loc[p.groupby("date")[col].idxmax()] if col == "score" else None
        line = {"run": key, "ic": float(ic.mean()),
                "icir": float(ic.mean() / ic.std() * np.sqrt(244)),
                "n_days": int(p.date.nunique())}
        if top1 is not None:
            line["top1_mean_bp"] = float(top1.fwd_cc.mean() * 1e4)
            line["top1_t"] = float(top1.fwd_cc.mean() / top1.fwd_cc.std()
                                   * np.sqrt(len(top1)))
            line["top1_st_share"] = float(top1.is_st.mean())
        meta.append(line)
        print(f"[{key}] IC={line['ic']:+.4f} ICIR={line['icir']:+.2f} | "
              f"blacklist mean={p.loc[p.bl,'fwd_cc'].mean()*1e4:+.1f}bp "
              f"hit={(p.loc[p.bl,'fwd_cc']>0).mean():.3f} | "
              f"kept mean={kept.fwd_cc.mean()*1e4:+.1f}bp"
              + (f" | top1={line['top1_mean_bp']:+.2f}bp t={line['top1_t']:.2f} "
                 f"ST={line['top1_st_share']*100:.2f}%" if top1 is not None else ""),
              flush=True)
        json.dump(line, open(out / f"meta_{key.replace(':','_')}.json", "w"), indent=1)

    pd.DataFrame(detail).to_csv(out / "fold_rows.csv", index=False)
    pd.DataFrame(meta).to_csv(out / "summary.csv", index=False)
    print(f"\n[done] {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
