# -*- coding: utf-8 -*-
"""Why does the model keep landing on ST names?

Two pieces, both measured on the same universe the model trains on:

1. model side  -- LightGBM gain share per feature, averaged over the 20 fold
   models, so we know *which* features actually move the score.
2. pick side   -- for each top1 pick, its cross-sectional percentile on those
   features, split by whether the stock really was ST that day.

A feature only explains the drift if it scores high on both: large gain share
*and* the ST picks sitting far in the tail.  ``tilt`` = gain_share x
(pct_st - pct_nonst) is the single number to sort by.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import st_mask

ROOT = Path(__file__).resolve().parents[1]
# the highest-gain features from the fold models, plus the names that a ST stock
# would plausibly sit at the top of.  Picked before looking at the pick profile.
DIAG = ["cs_turn", "turn", "cs_ret_5", "amp", "days_listed", "cs_ret_1",
        "cs_ma_dev_20", "cs_amt", "vol_20", "lu_cnt_5", "log_mktcap", "turn_ratio"]
BASE = ["ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
        "gap", "intraday", "close_pos", "amp",
        "vol_5", "vol_20", "vol_ratio",
        "ma_dev_5", "ma_dev_20", "ma_dev_60", "dist_high_20", "dist_low_20", "dist_high_5",
        "log_amt", "amt_ratio", "amt_trend", "turn", "turn_ratio", "log_mktcap",
        "lu_cnt_5", "lu_cnt_20", "z_ret", "streak_up", "days_listed",
        "roe", "net_profit_yoy", "revenue_yoy",
        "cs_ret_1", "cs_ret_5", "cs_amt", "cs_turn", "cs_ma_dev_20", "board"]


def universe(path: Path) -> pd.DataFrame:
    need = ["code", "date", "close", "volume", "amount", "band", "st_proxy",
            "exec_ok", "ret_20", "ret_1", "fwd_cc"] + DIAG
    need = list(dict.fromkeys(need))
    df = pd.read_parquet(path, columns=need)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["fwd_cc"].notna() & (df["exec_ok"] == 1)]
    amt = df["amount"].where(df["amount"].notna(), df["close"] * df["volume"])
    m = (df["close"].between(2.0, 300.0) & (amt >= 20_000_000.0) & (df["volume"] > 0)
         & (df["st_proxy"].fillna(0) < 0.5) & (df["ret_1"] < (df["band"] - 0.005))
         & df["ret_20"].notna())
    df = df[m.fillna(False)].drop(columns=["close", "volume", "amount", "band",
                                          "st_proxy", "exec_ok"]).reset_index(drop=True)
    df = df.sort_values(["code", "date"], kind="mergesort")
    for c in DIAG:
        df[c] = df.groupby("code", sort=False)[c].shift(1).astype(np.float32)
    df = df.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)
    st_mask.annotate(df)
    return df


def gain_table(models: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(models.glob("rank_*.txt")):
        b = lgb.Booster(model_file=str(p))
        g = pd.Series(b.feature_importance("gain"), index=b.feature_name(), dtype=float)
        rows.append((g / g.sum()).rename(p.stem))
    t = pd.concat(rows, axis=1)
    out = pd.DataFrame({"gain_share": t.mean(axis=1), "gain_sd": t.std(axis=1)})
    return out.sort_values("gain_share", ascending=False)


def main() -> None:
    mdir = ROOT / "outputs" / "walkforward_v2_lag1" / "models"
    gain = gain_table(mdir)
    gain.index.name = "feature"
    gain["gain_rank"] = np.arange(1, len(gain) + 1)
    print("[gain] top10 (gain share %)")
    print(gain.head(10).assign(gain_share=lambda d: (d.gain_share * 100).round(2),
                               gain_sd=lambda d: (d.gain_sd * 100).round(2))
          .to_string())

    df = universe(ROOT / "outputs" / "t1_dataset.parquet")
    print(f"[universe] {len(df):,} rows, ST {df.is_st.mean()*100:.2f}%")

    pct = df.groupby("date")[DIAG].rank(pct=True).astype(np.float32)
    key = pd.DataFrame({"date": df["date"].to_numpy(),
                        "code": df["code"].astype(str).to_numpy()})
    pct = pd.concat([key, pct], axis=1)
    del df

    t1 = pd.read_csv(ROOT / "outputs" / "exit_mode" / "top1_with_st_flag.csv",
                     parse_dates=["date"], dtype={"code": str})
    t1 = t1.sort_values(["date", "code"]).reset_index(drop=True)
    # the CSV's own ``is_st`` is the *current-name* shortcut; the PIT rebuild is st_hist
    st_flag = pd.to_numeric(t1.get("st_hist", 0), errors="coerce").fillna(0) > 0
    pick = t1.merge(pct, on=["date", "code"], how="left")
    print(f"[pick] {len(pick)} 天, 特征缺失 {int(pick[DIAG[0]].isna().sum())} 天")

    a, b = pick[st_flag], pick[~st_flag]
    tab = pd.DataFrame({
        "feature": DIAG,
        "pct_st": [a[f].mean() * 100 for f in DIAG],
        "pct_nonst": [b[f].mean() * 100 for f in DIAG],
    })
    tab["gap"] = tab.pct_st - tab.pct_nonst
    tab = tab.merge(gain.reset_index(), on="feature", how="left")
    tab["tilt"] = tab.gain_share * tab.gap
    tab = tab.sort_values("tilt", ascending=False)
    print(f"\n[pick profile] 被选中当天的横截面分位（0-100），ST {len(a)} 天 / 非 ST {len(b)} 天")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

    out = ROOT / "outputs" / "exit_mode"
    tab.to_csv(out / "st_attrib.csv", index=False)
    gain.to_csv(out / "feature_gain.csv")
    print(f"\n-> {out/'st_attrib.csv'} , {out/'feature_gain.csv'}")


if __name__ == "__main__":
    main()
