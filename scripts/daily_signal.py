# -*- coding: utf-8 -*-
"""Daily trade signal under the production rule (open-to-open, T+1 legal).

Rule baked in here so it cannot be misapplied:

    signal frame   D-1 close (features lagged 1 bar, model already trained that way)
    entry          D 09:30 opening auction
    skip today     if the opening gap (open / prev_close - 1) exceeds GAP_MAX
    exit           D+1 09:30 opening auction

Why open-to-open: with the strict lag-1 model the score on row (code, D-1) is knowable
before D's opening auction. Waiting until D's close gives up the whole D intraday leg
(+40.8bp in backtest) and keeps only the overnight leg (-18.5bp).

Usage:
    python scripts/daily_signal.py                  # newest bar in the dataset
    python scripts/daily_signal.py --asof 2026-10-23
    python scripts/daily_signal.py --top 5
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_v2 import FEATURES  # noqa: E402
from t1overnight.backtest import SelectConfig, tradable  # noqa: E402

GAP_MAX = 0.015        # skip the day if the open gaps up more than this
COST_RT = 0.0020       # round-trip cost assumed in the ledger's net column
BOARD = {0: "沪主板", 1: "深主板", 2: "创业板", 3: "科创板", 4: "北交所"}


def newest_model(model_dir: Path) -> Path:
    ms = sorted(model_dir.glob("rank_*.txt"))
    if not ms:
        raise FileNotFoundError(f"no rank_*.txt under {model_dir}")
    return ms[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="outputs/t1_dataset.parquet")
    ap.add_argument("--models", default="outputs/walkforward_v2_lag1/models")
    ap.add_argument("--universe",
                    default="/Users/yanglin/Desktop/未命名文件夹/datapool/parquet/universe.parquet")
    ap.add_argument("--out-dir", default="outputs/live")
    ap.add_argument("--asof", default=None, help="signal bar date; default = newest in dataset")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    db = Path(args.dataset)
    df = pd.read_parquet(db, columns=["code", "date"] + FEATURES +
                         ["close", "open", "high", "low", "volume", "amount",
                          "band", "st_proxy", "exec_ok"])
    df["date"] = pd.to_datetime(df["date"])
    df["board"] = df["board"].astype("int16")

    asof = pd.Timestamp(args.asof) if args.asof else df["date"].max()
    snap = df[df["date"] == asof].copy()
    if snap.empty:
        raise SystemExit(f"no rows on {asof.date()}")

    model = newest_model(Path(args.models))
    booster = lgb.Booster(model_file=str(model))
    print(f"[model] {model.name}")
    print(f"[signal bar] {asof.date()}  rows {len(snap):,}")

    snap["score"] = booster.predict(snap[FEATURES])

    # prev close, for the opening-gap filter
    prev = (df.loc[df["date"] < asof, ["code", "date", "close"]]
            .sort_values(["code", "date"]).groupby("code").tail(1)
            .rename(columns={"close": "prev_close"})[["code", "prev_close"]])
    snap = snap.merge(prev, on="code", how="left")

    cfg = SelectConfig(top_n=1)
    snap = snap[tradable(snap, cfg)].copy()
    snap["pct"] = snap["score"].rank(pct=True)

    names = pd.read_parquet(args.universe, columns=["code", "name"])
    names["code"] = names["code"].astype(str).str.zfill(6)
    snap["code"] = snap["code"].astype(str).str.zfill(6)
    snap = snap.merge(names, on="code", how="left")
    snap["板块"] = snap["board"].map(BOARD).fillna("其他")

    # ST hard block: st_proxy misses GEM/STAR names and band has no 5% tier,
    # so the name table is the only reliable exclusion available.
    is_st = snap["name"].fillna("").str.upper().str.contains("ST")
    n_st = int(is_st.sum())
    snap = snap[~is_st].copy()
    sn_st = snap["name"].fillna("").str.contains("退")
    n_del = int(sn_st.sum())
    snap = snap[~sn_st].copy()
    print(f"[exclude] ST {n_st} 只, 退市整理 {n_del} 只")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cols = ["code", "name", "板块", "close", "prev_close", "ret_1", "score", "pct", "amount"]
    snap.sort_values("score", ascending=False).to_csv(
        out / "scores_latest.csv", index=False, encoding="utf-8-sig")

    # 排雷：模型最不看好的 20%，这批票次日收益显著为负，任何情况下不碰
    bl = snap[snap["pct"] < 0.2].sort_values("score")
    bl[cols].to_csv(out / "blacklist_latest.csv", index=False, encoding="utf-8-sig")

    top = snap.nlargest(args.top, "score")[cols].copy()
    top["建议"] = "D 日 09:30 开盘买入"
    if len(top):
        top.iloc[0, top.columns.get_loc("建议")] = "首选（若开盘涨幅 ≤ 1.5%）"

    # next trading day: use the dataset's own calendar; fall back to weekdays for
    # dates beyond the data (holidays must be checked manually).
    known = np.sort(df["date"].unique())
    later = known[known > np.datetime64(asof)]
    if len(later):
        nxt = pd.Timestamp(later[0])
        cal_note = "按本地交易日历"
    else:
        nxt = asof + pd.offsets.BDay(1)
        cal_note = "本地数据未覆盖，按工作日推算（需核对节假日）"
    exit_d = nxt + pd.offsets.BDay(1)
    print(f"\n{'='*78}")
    print(f"交易信号：{asof.date()} 收盘后生成 → {nxt.date()} 开盘执行")
    print(f"{'='*78}")
    print(f"信号日 {asof.date()}（{cal_note}）→ 下一交易日 {nxt.date()}")
    print(f"买入 {nxt.date()} 09:30 集合竞价，卖出 {exit_d.date()} 09:30 集合竞价")
    print(f"过滤：开盘涨幅 > {GAP_MAX:.1%} 当日放弃（回测验证期增量 +4bp，不作核心）")
    print(f"股票池：{len(snap):,} 只（可交易）  排雷黑名单：{len(bl):,} 只\n")
    show = top.copy()
    show["close"] = show["close"].round(2)
    show["prev_close"] = show["prev_close"].round(2)
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if len(top):
        b = top.iloc[0]
        print(f"\n首选：{b['code']} {b['name']}（{b['板块']}）  昨收 {b['prev_close']:.2f}")
        print(f"  开盘涨幅 ≤ {GAP_MAX:.1%} → 昨收 × {1+GAP_MAX:.3f} = {b['prev_close']*(1+GAP_MAX):.2f} 以下才买")
        print(f"  若开盘价高于该价位，放弃，次日重新出信号")

    # forward ledger: one row per signal, filled in after execution
    ledger = out / "forward_ledger.csv"
    row = pd.DataFrame([{
        "signal_bar": asof.date().isoformat(),
        "trade_date": nxt.date().isoformat(),
        "code": top.iloc[0]["code"] if len(top) else "",
        "name": top.iloc[0]["name"] if len(top) else "",
        "board": top.iloc[0]["板块"] if len(top) else "",
        "score": round(float(top.iloc[0]["score"]), 5) if len(top) else np.nan,
        "prev_close": float(top.iloc[0]["prev_close"]) if len(top) else np.nan,
        "gap_cap": round(float(top.iloc[0]["prev_close"]) * (1 + GAP_MAX), 2) if len(top) else np.nan,
        "entry_price": np.nan, "exit_date": "", "exit_price": np.nan,
        "gross_ret": np.nan, "net_ret": np.nan, "traded": "",
        "logged_at": datetime.now().isoformat(timespec="seconds"),
    }])
    if ledger.exists():
        old = pd.read_csv(ledger, dtype={"code": str})
        old = old[old["signal_bar"] != asof.date().isoformat()]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(ledger, index=False, encoding="utf-8-sig")
    print(f"\n[ledger] {ledger}  ({len(row)} signals recorded)")

    print(f"\n[files] {out/'scores_latest.csv'}")
    print(f"        {out/'blacklist_latest.csv'}")


if __name__ == "__main__":
    main()
