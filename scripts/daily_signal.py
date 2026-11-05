# -*- coding: utf-8 -*-
"""Daily trade signal under the production rule (open-to-open, T+1 legal).

One naming convention for the whole project -- S is the signal bar:

    signal bar     S close (features lagged 1 bar, model already trained that way)
    entry          S+1 09:30 opening auction
    skip           if the opening gap (open / prev_close - 1) exceeds GAP_MAX
    exit           S+2 09:30 opening auction

``daily_signal.py``, ``settle_ledger.py`` and the ledger columns use exactly this
S/S+1/S+2 frame (``signal_bar`` / ``trade_date`` / ``exit_date``).  Earlier the
docstring here called the entry day "D", which read as one day earlier than the
same letter in settle_ledger and produced a void row bought on the wrong basis.

S+1 and S+2 come from the exchange calendar.  Weekday arithmetic (``BDay``) is
wrong twice a year: a signal on the last session before a holiday would name a
holiday as the buy day, and settle_ledger -- which only grades rows whose
trade_date exists in the dataset -- would then never grade that trade.

Why open-to-open: with the strict lag-1 model the score on row (code, S) is knowable
before S+1's opening auction. Waiting until S+1's close gives up the whole intraday leg
(+40.8bp in backtest) and keeps only the overnight leg (-18.5bp).

Usage:
    python scripts/daily_signal.py                  # newest bar in the dataset
    python scripts/daily_signal.py --asof 2026-10-23
    python scripts/daily_signal.py --top 5
"""
from __future__ import annotations

import argparse
import os
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

# Opening-gap cap.  Sourced from deploy/config.env (T1_GAP_MAX) so the number that
# gates the entry has one home; the literal here is only the fallback for a bare
# `python scripts/daily_signal.py` run.  Values must not drift between the two.
GAP_MAX = float(os.environ.get("T1_GAP_MAX", "0.015"))
# Cost is applied once, in scripts/settle_ledger.py (COST_ROUND_TRIP = 13bp).
# A second constant used to live here claiming 20bp "in the ledger's net column",
# which was simply false and would have made two files disagree about the same
# number.  Do not reintroduce one.
BOARD = {0: "沪主板", 1: "深主板", 2: "创业板", 3: "科创板", 4: "北交所"}


def newest_model(model_dir: Path, pattern: str) -> Path:
    ms = sorted(model_dir.glob(pattern))
    if not ms:
        raise FileNotFoundError(f"no {pattern} under {model_dir}")
    return ms[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="outputs/t1_dataset.parquet")
    # v4: ST rows removed from training and from the prediction universe
    ap.add_argument("--models", default="outputs/walkforward_v4/models")
    ap.add_argument("--pattern", default="drop_rank_*.txt")
    ap.add_argument("--universe",
                    default=os.environ.get("T1_UNIVERSE",
                                       os.path.expanduser("~/datapool/parquet/universe.parquet")))
    ap.add_argument("--out-dir", default="outputs/live")
    ap.add_argument("--asof", default=None, help="signal bar date; default = newest in dataset")
    ap.add_argument("--top", type=int, default=5)
    # default = the account's real permissions: main boards (no qualification)
    # + ChiNext (100k + 24m).  STAR board (688) needs 500k + 24m and is excluded.
    ap.add_argument("--boards", default="0,1,2",
                    help="permitted boards: 0 沪主板 1 深主板 2 创业板 3 科创板 4 北交所. "
                         "Main boards need no account qualification; ChiNext needs 100k+24m; "
                         "STAR needs 500k+24m.")
    ap.add_argument("--tier", default="", help="label written into the forward ledger")
    args = ap.parse_args()

    allowed = tuple(int(b) for b in str(args.boards).split(",") if str(b).strip() != "")

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

    model = newest_model(Path(args.models), args.pattern)
    booster = lgb.Booster(model_file=str(model))
    print(f"[model] {model.name}")
    print(f"[signal bar] {asof.date()}  rows {len(snap):,}")

    snap["score"] = booster.predict(snap[FEATURES])

    # Opening-gap benchmark.  The model is predicting open(D)/open(D-1) with
    # D = the bar *after* asof, and the features of row asof are already lag-1,
    # so asof IS D-1: the reference price must be close(asof), i.e. snap["close"].
    # Using close(asof-1) shifts the filter a full day -- for a name that moved
    # 4% on the signal bar the cap lands *below* the signal close and the day is
    # always skipped.
    snap["prev_close"] = snap["close"]

    cfg = SelectConfig(top_n=1)
    snap = snap[tradable(snap, cfg)].copy()
    n_all = len(snap)
    snap = snap[snap["board"].isin(allowed)].copy()
    if snap.empty:
        raise SystemExit(f"no tradable name on boards {allowed}")
    print(f"[boards] 允许 {allowed} → 池子 {n_all:,} → {len(snap):,}")
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
    top["建议"] = "次日 09:30 开盘买入"
    if len(top):
        top.iloc[0, top.columns.get_loc("建议")] = "首选（若开盘涨幅 ≤ 1.5%）"

    # S+1 / S+2 must come from the exchange calendar, not from weekday arithmetic
    # (see the module docstring).  The dataset can only answer this when replaying
    # a past bar with later sessions already loaded, so it is the fallback.
    known = np.sort(df["date"].unique())
    nxt = exit_d = None
    cal_note = ""
    try:
        from fetch_today import next_trade_day          # noqa: E402
        a = next_trade_day(asof.strftime("%Y%m%d"), 1)
        b = next_trade_day(asof.strftime("%Y%m%d"), 2)
        if a:
            nxt = pd.Timestamp(a)
            cal_note = "交易所日历"
        if b:
            exit_d = pd.Timestamp(b)
    except Exception as exc:                            # noqa: BLE001
        print(f"[cal] 日历查询失败（{type(exc).__name__}），退回本地推算：{exc}")

    if nxt is None:
        later = known[known > np.datetime64(asof)]
        if len(later):
            nxt = pd.Timestamp(later[0])
            cal_note = "按本地交易日历"
        else:
            nxt = asof + pd.offsets.BDay(1)
            cal_note = "本地数据未覆盖，按工作日推算（遇节假日会错，需人工核对）"
    if exit_d is None:
        after = known[known > np.datetime64(nxt)]
        exit_d = pd.Timestamp(after[0]) if len(after) else nxt + pd.offsets.BDay(1)
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

    # forward ledger: one row per (signal bar, permission tier), filled after execution
    ledger = out / "forward_ledger.csv"
    tier = args.tier or ",".join(map(str, allowed))
    row = pd.DataFrame([{
        "signal_bar": asof.date().isoformat(),
        "tier": tier,
        "trade_date": nxt.date().isoformat(),
        "code": top.iloc[0]["code"] if len(top) else "",
        "name": top.iloc[0]["name"] if len(top) else "",
        "board": top.iloc[0]["板块"] if len(top) else "",
        "score": round(float(top.iloc[0]["score"]), 5) if len(top) else np.nan,
        "prev_close": float(top.iloc[0]["prev_close"]) if len(top) else np.nan,
        "gap_cap": round(float(top.iloc[0]["prev_close"]) * (1 + GAP_MAX), 2) if len(top) else np.nan,
        "entry_price": np.nan, "exit_date": "", "exit_price": np.nan,
        "gross_ret": np.nan, "net_ret": np.nan,
        # an empty pick is a fact worth recording: without a marker it looks like
        # a trade that simply has not opened yet, and every count downstream
        # (settle / status board) treats it as pending forever
        "traded": "" if len(top) else "no_pick",
        "logged_at": datetime.now().isoformat(timespec="seconds"),
    }])
    if ledger.exists():
        old = pd.read_csv(ledger, dtype={"code": str})
        if "tier" not in old.columns:
            old["tier"] = "全市场"
        stay = ~((old["signal_bar"] == asof.date().isoformat()) & (old["tier"] == tier))
        row = pd.concat([old[stay], row], ignore_index=True)
    row.to_csv(ledger, index=False, encoding="utf-8-sig")
    print(f"\n[ledger] {ledger}  ({len(row)} signals recorded)")

    print(f"\n[files] {out/'scores_latest.csv'}")
    print(f"        {out/'blacklist_latest.csv'}")


if __name__ == "__main__":
    main()
