# -*- coding: utf-8 -*-
"""按周统计"每天只买信号第 1 名"的收益。

两个口径分开算，不许混：

  A. 影子盘（唯一真实口径）—— outputs/live/forward_ledger.csv 里已成交的行，
     收益直接读账本的 net_ret（settle_ledger 已扣 13bp）。
  B. 回测重算 —— outputs/walkforward_v4/pred_drop_rank.parquet，按生产规则复现：
     tradable() 过滤 + 板块 0/1/2 + 非 ST + 每日 score 第 1 名 +
     开盘涨幅 >1.5% 当日放弃 + 扣 13bp。
     这一层用来回答"周维度长什么样"，不能替代 A。

组合口径：单票 40% 仓位（T1_POSITION_PCT），隔夜只持一天，同一周内不重叠，
故周组合收益 = 该周各笔 net_ret × 0.40 之和。

用法：python scripts/weekly_review.py [--weeks 12]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t1overnight.backtest import SelectConfig, tradable  # noqa: E402

COST = 0.0013      # 与 settle_ledger.COST_ROUND_TRIP 一致
GAP = 0.015        # 与 config.env T1_GAP_MAX 一致
POS = 0.40         # 与 config.env T1_POSITION_PCT 一致
BOARDS = (0, 1, 2)


def load_book() -> pd.DataFrame:
    p = ROOT / "outputs/live/forward_ledger.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p, dtype={"code": str}, encoding="utf-8-sig")
    d["traded"] = d.get("traded", "").fillna("").astype(str)
    for c in ("gross_ret", "net_ret", "entry_price", "exit_price"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    return d


def replay() -> pd.DataFrame:
    """按生产规则复现。收益用项目自带 fwd 字段计算，避免二次 merge 引入偏差。

    两套口径必须分开看：
      d_oo_D  官方 v4_eval 口径 = open(D)→open(D+1)，用了信号日开盘，
              而信号在 D 收盘后才有 —— 不可实现，只能当上界参考。
      d_oo_D1 可实现口径 = open(D+1)→open(D+2)，即生产实际执行的
              「次日开盘买、后日开盘卖」。
    """
    cols = ["code", "date", "open", "close", "fwd_cc", "fwd_co",
            "ret_1", "amount", "volume", "band", "st_proxy", "board", "is_st", "score"]
    p = pd.read_parquet(ROOT / "outputs/walkforward_v4/pred_drop_rank.parquet", columns=cols)
    p["code"] = p["code"].astype(str).str.zfill(6)
    p["date"] = pd.to_datetime(p["date"])

    m = tradable(p, SelectConfig(top_n=1)) & (~p["is_st"].fillna(False)) & p["board"].isin(BOARDS)
    top = (p[m].sort_values(["date", "score"], ascending=[True, False])
                .groupby("date", sort=True).head(1).copy())
    top["d_oc"] = top["close"] / top["open"] - 1
    top["d_oo_D"] = (1 + top["fwd_co"]) * (1 + top["d_oc"]) - 1

    dates = np.sort(p["date"].unique())
    nxt1 = dict(zip(dates[:-1], dates[1:]))
    top["d1"] = top["date"].map(nxt1)
    ref = p[["code", "date", "fwd_cc", "fwd_co", "open", "close"]].rename(
        columns={"date": "d1", "fwd_cc": "cc1", "fwd_co": "co1", "open": "o1", "close": "c1"})
    top = top.merge(ref, on=["code", "d1"], how="left")

    top["d_oo_D1"] = (1 + top["co1"]) * (1 + top["c1"] / top["o1"] - 1) - 1
    top["gap"] = top["o1"] / top["close"] - 1
    top["filled"] = top["gap"] <= GAP

    top["gross_ret"] = top["d_oo_D1"]                    # 成交与放弃都保留原始收益
    top["net_ret"] = np.where(top["filled"], top["d_oo_D1"] - COST, np.nan)
    return top


def weekly(df: pd.DataFrame, date_col: str, ret_col: str, label: str) -> pd.DataFrame:
    d = df[df[ret_col].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["week"] = d[date_col].dt.to_period("W-SUN")
    g = d.groupby("week")
    out = pd.DataFrame({
        "周笔数": g.size(),
        "单笔净均bp": g[ret_col].mean() * 1e4,
        "胜率": g[ret_col].apply(lambda s: (s > 0).mean()),
        "周组合收益": g[ret_col].apply(lambda s: (1 + s * POS).prod() - 1),
        "周毛收益": g[ret_col].apply(lambda s: (1 + (s + COST) * POS).prod() - 1),
        "最差单笔bp": g[ret_col].min() * 1e4,
        "最好单笔bp": g[ret_col].max() * 1e4,
    })
    out["口径"] = label
    return out.reset_index()


def summarize(w: pd.DataFrame, label: str) -> dict:
    r = w["周组合收益"]
    return {
        "口径": label,
        "周数": len(w),
        "周均": r.mean(),
        "周中位": r.median(),
        "正周占比": (r > 0).mean(),
        "最好周": r.max(),
        "最差周": r.min(),
        "周波动": r.std(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=12)
    args = ap.parse_args()

    book = load_book()
    rep = replay()

    bw = weekly(book, "trade_date", "net_ret", "影子盘账本")
    rw = weekly(rep, "date", "net_ret", "回测重算")

    print("=" * 78)
    print("A. 影子盘账本（真实信号，唯一收益口径）")
    print("=" * 78)
    if bw.empty:
        print("  尚无已结算交易")
    else:
        print(bw[["week", "周笔数", "单笔净均bp", "胜率", "周组合收益", "最差单笔bp"]].to_string(index=False))
        print()
        print(pd.DataFrame([summarize(bw, "影子盘")]).to_string(index=False))

    print()
    print("=" * 78)
    print("B. 回测重算（2022-01-04 起，每日 score 第 1 名，1.5% 跳空上限，扣 13bp）")
    print("=" * 78)
    n_days = len(rep)
    n_fill = int(rep["filled"].sum())
    print(f"  信号日 {n_days} 天 · 实际成交 {n_fill} 天 · 跳空放弃 {n_days - n_fill} 天"
          f"（{1 - n_fill / n_days:.1%}）")
    print(f"  [官方口径 d_oo(D)]   毛均 {rep['d_oo_D'].mean() * 1e4:+.2f}bp · "
          f"命中率 {(rep['d_oo_D'] > 0).mean():.1%}  ← 用信号日开盘价，不可实现")
    print(f"  [可实现 d_oo(D+1)]   毛均 {rep['d_oo_D1'].mean() * 1e4:+.2f}bp · "
          f"命中率 {(rep['d_oo_D1'] > 0).mean():.1%}  ← 生产实际执行口径")
    print(f"  成交后：毛均 {rep.loc[rep['filled'], 'd_oo_D1'].mean() * 1e4:+.2f}bp · "
          f"净均 {rep['net_ret'].mean() * 1e4:+.2f}bp · 胜率 {(rep['net_ret'] > 0).mean():.1%}")
    print(f"  被跳空上限放弃的 {n_days - n_fill} 天，若买入毛均 "
          f"{rep.loc[~rep['filled'], 'd_oo_D1'].mean() * 1e4:+.2f}bp")
    print()
    print(pd.DataFrame([summarize(rw, "回测重算")]).to_string(index=False))
    print()
    print(f"  最近 {args.weeks} 周：")
    tail = rw.tail(args.weeks)
    print(tail[["week", "周笔数", "单笔净均bp", "胜率", "周组合收益"]].to_string(index=False))

    # 年度视角：周维度之外，看是否某一年整体失效
    rep2 = rep[rep["net_ret"].notna()].copy()
    rep2["年"] = rep2["date"].dt.year
    print()
    print("  年度：")
    yd = rep2.groupby("年").agg(笔数=("net_ret", "size"),
                              单笔净均bp=("net_ret", lambda s: s.mean() * 1e4),
                              胜率=("net_ret", lambda s: (s > 0).mean()))
    yd["组合年收益"] = rep2.groupby("年")["net_ret"].apply(lambda s: (1 + s * POS).prod() - 1)
    print(yd.to_string())


if __name__ == "__main__":
    main()
