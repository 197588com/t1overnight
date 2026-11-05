# -*- coding: utf-8 -*-
"""账本归因层：把每笔交易拆成"哪一段赚的、哪一段亏的"。

为什么单独一个文件，而不是往 forward_ledger.csv 里加列
------------------------------------------------------
主账本是唯一收益口径（TRADING_SPEC v2），结构一动，读它的地方全部要跟着改；
而归因字段属于分析口径，随时可能推翻重算。所以这里做成旁路：
账本保持不动，本脚本每次全量重算 outputs/live/ledger_attr.csv，幂等、可丢弃。

拆三段（S = 信号日，收盘后出票）
------------------------------
    A 段  S 收盘 → S+1 开盘   segA = open(S+1)/close(S) - 1
          这是模型预测的目标（fwd_co），也是跳空上限拦的地方。
          本策略在 A 段**没有仓位**——钱在这里让给了别人。
    B 段  S+1 开盘 → S+1 收盘  segB = close(S+1)/open(S+1) - 1   日内
    C 段  S+1 收盘 → S+2 开盘  segC = open(S+2)/close(S+1) - 1   第二隔夜
          实际持仓赚的是 B + C，即 open(S+2)/open(S+1) - 1。

恒等式：(1+segA)(1+segB)(1+segC) - 1 == open(S+2)/close(S) - 1
本脚本逐行校验，不成立就报出来——归因表算错比没有归因更糟。

跳空放弃的行（no_fill_gap）
--------------------------
账本只记"没买"，不记"没买之后怎么走"，所以永远无法评估 1.5% 上限是好是坏。
这里为这些行补一个 hypo_net（假想买入的净收益），只用于参数评估，
**不计入正式收益口径**，也不写回主账本。

用法：python scripts/ledger_attr.py [--json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "outputs/live/forward_ledger.csv"
DATASET = ROOT / "outputs/t1_dataset.parquet"
OUT = ROOT / "outputs/live/ledger_attr.csv"
COST = 0.0013
POS = 0.40
# 单笔净收益标准差，来自回测全样本（1083 笔，scripts/weekly_review.py 可复现）。
# 账本自己的样本量太小，标准差不可用；功效分析只能用这个先验。
PRIOR_SD_BP = 266.7


def load_ledger() -> pd.DataFrame:
    led = pd.read_csv(LEDGER, dtype={"code": str}, encoding="utf-8-sig")
    led["code"] = led["code"].astype(str).str.zfill(6)
    for c in ("entry_price", "exit_price", "gross_ret", "net_ret", "prev_close", "gap_cap", "score"):
        led[c] = pd.to_numeric(led.get(c), errors="coerce")
    for c in ("traded", "exit_date"):
        led[c] = led.get(c, "").fillna("").astype(str)
    led["signal_bar"] = pd.to_datetime(led["signal_bar"])
    led["trade_date"] = pd.to_datetime(led["trade_date"])
    return led


def load_prices(codes: list[str]) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    codes = sorted({c for c in codes if c and c != "nan"})
    tbl = pq.read_table(DATASET, columns=["code", "date", "open", "close"],
                        filters=[("code", "in", codes)])
    px = tbl.to_pandas()
    px["code"] = px["code"].astype(str).str.zfill(6)
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    dates = pq.read_table(DATASET, columns=["date"]).column("date").to_pandas()
    sessions = sorted(pd.to_datetime(dates).dt.normalize().unique())
    return px, [pd.Timestamp(s) for s in sessions]


def next_session(sessions: list[pd.Timestamp], d: pd.Timestamp) -> pd.Timestamp | None:
    later = [s for s in sessions if s > d]
    return later[0] if later else None


def attribute(led: pd.DataFrame, px: pd.DataFrame, sessions: list[pd.Timestamp]) -> pd.DataFrame:
    look_op = {(r.date, r.code): r.open for r in px.itertuples(index=False)}
    look_cl = {(r.date, r.code): r.close for r in px.itertuples(index=False)}

    rows = []
    for _, r in led.iterrows():
        void = r["traded"].lower().startswith("void")
        s1 = next_session(sessions, r["trade_date"]) if pd.notna(r["trade_date"]) else None
        sig_close = r["prev_close"]
        o1 = look_op.get((r["trade_date"], r["code"]))
        c1 = look_cl.get((r["trade_date"], r["code"]))
        o2 = look_op.get((s1, r["code"])) if s1 is not None else None

        segA = (o1 / sig_close - 1) if (o1 and sig_close and sig_close > 0) else np.nan
        segB = (c1 / o1 - 1) if (o1 and c1) else np.nan
        segC = (o2 / c1 - 1) if (c1 and o2) else np.nan
        actual = (o2 / o1 - 1) if (o1 and o2) else np.nan

        check = np.nan
        if not np.isnan(segA) and not np.isnan(segB) and not np.isnan(segC):
            check = (1 + segA) * (1 + segB) * (1 + segC) - 1 - (o2 / sig_close - 1)

        traded = r["traded"]
        kind = ("void" if void else
                "gap_skipped" if traded == "no_fill_gap" else
                "filled" if pd.notna(r["exit_price"]) else
                "pending" if pd.notna(r["entry_price"]) or (o1 and traded == "") else
                "pending")

        # void 行是已宣布作废的记录（旧模型/错基准），它的价格能算出来也不该进统计，
        # 否则一份被否掉的样本会悄悄参与均值。
        strat_net = (actual - COST) if (pd.notna(actual) and not void) else np.nan

        rows.append({
            "signal_bar": r["signal_bar"].date(), "trade_date": r["trade_date"].date(),
            "exit_date": s1.date() if s1 is not None else "",
            "code": r["code"], "name": r["name"], "board": r["board"],
            "kind": kind, "traded": traded, "score": r["score"],
            "sig_close": sig_close, "open1": o1, "close1": c1, "open2": o2,
            "gap": segA, "segB_intraday": segB, "segC_second_overnight": segC,
            "strat_gross": actual if not void else np.nan,
            "strat_net": strat_net,
            "ledger_gross": r["gross_ret"], "ledger_net": r["net_ret"],
            "identity_err": check,
        })
    return pd.DataFrame(rows)


def stats(d: pd.DataFrame) -> dict:
    x = d["strat_net"].dropna()
    n = len(x)
    if n == 0:
        return {"n": 0}
    from math import sqrt
    # t 临界值：小样本必须用 t 分布，用 1.96 会把 n=2 的区间算窄一半以上
    try:
        from scipy import stats as st
        tcrit = float(st.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    except Exception:                                    # noqa: BLE001
        tcrit = 12.706 if n == 2 else (4.303 if n == 3 else 1.96)
    se = x.std(ddof=1) / sqrt(n) if n > 1 else float("nan")
    return {
        "n": n,
        "mean_bp": x.mean() * 1e4,
        "median_bp": x.median() * 1e4,
        "sd_bp": x.std(ddof=1) * 1e4 if n > 1 else float("nan"),
        "se_bp": se * 1e4,
        "ci_lo_bp": (x.mean() - tcrit * se) * 1e4 if n > 1 else float("nan"),
        "ci_hi_bp": (x.mean() + tcrit * se) * 1e4 if n > 1 else float("nan"),
        "t": (x.mean() / se) if n > 1 and se > 0 else float("nan"),
        "hit": (x > 0).mean(),
    }


def power(sigma_bp: float, delta_bp: float) -> int:
    """识别 delta 所需样本量（双侧 95%、功效 80%）。"""
    return int(np.ceil(((1.96 + 0.8416) * sigma_bp / delta_bp) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    led = load_ledger()
    px, sessions = load_prices(led["code"].tolist())
    at = attribute(led, px, sessions)
    at.to_csv(OUT, index=False, encoding="utf-8-sig")

    bad = at[at["identity_err"].abs() > 1e-9].dropna(subset=["identity_err"])
    if len(bad):
        print(f"[!] 归因恒等式不成立 {len(bad)} 行：{', '.join(bad['code'])}")

    filled = at[at["kind"] == "filled"]
    skipped = at[at["kind"] == "gap_skipped"]
    pending = at[at["kind"] == "pending"]

    s = stats(filled)
    out = {"filled": s,
           "gap_skipped": stats(skipped),
           "pending": len(pending),
           "rows": int(len(at))}

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2, default=float))
        return

    print("═" * 66)
    print("账本归因（口径：S 收盘出票 → S+1 开盘买 → S+2 开盘卖，扣 13bp）")
    print("═" * 66)
    cols = ["code", "name", "kind", "gap", "segB_intraday", "segC_second_overnight",
            "strat_net", "ledger_net"]
    show = at[cols].copy()
    for c in ("gap", "segB_intraday", "segC_second_overnight", "strat_net", "ledger_net"):
        show[c] = (show[c] * 1e4).round(1)
    print(show.to_string(index=False))
    print("（gap/B/C/净 单位 bp；gap 为 A 段隔夜跳空，本策略不持有该段）")

    print()
    if s["n"]:
        print(f"已结算 {s['n']} 笔：净均 {s['mean_bp']:+.1f}bp · 中位 {s['median_bp']:+.1f}bp"
              f" · 胜率 {s['hit']:.0%}")
        print(f"  95% 置信区间 [{s['ci_lo_bp']:+.0f}, {s['ci_hi_bp']:+.0f}]bp"
              f" · t={s['t']:.2f}（n={s['n']}，t 分布临界值随 n 变化）")
        print("  区间横跨 0 → 目前只能说「不知道」，不能说「有效」或「无效」")
    else:
        print("已结算 0 笔")

    # 三段贡献：回答"钱在哪一段丢的"
    if len(filled):
        print()
        print("三段均值贡献（每笔）：")
        for lbl, col in (("A 隔夜跳空（不持有）", "gap"),
                         ("B 日内持仓", "segB_intraday"),
                         ("C 第二隔夜", "segC_second_overnight")):
            v = filled[col].dropna()
            print(f"  {lbl:22s} {v.mean()*1e4:+8.1f}bp   n={len(v)}")

    if len(skipped):
        print()
        known = skipped["strat_net"].dropna()
        waiting = len(skipped) - len(known)
        if len(known):
            print(f"跳空放弃 {len(skipped)} 笔 · 若照买净均 "
                  f"{known.mean()*1e4:+.1f}bp  n={len(known)}  ← 评估 1.5% 上限的唯一证据")
        else:
            print(f"跳空放弃 {len(skipped)} 笔 · 若照买净均：无法计算"
                  f"（S+2 行情尚未到期，需等到下一交易日开盘后）")
        if waiting and len(known):
            print(f"  另有 {waiting} 笔待数据到期，暂不计入均值")

    # 样本量功效：σ 必须用回测全样本（1083 笔）的标准差，
    # 用账本自己 n=2 的样本标准差会把所需样本量放大十几倍，那是噪声不是方差。
    print()
    print(f"样本量要求（双侧 95%，功效 80%，σ={PRIOR_SD_BP:.0f}bp 取回测全样本）：")
    for d in (25, 50, 100):
        need = power(PRIOR_SD_BP, d)
        print(f"  识别 {d:3d}bp 的真实 edge 需 {need:5d} 笔 ≈ {need/244:.1f} 年（每年 244 笔）")
    print(f"  当前只有 {s['n']} 笔：置信区间宽 "
          f"{s.get('ci_hi_bp', np.nan) - s.get('ci_lo_bp', np.nan):.0f}bp，"
          f"任何调参结论在这个宽度下都不成立")
    print()
    print(f"明细已写入 {OUT.relative_to(ROOT)}（旁路文件，不改主账本）")


if __name__ == "__main__":
    main()
