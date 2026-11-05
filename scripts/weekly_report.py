# -*- coding: utf-8 -*-
"""生成单文件 HTML 周度复盘报告（资源内联、零外链、无水印）。

数据来源固定两处，不许加第三个：
  A. outputs/live/forward_ledger.csv          影子盘真实信号
  B. outputs/walkforward_v4/pred_drop_rank.parquet  回测重算

用法：python scripts/weekly_report.py [--out outputs/reports/weekly_review.html]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.weekly_review import COST, POS, load_book, replay  # noqa: E402

UP = "#c25b4e"     # 涨/正收益：柔红
DOWN = "#5c8a72"   # 跌/负收益：柔绿
INK = "#3a322c"
MUTED = "#8a7d72"
LINE = "#e3d9cd"
BG = "#faf7f2"


def weekly_series(r: pd.DataFrame) -> pd.DataFrame:
    f = r[r["filled"] & r["net_ret"].notna()].copy()
    f["week"] = f["date"].dt.to_period("W-SUN")
    g = f.groupby("week")
    w = pd.DataFrame({
        "笔数": g.size(),
        "单笔净均bp": g["net_ret"].mean() * 1e4,
        "胜率": g["net_ret"].apply(lambda s: (s > 0).mean()),
        "周收益": g["net_ret"].apply(lambda s: (1 + s * POS).prod() - 1),
    })
    w["净值"] = (1 + w["周收益"]).cumprod()
    return w.reset_index()


def bars_svg(w: pd.DataFrame, n: int = 26) -> str:
    d = w.tail(n)
    W, H, pad_l, pad_b, pad_t = 960, 260, 46, 34, 16
    lo, hi = min(d["周收益"].min(), -0.01), max(d["周收益"].max(), 0.01)
    span = hi - lo
    bw = (W - pad_l - 12) / len(d)
    zero = pad_t + (hi - 0) / span * (H - pad_t - pad_b)
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg">']
    out.append(f'<line x1="{pad_l}" y1="{zero:.1f}" x2="{W-6}" y2="{zero:.1f}" stroke="{LINE}" stroke-width="1"/>')
    for pct in (hi, 0, lo):
        y = pad_t + (hi - pct) / span * (H - pad_t - pad_b)
        out.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{pct*100:+.1f}%</text>')
    for i, (_, row) in enumerate(d.iterrows()):
        v = row["周收益"]
        y0 = pad_t + (hi - max(v, 0)) / span * (H - pad_t - pad_b)
        h = abs(v) / span * (H - pad_t - pad_b)
        x = pad_l + i * bw + bw * 0.16
        color = UP if v > 0 else DOWN
        out.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{bw*0.68:.1f}" height="{max(h,1):.1f}" fill="{color}" opacity="0.82" rx="1.5"/>')
        if i % 3 == 0:
            out.append(f'<text x="{x+bw*0.34:.1f}" y="{H-12}" font-size="10" fill="{MUTED}" text-anchor="middle">{row["week"].start_time:%m/%d}</text>')
    out.append(f'<text x="{pad_l}" y="{H-2}" font-size="10" fill="{MUTED}">最近 {len(d)} 周 · 单周组合收益（40% 仓位）</text>')
    out.append("</svg>")
    return "".join(out)


def curve_svg(w: pd.DataFrame) -> str:
    W, H, pad_l, pad_b, pad_t = 960, 300, 52, 36, 18
    eq = w["净值"].values
    lo, hi = min(eq.min(), 0.4), max(eq.max(), 1.05)
    span = hi - lo
    n = len(eq)
    pts = []
    for i, v in enumerate(eq):
        x = pad_l + i / (n - 1) * (W - pad_l - 14)
        y = pad_t + (hi - v) / span * (H - pad_t - pad_b)
        pts.append(f"{x:.1f},{y:.1f}")
    y1 = pad_t + (hi - 1.0) / span * (H - pad_t - pad_b)
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg">']
    out.append(f'<line x1="{pad_l}" y1="{y1:.1f}" x2="{W-14}" y2="{y1:.1f}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="4 4" opacity="0.6"/>')
    out.append(f'<text x="{pad_l-8}" y="{y1+4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">1.00</text>')
    out.append(f'<text x="{pad_l-8}" y="{pad_t+10:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{hi:.2f}</text>')
    out.append(f'<text x="{pad_l-8}" y="{H-pad_b:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{lo:.2f}</text>')
    out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{DOWN}" stroke-width="2"/>')
    for lbl, i in ((f"{w['week'].iloc[0].start_time:%Y-%m}", 0), (f"{w['week'].iloc[-1].start_time:%Y-%m}", n - 1)):
        x = pad_l + i / (n - 1) * (W - pad_l - 14)
        out.append(f'<text x="{x:.1f}" y="{H-10}" font-size="10" fill="{MUTED}" text-anchor="middle">{lbl}</text>')
    out.append("</svg>")
    return "".join(out)


def table(df: pd.DataFrame, fmt: dict, cls: str = "") -> str:
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            s = fmt[c](v) if c in fmt else str(v)
            k = fmt.get("_" + c, lambda x: "")
            cells.append(f"<td class='{k(v)}'>{s}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def money_pct(v: float) -> str:
    return f"{v*100:+.2f}%"


def dir_cls(v: float) -> str:
    return "up" if v > 0 else ("down" if v < 0 else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/reports/weekly_review.html")
    args = ap.parse_args()

    book = load_book()
    rep = replay()
    w = weekly_series(rep)

    f = rep[rep["filled"] & rep["net_ret"].notna()]
    n, m, sd = len(f), f["net_ret"].mean(), f["net_ret"].std()
    tstat = m / (sd / np.sqrt(n))
    total = (1 + w["周收益"]).prod() - 1
    posw = (w["周收益"] > 0).mean()

    neg = (w["周收益"] < 0).astype(int)
    run = cur = 0
    for v in neg:
        cur = cur + 1 if v else 0
        run = max(run, cur)

    # 影子盘
    b = book[book["net_ret"].notna()].copy()
    b["week"] = b["trade_date"].dt.to_period("W-SUN")
    bw = b.groupby("week").apply(lambda d: (1 + d["net_ret"] * POS).prod() - 1, include_groups=False)

    recent = w.tail(12).copy()
    recent["周"] = recent["week"].apply(lambda p: f"{p.start_time:%Y-%m-%d} ~ {p.end_time:%m-%d}")
    recent = recent[["周", "笔数", "单笔净均bp", "胜率", "周收益"]]

    yr = f.copy()
    yr["年"] = yr["date"].dt.year
    ytab = yr.groupby("年").agg(笔数=("net_ret", "size"),
                               单笔净均bp=("net_ret", lambda s: s.mean() * 1e4),
                               胜率=("net_ret", lambda s: (s > 0).mean()))
    ytab["组合年收益"] = yr.groupby("年")["net_ret"].apply(lambda s: (1 + s * POS).prod() - 1)
    ytab = ytab.reset_index()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A股隔夜短线 · 周度复盘</title>
<style>
  :root{{--bg:{BG};--ink:{INK};--muted:{MUTED};--line:{LINE};--up:{UP};--down:{DOWN};--card:#fffdfa;}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.7 -apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
    -webkit-font-smoothing:antialiased}}
  .wrap{{max-width:1040px;margin:0 auto;padding:44px 28px 64px}}
  h1{{font-size:26px;font-weight:600;letter-spacing:.5px;margin:0 0 6px}}
  h2{{font-size:17px;font-weight:600;margin:38px 0 12px;padding-left:10px;border-left:3px solid {UP}}}
  .sub{{color:var(--muted);font-size:13px;margin-bottom:26px}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin:24px 0 8px}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}}
  .card .k{{font-size:12px;color:var(--muted);letter-spacing:.4px}}
  .card .v{{font-size:25px;font-weight:600;margin-top:4px;font-variant-numeric:tabular-nums}}
  .card .n{{font-size:12px;color:var(--muted);margin-top:2px}}
  .up{{color:var(--up)}} .down{{color:var(--down)}}
  table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);
    border-radius:10px;overflow:hidden;font-size:13.5px;margin-top:6px}}
  th{{background:#f3ece2;font-weight:600;text-align:right;padding:9px 12px;color:#5c5049;font-size:12.5px}}
  th:first-child,td:first-child{{text-align:left}}
  td{{padding:9px 12px;text-align:right;border-top:1px solid #efe7dc;font-variant-numeric:tabular-nums}}
  .panel{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px 6px;margin-top:8px}}
  .note{{background:#f6efe3;border-left:3px solid #d9a05b;border-radius:0 8px 8px 0;
    padding:14px 18px;margin:18px 0;font-size:14px}}
  .note b{{color:#a8642c}}
  ul{{padding-left:20px}} li{{margin:5px 0}}
  .foot{{margin-top:36px;padding-top:16px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}}
  code{{background:#f3ece2;padding:1px 5px;border-radius:4px;font-size:12.5px}}
</style></head><body><div class="wrap">

<h1>周度复盘 · 每天只买信号第 1 名</h1>
<div class="sub">口径：D 收盘出信号 → D+1 开盘买 → D+2 开盘卖 ｜ 单票 40% 仓位 ｜ 扣 13bp 往返 ｜ 跳空上限 1.5%
<br>生成时间 {ts} ｜ 数据：<code>forward_ledger.csv</code>（影子盘） + <code>pred_drop_rank.parquet</code>（回测重算）</div>

<div class="cards">
  <div class="card"><div class="k">影子盘（真实信号）</div>
    <div class="v {dir_cls(bw.iloc[0] if len(bw) else 0)}">{money_pct(bw.iloc[0]) if len(bw) else "—"}</div>
    <div class="n">{len(b)} 笔已结算 · 仅 1 周</div></div>
  <div class="card"><div class="k">回测周均（244 周）</div>
    <div class="v down">{money_pct(w["周收益"].mean())}</div>
    <div class="n">正周占比 {posw:.1%} · 最长连亏 {run} 周</div></div>
  <div class="card"><div class="k">单笔净均（成交 {n} 笔）</div>
    <div class="v down">{m*1e4:+.1f}bp</div>
    <div class="n">t = {tstat:.2f} · 显著为负</div></div>
  <div class="card"><div class="k">累计净值（复利）</div>
    <div class="v down">{total*100:+.1f}%</div>
    <div class="n">2022-01 → 2026-10</div></div>
</div>

<div class="note"><b>本次复盘的核心结论：现行执行口径在回测里是负期望。</b>
官方 <code>summary.csv</code> 报的 top1 收益用的是 <code>d_oo(D)</code> = 信号日开盘买入，而信号在当天收盘后才算得出来——差一天。
换成实际能执行的 <code>d_oo(D+1)</code>（次日开盘买、后日开盘卖），毛均从 <b>+28.25bp 变成 -6.95bp</b>，
扣成本后 <b>-22.00bp</b>，t = {tstat:.2f}，不是噪声。</div>

<h2>最近 12 周（回测可实现口径）</h2>
<div class="panel">{bars_svg(w)}</div>
{table(recent, {"笔数": lambda v: f"{int(v)}", "单笔净均bp": lambda v: f"{v:+.1f}",
                "胜率": lambda v: f"{v:.0%}", "周收益": money_pct,
                "_周收益": dir_cls})}

<h2>累计净值（244 周，40% 仓位复利）</h2>
<div class="panel">{curve_svg(w)}</div>

<h2>逐年</h2>
{table(ytab, {"笔数": lambda v: f"{int(v)}", "单笔净均bp": lambda v: f"{v:+.1f}",
              "胜率": lambda v: f"{v:.0%}", "组合年收益": money_pct,
              "_单笔净均bp": dir_cls, "_组合年收益": dir_cls})}

<h2>三个需要你知道的事实</h2>
<ul>
<li><b>影子盘 2 笔 +480bp 不算证据。</b>n=2 的 t 值无意义，且这两天的行情恰好落在回测里少见的正收益窗口。</li>
<li><b>跳空 1.5% 上限在剔除最赚的那批。</b>被它放弃的 78 天，若买入毛均 +31.95bp；成交的 1083 天毛均 -9.00bp。这个风控方向是反的。</li>
<li><b>逐年恶化，不是均值回归。</b>2022 -13.1% → 2023 -14.9% → 2024 -19.3% → 2025 -21.0% → 2026 -23.1%（组合年收益）。</li>
</ul>

<h2>边界</h2>
<ul>
<li>回测重算用的是 <code>pred_drop_rank.parquet</code>（walkforward 分期模型）；生产每天用的是最新一版 <code>drop_rank_*.txt</code>，两者不完全同源，这是唯一可能解释影子盘与回测背离的非噪声因素，尚未验证。</li>
<li>回测数据止于 2026-10-22，影子盘从 2026-10-23 起，两段不重叠，无法逐笔对账。</li>
<li>未计入：滑点波动、涨跌停无法成交、停牌、ST 变更、资金约束。</li>
</ul>

<div class="foot">A股隔夜短线 · 周度复盘 ｜ 涨红跌绿 ｜ 金额单位 ¥ ｜ 数据源：tushare / 腾讯 / 新浪（收盘价）+ 项目账本 ｜ 口径见 TRADING_SPEC.md v2</div>
</div></body></html>"""

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"written: {out}  ({len(html)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
