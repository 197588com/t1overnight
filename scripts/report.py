# -*- coding: utf-8 -*-
"""Render the single-file HTML report from backtest artefacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "report"

PALETTE = {
    "bg": "#f7f2eb", "card": "#fffdfa", "ink": "#332e2a", "muted": "#8b7d72",
    "line": "#e3d8cb", "up": "#b23a2e", "down": "#3f7a5e",
    "accent": "#b5643c", "accent2": "#7d8a6a",
}


def svg_lines(curves: dict, width: int = 980, height: int = 360) -> str:
    series = {k: v for k, v in curves.items() if v.get("equity")}
    if not series:
        return ""
    pad_l, pad_r, pad_t, pad_b = 64, 18, 16, 34
    all_y = [y for v in series.values() for y in v["equity"]]
    n = max(len(v["equity"]) for v in series.values())
    ymin, ymax = min(all_y) * 0.98, max(all_y) * 1.02
    if ymax <= ymin:
        ymax = ymin + 1
    colors = [PALETTE["accent"], PALETTE["accent2"], PALETTE["up"], PALETTE["muted"]]
    labels = {"rank_top1": "排序头 Top1（每日 1 票满仓）", "rank_top5": "排序头 Top5 等权",
              "prob_top1": "概率头 Top1", "baseline_random": "随机基线"}

    def xp(i):
        return pad_l + (width - pad_l - pad_r) * (i / max(n - 1, 1))

    def yp(y):
        return pad_t + (height - pad_t - pad_b) * (1 - (y - ymin) / (ymax - ymin))

    p = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,PingFang SC,sans-serif">']
    for g in range(5):
        y = ymin + (ymax - ymin) * g / 4
        p.append(f'<line x1="{pad_l}" y1="{yp(y):.1f}" x2="{width-pad_r}" y2="{yp(y):.1f}" stroke="{PALETTE["line"]}"/>')
        p.append(f'<text x="{pad_l-8}" y="{yp(y)+4:.1f}" font-size="11" fill="{PALETTE["muted"]}" text-anchor="end">{y/1000:.0f}k</text>')
    if ymin <= 100000 <= ymax:
        p.append(f'<line x1="{pad_l}" y1="{yp(100000):.1f}" x2="{width-pad_r}" y2="{yp(100000):.1f}" stroke="{PALETTE["muted"]}" stroke-dasharray="4 4"/>')
    for idx, (k, v) in enumerate(series.items()):
        pts = " ".join(f"{xp(i):.1f},{yp(y):.1f}" for i, y in enumerate(v["equity"]))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{colors[idx % len(colors)]}" stroke-width="1.8"/>')
        p.append(f'<text x="{pad_l}" y="{height-12}" font-size="11" fill="{PALETTE["muted"]}">{v["date"][0]}</text>')
        p.append(f'<text x="{width-pad_r}" y="{height-12}" font-size="11" fill="{PALETTE["muted"]}" text-anchor="end">{v["date"][-1]}</text>')
        ly = pad_t + 14 + idx * 18
        p.append(f'<rect x="{pad_l+10}" y="{ly-8}" width="10" height="3" fill="{colors[idx % len(colors)]}"/>')
        p.append(f'<text x="{pad_l+26}" y="{ly-3}" font-size="11.5" fill="{PALETTE["ink"]}">{labels.get(k, k)}</text>')
    p.append("</svg>")
    return "".join(p)


def svg_bars(records: list[dict], xkey: str, ykey: str, width: int = 980, height: int = 280,
             ylabel: str = "", pctfmt: bool = True) -> str:
    if not records:
        return ""
    pad_l, pad_r, pad_t, pad_b = 64, 16, 16, 44
    ys = [r[ykey] for r in records]
    ymin, ymax = min(0.0, min(ys)), max(0.0, max(ys))
    span = (ymax - ymin) or 1
    ymin -= span * 0.14
    ymax += span * 0.16
    step = (width - pad_l - pad_r) / len(records)
    bw = step * 0.58

    def yp(v):
        return pad_t + (height - pad_t - pad_b) * (1 - (v - ymin) / (ymax - ymin))

    p = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,PingFang SC,sans-serif">']
    for g in range(5):
        v = ymin + (ymax - ymin) * g / 4
        p.append(f'<line x1="{pad_l}" y1="{yp(v):.1f}" x2="{width-pad_r}" y2="{yp(v):.1f}" stroke="{PALETTE["line"]}"/>')
        lab = f"{v*100:.2f}%" if pctfmt else f"{v:.1f}"
        p.append(f'<text x="{pad_l-8}" y="{yp(v)+4:.1f}" font-size="10.5" fill="{PALETTE["muted"]}" text-anchor="end">{lab}</text>')
    p.append(f'<line x1="{pad_l}" y1="{yp(0):.1f}" x2="{width-pad_r}" y2="{yp(0):.1f}" stroke="{PALETTE["muted"]}" stroke-width="1.2"/>')
    for i, r in enumerate(records):
        v = r[ykey]
        cx = pad_l + step * (i + 0.5)
        y0, y1 = (yp(v), yp(0)) if v >= 0 else (yp(0), yp(v))
        color = PALETTE["up"] if v >= 0 else PALETTE["down"]
        p.append(f'<rect x="{cx-bw/2:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{max(abs(y1-y0),0.6):.1f}" fill="{color}" opacity="0.87" rx="1.5"/>')
        p.append(f'<text x="{cx:.1f}" y="{height-25}" font-size="10.5" fill="{PALETTE["muted"]}" text-anchor="middle">{r[xkey]}</text>')
        lab = f"{v*100:.2f}" if pctfmt else f"{v:.1f}"
        ty = (y0 - 5) if v >= 0 else (y1 + 13)
        p.append(f'<text x="{cx:.1f}" y="{ty:.1f}" font-size="9.8" fill="{PALETTE["ink"]}" text-anchor="middle">{lab}</text>')
    if ylabel:
        p.append(f'<text x="{pad_l}" y="{height-6}" font-size="10.5" fill="{PALETTE["muted"]}">{ylabel}</text>')
    p.append("</svg>")
    return "".join(p)


def table(df: pd.DataFrame, cols: list[str], headers: list[str], highlight: str | None = None) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = []
    for _, r in df.iterrows():
        cls = ' class="hl"' if highlight and str(r[cols[0]]) == highlight else ""
        body.append(f"<tr{cls}>" + "".join(f"<td>{r[c]}</td>" for c in cols) + "</tr>")
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def avoid_effect(pred_path: Path, dataset: Path) -> dict:
    """How much does dropping the model's worst-ranked names help a neutral holder?"""
    from t1overnight.backtest import SelectConfig, tradable

    p = pd.read_parquet(pred_path)
    p["date"] = pd.to_datetime(p["date"])
    ex = pd.read_parquet(dataset, columns=["code", "date", "amt_ratio", "ret_20", "ma_dev_20"])
    ex["date"] = pd.to_datetime(ex["date"])
    p = p.merge(ex, on=["code", "date"], how="left")
    p = p[tradable(p, SelectConfig(top_n=1))].copy()
    p["pct"] = p.groupby("date")["score"].rank(pct=True)
    res = {"all": (float(p["fwd_cc"].mean()), float((p["fwd_cc"] > 0).mean()))}
    for cut in (0.1, 0.2, 0.3):
        k = p[p["pct"] >= cut]
        res[cut] = (float(k["fwd_cc"].mean()), float((k["fwd_cc"] > 0).mean()))
    return res


def build() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    rb = json.loads((ROOT / "outputs/backtest_v2/report_data.json").read_text())
    rbB = json.loads((ROOT / "outputs/backtest_v2_lag1/report_data.json").read_text())
    hp = json.loads((ROOT / "outputs/hold/hold_period.json").read_text())

    s = {x["variant"]: x for x in rb["summary"]}
    be = {x["variant"]: x for x in rb["breakeven"]}
    sB = {x["variant"]: x for x in rbB["summary"]}
    beB = {x["variant"]: x for x in rbB["breakeven"]}
    top1, rnd = s["rank_top1"], s["baseline_random"]
    b1 = be["rank_top1"]

    summ = pd.DataFrame(rb["summary"])
    summ["hit_s"] = (summ["hit_rate"] * 100).round(2)
    summ["mean_s"] = (summ["mean_daily"] * 100).round(3)
    summ["ann_s"] = (summ["annual_return"] * 100).round(1)
    summ["dd_s"] = (summ["max_drawdown"] * 100).round(1)
    summ["sharpe_s"] = summ["sharpe"].round(2)
    summ["t_s"] = summ["t_stat"].round(2)
    name_map = {
        "rank_top1": "排序头 Top1（每日1票）", "rank_top3": "排序头 Top3 等权",
        "rank_top5": "排序头 Top5 等权", "rank_top10": "排序头 Top10 等权",
        "prob_top1": "概率头 Top1", "prob_top3": "概率头 Top3", "prob_top5": "概率头 Top5",
        "prob_top10": "概率头 Top10", "rank_top1_prob50": "Top1 + 概率≥0.50",
        "rank_top1_prob52": "Top1 + 概率≥0.52", "rank_top1_prob55": "Top1 + 概率≥0.55",
        "rank_top1_timed": "Top1 + 强度择时", "baseline_random": "基线·随机",
        "baseline_highest_amount": "基线·成交额最大", "baseline_amt_ratio": "基线·放量最猛",
    }
    summ["label"] = summ["variant"].map(name_map).fillna(summ["variant"])

    dec = pd.DataFrame(rb["decile"])
    dec_rows = [{"d": f"D{int(r['decile'])+1}", "m": float(r["mean"])} for _, r in dec.iterrows()]
    decB = pd.DataFrame(rbB["decile"])
    decB_rows = [{"d": f"D{int(r['decile'])+1}", "m": float(r["mean"])} for _, r in decB.iterrows()]

    cs_rows = [{"c": f"{r['cost_rt']*100:.2f}%", "a": float(r["annual_return"])} for r in rb["cost_sensitivity"]]

    hp_df = pd.DataFrame([{
        "k": f"{r['k']} 日", "trades": r["trades"], "hit": round(r["hit"] * 100, 1),
        "gross": round(r["gross_mean"] * 100, 3), "net": round(r["net_mean"] * 100, 3),
        "cost": round(r["cost_per_year"] * 100, 1), "ann1": round(r["ann_simple"] * 100, 1),
        "ann2": round(r["ann_compound"] * 100, 1), "t": round(r["t_stat"], 2),
    } for r in hp["table"]])

    gain = pd.read_csv(ROOT / "outputs/walkforward_v2/feature_gain_rank.csv", index_col=0)["gain"]
    gp = (gain / gain.sum() * 100).round(2).head(10)
    gain_rows = [{"f": k, "g": float(v)} for k, v in gp.items()]

    daily = pd.read_csv(ROOT / "outputs/backtest_v2/daily_rank_top1.csv", parse_dates=["date"])
    daily["year"] = daily["date"].dt.year
    yr = daily.groupby("year").agg(days=("net", "size"), hit=("hit", "mean"), gross=("gross", "mean"),
                                   uni=("uni_gross", "mean"), net=("net", "mean"),
                                   cum=("net", lambda x: float((1 + x).prod() - 1))).reset_index()
    yr_df = pd.DataFrame([{"y": int(r.year), "d": int(r.days), "h": round(r.hit * 100, 1),
                           "g": round(r.gross * 100, 3), "u": round(r.uni * 100, 3),
                           "n": round(r.net * 100, 3), "c": round(r.cum * 100, 1)} for _, r in yr.iterrows()])

    ex_t = float(daily["excess_gross"].mean() / (daily["excess_gross"].std() / np.sqrt(len(daily))))
    t_gross = float(b1["gross_mean"] / (daily["gross"].std() / np.sqrt(len(daily))))
    ic, icB = rb["ic"], rbB["ic"]

    av = avoid_effect(ROOT / "outputs/walkforward_v2/predictions.parquet", ROOT / "outputs/t1_dataset.parquet")
    avB = avoid_effect(ROOT / "outputs/walkforward_v2_lag1/predictions.parquet", ROOT / "outputs/t1_dataset.parquet")
    av_df = pd.DataFrame([
        {"c": "全样本（不过滤）", "a": f"{av['all'][0]*100:+.4f}%", "ah": f"{av['all'][1]*100:.2f}%",
         "b": f"{avB['all'][0]*100:+.4f}%", "bh": f"{avB['all'][1]*100:.2f}%"},
        {"c": "剔除最差 10%", "a": f"{av[0.1][0]*100:+.4f}%", "ah": f"{av[0.1][1]*100:.2f}%",
         "b": f"{avB[0.1][0]*100:+.4f}%", "bh": f"{avB[0.1][1]*100:.2f}%"},
        {"c": "剔除最差 20%", "a": f"{av[0.2][0]*100:+.4f}%", "ah": f"{av[0.2][1]*100:.2f}%",
         "b": f"{avB[0.2][0]*100:+.4f}%", "bh": f"{avB[0.2][1]*100:.2f}%"},
        {"c": "剔除最差 30%", "a": f"{av[0.3][0]*100:+.4f}%", "ah": f"{av[0.3][1]*100:.2f}%",
         "b": f"{avB[0.3][0]*100:+.4f}%", "bh": f"{avB[0.3][1]*100:.2f}%"},
    ])

    scen = pd.DataFrame([
        {"name": "保守：万 2.5 佣金 + 0.05% 滑点×2", "rt": 0.0020},
        {"name": "优化：万一佣金 + 0.02% 滑点×2", "rt": 0.0010},
        {"name": "极优：万一 + 限价单无滑点", "rt": 0.0007},
        {"name": "理论下限：零摩擦", "rt": 0.0},
    ])
    scen["net"] = b1["gross_mean"] - scen["rt"]
    scen["rt_s"] = scen["rt"].map(lambda v: f"{v*100:.2f}%")
    scen["net_s"] = scen["net"].map(lambda v: f"{v*100:+.3f}%")
    scen["ann_s"] = scen["net"].map(lambda v: f"{((1+v)**244-1)*100:+.1f}%")

    latest = pd.read_csv(ROOT / "outputs/backtest_v2/latest_top20.csv")
    nxt = pd.read_csv(ROOT / "outputs/live/blacklist_next_session.csv", dtype={"code": str})
    nxt["code"] = nxt["code"].str.zfill(6)

    css = f"""
    :root {{ --bg:{PALETTE['bg']}; --card:{PALETTE['card']}; --ink:{PALETTE['ink']};
             --muted:{PALETTE['muted']}; --line:{PALETTE['line']}; --up:{PALETTE['up']};
             --down:{PALETTE['down']}; --accent:{PALETTE['accent']}; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink);
      font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB",sans-serif;
      font-size:15px; line-height:1.75; -webkit-font-smoothing:antialiased; }}
    .wrap {{ max-width:1040px; margin:0 auto; padding:54px 28px 90px; }}
    h1 {{ font-size:29px; line-height:1.36; margin:0 0 8px; letter-spacing:.2px; }}
    h2 {{ font-size:19px; margin:50px 0 12px; padding-bottom:9px; border-bottom:1px solid var(--line); }}
    h3 {{ font-size:15.5px; margin:24px 0 6px; color:#4a423c; }}
    .sub {{ color:var(--muted); font-size:13.3px; margin-bottom:30px; line-height:1.7; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:20px 22px; margin:16px 0; }}
    .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:22px 0 6px; }}
    .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:15px 15px 13px; }}
    .kpi .l {{ font-size:11.5px; color:var(--muted); letter-spacing:.3px; }}
    .kpi .v {{ font-size:22px; font-weight:600; margin-top:4px; font-variant-numeric:tabular-nums; }}
    .kpi .n {{ font-size:11px; color:var(--muted); margin-top:2px; line-height:1.5; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; margin:8px 0 2px; font-variant-numeric:tabular-nums; }}
    th {{ text-align:right; font-weight:600; color:var(--muted); font-size:11.5px;
          border-bottom:1px solid var(--line); padding:8px 7px; }}
    th:first-child, td:first-child {{ text-align:left; }}
    td {{ text-align:right; padding:6.5px 7px; border-bottom:1px solid #efe7dc; }}
    tr.hl td {{ background:#fbf3ea; font-weight:600; }}
    .up {{ color:var(--up); }} .down {{ color:var(--down); }}
    ul {{ padding-left:20px; }} li {{ margin:5px 0; }}
    .note {{ font-size:12.5px; color:var(--muted); line-height:1.65; }}
    .warn {{ border-left:3px solid var(--accent); padding-left:15px; background:#fdf8f3; }}
    .good {{ border-left:3px solid var(--accent2); padding-left:15px; background:#f7f8f3; }}
    code {{ background:#f2ebe2; padding:1px 5px; border-radius:4px; font-size:12.3px; }}
    .tag {{ display:inline-block; font-size:11px; padding:2px 8px; border-radius:20px;
            border:1px solid var(--line); color:var(--muted); margin:0 5px 5px 0; }}
    footer {{ margin-top:56px; padding-top:16px; border-top:1px solid var(--line); color:var(--muted); font-size:11.8px; }}
    """

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>T+1 隔夜选股模型 · 建模与回测报告</title><style>{css}</style></head><body><div class="wrap">

<h1>A股 T+1 隔夜选股模型 · 建模与回测</h1>
<div class="sub">数据：本地十年硬数据池 5,577 只 · 2016-01-04 ～ {rb['date_max']}（10,888,307 根日 K）<br>
样本外：2022-01 ～ {rb['date_max']} · {rb['n_dates']} 个交易日 · 季度滚动重训 20 折 · 全部结论扣双边成本</div>

<div class="kpis">
  <div class="kpi"><div class="l">横截面 IC（当日收盘口径）</div><div class="v">{ic['mean']:.3f}</div>
    <div class="n">ICIR {ic['icir']:.1f} · 正 IC 日 {ic['positive_share']*100:.0f}%</div></div>
  <div class="kpi"><div class="l">Top1 毛收益（当日收盘口径）</div>
    <div class="v up">{b1['gross_mean']*100:+.3f}%</div>
    <div class="n">胜率 {top1['hit_rate']*100:.1f}% · 全市场 {rnd['hit_rate']*100:.1f}%</div></div>
  <div class="kpi"><div class="l">Top1 毛收益（严格口径 D−1）</div>
    <div class="v down">{beB['rank_top1']['gross_mean']*100:+.3f}%</div>
    <div class="n">胜率 {sB['rank_top1']['hit_rate']*100:.1f}% · IC {icB['mean']:.3f}</div></div>
  <div class="kpi"><div class="l">保守往返成本</div><div class="v">0.20%</div>
    <div class="n">每日满换仓 ≈ 年化摩擦 49%</div></div>
</div>

<h2>1. 结论先行（三条，按重要性排）</h2>
<div class="card warn">
<p><strong>① 日频数据做"每日 1 票"选股，选不出正期望的头。</strong>
把特征严格滞后到 D−1（只用昨天及以前的数据下单），Top1 次日毛收益
<strong>{beB['rank_top1']['gross_mean']*100:+.3f}%</strong>、胜率 {sB['rank_top1']['hit_rate']*100:.1f}%，
不但不赚，还明显跑输随机（随机基线毛 {be['baseline_random']['gross_mean']*100:+.3f}%）。
同期 IC 仍有 {icB['mean']:.3f}，说明信号集中在"识别烂票"一端，不在"挑出好票"一端。</p>
<p><strong>② 加上当日收盘信息，选股能力出现了，但正好被成本吃掉。</strong>
用 D 日完整日 K 时，Top1 毛收益 <strong>{b1['gross_mean']*100:+.3f}%</strong>／日、
日均超额 <strong>{b1['excess_gross_mean']*100:+.3f}%</strong>（t={ex_t:.2f}）、IC {ic['mean']:.3f}，信号真实。
但保守往返成本 0.20% 意味着年化摩擦约 49%，净收益 {top1['mean_daily']*100:+.3f}%／日，
10 万 4.8 年做到 <strong>{100000*(1+top1['total_return']):,.0f} 元</strong>
（同期随机基线 {100000*(1+rnd['total_return']):,.0f} 元）。</p>
<p><strong>③ 立刻能用的不是选股，是排雷。</strong>
剔掉模型最不看好的 20%，保留样本的次日平均收益从 {avB['all'][0]*100:+.4f}% 抬到
<strong>{avB[0.2][0]*100:+.4f}%</strong>——严格口径下依然成立。不需要每天满仓换股，直接套在现有候选池上。</p>
</div>

<h2>2. 口径与方法</h2>
<h3>2.1 标签与可执行性</h3>
<ul>
<li>标签：<code>fwd_cc = close(D+1)/close(D) − 1</code>，D 日收盘买入、D+1 日收盘卖出，T+1 合规。</li>
<li>剔除 D+1 停牌（次日无 bar）样本；剔除 D 日收盘涨停（买不到）、成交额 &lt; 2,000 万、价格不在 2～300 元区间。</li>
<li>ST 处理：数据源无历史 ST 名称（财报表 <code>name</code> 是当前名，回溯使用会引入未来函数），
改用<strong>点位内代理</strong>——过去 60 日收益率是否被锁在 ±5.25% 内，反推 5% 涨跌停制度。</li>
<li>标签已用原始日 K 独立复算校验：50 个抽样点最大误差 5e-8（float32 精度级），无前视、无停牌错配。</li>
</ul>

<h3>2.2 特征（41 个，全部点位内可得）</h3>
<p><span class="tag">短周期量价</span>
<code>ret_1/3/5/10/20/60</code> <code>gap</code> <code>intraday</code> <code>close_pos</code>
<code>amp</code> <code>z_ret</code> <code>streak_up</code></p>
<p><span class="tag">风险与位置</span>
<code>vol_5/20</code> <code>vol_ratio</code> <code>ma_dev_5/20/60</code>
<code>dist_high_20</code> <code>dist_low_20</code> <code>dist_high_5</code></p>
<p><span class="tag">流动性</span>
<code>log_amt</code> <code>amt_ratio</code> <code>amt_trend</code> <code>turn</code>
<code>turn_ratio</code> <code>log_mktcap</code> <code>days_listed</code></p>
<p><span class="tag">情绪 / 基本面</span>
<code>lu_cnt_5/20</code>（近 5/20 日涨停次数）· ROE · 净利同比 · 营收同比（按公告日 asof 对齐）</p>
<p><span class="tag">横截面分位</span>
<code>cs_ret_1</code> <code>cs_ret_5</code> <code>cs_amt</code> <code>cs_turn</code> <code>cs_ma_dev_20</code></p>

<h3>2.3 训练设计（三个刻意的选择）</h3>
<ul>
<li><strong>只给横截面排序，不给大盘择时。</strong>刻意剔除大盘涨跌、涨跌家数等日级特征。第一版曾喂进去，
gain 前三全被市场特征占据、排序能力退化——单票选择只需要横截面信息。</li>
<li><strong>双头。</strong>排序头回归"当日收益横截面分位（0～1）"，用于选股；概率头做"上涨概率"二分类，用于校准与报告。
实测<strong>排序头明显优于概率头</strong>（毛 {b1['gross_mean']*100:+.3f}% vs {be['prob_top1']['gross_mean']*100:+.3f}%）：
预测"涨不涨"偏向低波动安全票，预测"涨多少的相对次序"才对准收益。</li>
<li><strong>只在可买样本上训练。</strong>涨停、低流动性、ST 代理从训练集剔掉，避免模型学会一堆买不到的票。</li>
</ul>

<h2>3. 关键对照：那"20 分钟"值多少 alpha</h2>
<p>整份报告最重要的一张表。两套口径除特征截止时点外完全相同（同数据、同特征、同折、同参数）：</p>
<table><thead><tr><th>口径</th><th>特征截止</th><th>IC</th><th>Top1 毛收益</th><th>Top1 胜率</th><th>Top10 毛收益</th></tr></thead><tbody>
<tr><td>当日收盘（可实盘近似）</td><td>D 日收盘</td><td>{ic['mean']:.3f}</td><td class="up">{b1['gross_mean']*100:+.3f}%</td><td>{top1['hit_rate']*100:.1f}%</td><td class="up">{be['rank_top10']['gross_mean']*100:+.3f}%</td></tr>
<tr class="hl"><td>严格口径</td><td>D−1 日收盘</td><td>{icB['mean']:.3f}</td><td class="down">{beB['rank_top1']['gross_mean']*100:+.3f}%</td><td>{sB['rank_top1']['hit_rate']*100:.1f}%</td><td class="down">{beB['rank_top10']['gross_mean']*100:+.3f}%</td></tr>
</tbody></table>
<p>严格口径下分位结构仍单调（D1 {decB['mean'].iloc[0]*100:+.2f}% → D10 {decB['mean'].iloc[9]*100:+.2f}%），
但<strong>顶部几乎是平的</strong>；当日收盘口径下 D10 才抬到 {dec['mean'].iloc[9]*100:+.2f}%。</p>
{svg_bars(decB_rows, 'd', 'm', ylabel='严格口径（D−1）：次日平均收益（%）')}
<p class="note">含义很直接：<strong>一天里最值钱的信息，是"今天盘中到收盘这段发生了什么"</strong>——
跌幅、换手、收盘位置。而这些恰好是 14:36 下单时还看不到的。日频数据这条路，在这里就断了。</p>

<h2>4. 当日收盘口径的完整结果</h2>
<p class="note">样本外、扣 0.20% 双边成本（佣金万 2.5×2 + 印花税 0.05% + 滑点 0.05%×2）。</p>
{table(summ, ['label','days','hit_s','mean_s','ann_s','dd_s','sharpe_s','t_s'],
       ['策略','交易日','胜率(%)','日均净收益(%)','年化(%)','最大回撤(%)','Sharpe','t 值'], 'rank_top1')}

<h3>4.1 分位结构：避开烂票，而不是挑出好票</h3>
{svg_bars(dec_rows, 'd', 'm', ylabel='当日收盘口径：次日平均收益（%）')}
<p class="note">D1 档是"当日涨幅最大 + 换手率横截面分位最高"的票（平均当日 {dec['ret1'].iloc[0]*100:+.2f}%），次日 {dec['mean'].iloc[0]*100:+.2f}%；
D10 档是"当日大跌 + 高换手"的票（平均当日 {dec['ret1'].iloc[9]*100:+.2f}%），次日 {dec['mean'].iloc[9]*100:+.2f}%。
典型的一日反转结构，且<strong>左尾比右尾肥</strong>（D1 档 5% 分位 {dec['p05'].iloc[0]*100:+.1f}%）。</p>

<h3>4.2 净值曲线（10 万本金）</h3>
{svg_lines(rb['curves'])}

<h3>4.3 分年表现（Top1）</h3>
{table(yr_df, ['y','d','h','g','u','n','c'],
       ['年份','交易日','胜率(%)','毛收益(%)','全市场(%)','净收益(%)','当年累计(%)'])}
<p class="note">2024、2025 两年为正，2022、2023、2026 为负。信号强依赖"高换手 + 高波动"的市场状态，
缩量年份持续失血——不能无脑全天候运行。</p>

<h2>5. 成本是决定性变量</h2>
{svg_bars(cs_rows, 'c', 'a', ylabel='年化收益（扣对应双边成本后）')}
{table(scen, ['name','rt_s','net_s','ann_s'], ['成本口径','双边成本','Top1 日均净收益','粗略年化'])}
<p class="note">盈亏平衡点 ≈ <strong>{b1['breakeven_cost']*100:.3f}%</strong>。
散户最好档位（万一佣金 + 印花税 + 极小滑点）约 0.07%～0.10%，此时年化 +2%～+5%，
却要背 {top1['max_drawdown']*100:.0f}% 级别回撤，风险收益比不成立。
另注：10 万拆 5 只时每笔 2 万，最低佣金 5 元相当于 0.025%／边，成本反而比单票更高。</p>

<h2>6. 换持有期能不能救</h2>
<p>同一个 1 日排序信号，改为持有 k 日后卖出（每 k 天换一次仓），摩擦按天数摊薄——不增加信息源时唯一的结构性杠杆。</p>
{table(hp_df, ['k','trades','hit','gross','net','cost','ann1','ann2','t'],
       ['持有期','交易笔数','胜率(%)','平均毛收益(%)','平均净收益(%)','年化摩擦(%)','年化(无波动拖累)','年化(含拖累)','t 值'])}
<p class="note">alpha 随持有期快速衰减：毛收益 1 日 {hp_df.iloc[0]['gross']:.3f}%、2 日 {hp_df.iloc[1]['gross']:.3f}%、
3 日 {hp_df.iloc[2]['gross']:.3f}%、5 日 {hp_df.iloc[3]['gross']:.3f}%，按日摊薄从 0.14% 掉到 0.06%。
摩擦按 1/k 下降，两者赛跑的结果：2 日持有在无波动拖累口径下年化 {hp_df.iloc[1]['ann1']:.1f}%，
含波动拖累后 {hp_df.iloc[1]['ann2']:.1f}%。<strong>救不回来。</strong></p>

<h2>7. 唯一立刻能用的东西：排雷清单</h2>
<div class="card good">
<p>不去"选一只明天要涨的"，而是"标记出明天大概率跑输的一批"。这件事在严格口径下也成立：</p>
{table(av_df, ['c','a','ah','b','bh'],
       ['处理方式','当日收盘口径 · 次日均值','胜率','严格口径 · 次日均值','胜率'])}
<p class="note">用法：套在现有候选池上——凡落在 D1/D2 档的票，当日不买或减仓。
严格口径下仍有 +0.049%／日的相对改善，且成本几乎为零（只减少交易，不增加）。</p>
</div>

<h2>8. 模型实际依赖的特征</h2>
<p>排序头 gain 占比（最后一折）：换手率横截面分位、当日涨幅横截面分位、收盘位置三者合计约
{gp.iloc[0]:.0f}% + {gp.iloc[1]:.0f}% + {gp.iloc[2]:.0f}%，本质是<strong>"高位放量滞涨 → 次日回落"</strong>的反转结构。
财报类特征在日频几乎没有贡献，符合预期。</p>
{svg_bars(gain_rows, 'f', 'g', pctfmt=False, ylabel='gain 占比（%）')}

<h2>9. 最新一日输出（演示，非建议）</h2>
<p>截至数据最新交易日 {rb['date_max']}，可买样本内排序头前 8 名（当日收盘口径模型，仅供理解模型偏好）：</p>
{table(latest.head(8), ['code','name','close','ret_1','score','prob'],
       ['代码','名称','收盘','当日涨跌(%)','排序分','上涨概率'])}
<p class="note">这一天整体呈现"当日深跌 3%～9%、高换手、收盘接近全天最低"的形态，
正是模型学到的反转结构。以上为模型输出，未经实盘验证。</p>

<h3>9.1 下一交易日可直接用的排雷清单</h3>
<p>严格口径模型只需 D−1 数据，因此可以真正"提前"出名单。以 2026-10-23 收盘特征对下一交易日打分，
黑名单（垫底 20%，共 {len(nxt):,} 只）中排序最低的 10 只：</p>
{table(nxt.head(10).assign(
        close=nxt['close'].round(2), ret_1=(nxt['ret_1'] * 100).round(2),
        score=nxt['score'].round(4)), ['code','name','close','ret_1','score'],
       ['代码','名称','参考价','当日涨跌(%)','排序分'])}
<p class="note">完整名单见 <code>outputs/live/blacklist_next_session.csv</code>，
全市场打分见 <code>outputs/live/scores_snapshot.csv</code>。用法：这些票当日不新开仓；已持有的可考虑减。
注意：上线时应用真实证券简称再过滤一次 ST（代理识别有漏网）。</p>

<h2>10. 诚实边界</h2>
<ul>
<li><strong>当日收盘口径含 20 分钟偏差</strong>：特征用到 D 日收盘 bar，实盘 14:36 下单时收盘尚未发生。
严格口径（D−1）已给出对照，这是报告的核心结论之一——<strong>不要只看第 4 节的光鲜数字</strong>。</li>
<li><strong>幸存者偏差</strong>：数据池只含当前存续 5,577 只，缺退市股，系统性抬高所有多头策略。</li>
<li><strong>ST 是代理识别</strong>：创业板/科创板 ST（20% 制度）识别不到。</li>
<li><strong>成交假设乐观</strong>：假设收盘价全额成交，未建模涨停封单、跌停卖不出、流动性冲击。</li>
<li><strong>统计显著性不足</strong>：Top1 毛收益 t 值约 {t_gross:.2f}，4.8 年样本外不足以证明 alpha 稳定；分年有 3 年为负。</li>
</ul>

<h2>11. 下一步（按性价比排序）</h2>
<ul>
<li><strong>P0 · 接分钟数据</strong>：结论已经说死——日频收盘数据的天花板在这里。
真正有 alpha 的是"当天到尾盘为止的跌幅 + 换手 + 收盘位置"，必须用 14:36 快照或分钟线重算这部分特征。</li>
<li><strong>P0 · 排雷清单先上线</strong>：第 7 节那个用法零成本、严格口径有效，可直接套进现有 10 万实盘流程。</li>
<li><strong>P1 · 降换手 / 降成本</strong>：即便信号成立，49% 年化摩擦也吃光。要么 2～3 日持有，要么把双边压到 0.10% 以内。</li>
<li><strong>P1 · 目标函数改成净收益</strong>：现在排序头学"收益分位"，应直接学 <code>fwd_cc − 成本 − 尾部惩罚</code>。</li>
<li><strong>P2 · 做空 / 对冲端</strong>：D1 档次日 {dec['mean'].iloc[0]*100:+.2f}% 的识别力远强于 D10 档 {dec['mean'].iloc[9]*100:+.2f}%。
这份能力在"排雷""指数对冲""持仓再平衡"上比在多头选股上更值钱。</li>
</ul>

<footer>数据源：本地 A 股十年硬数据池（日 K 前复权，2016-01-04 ～ {rb['date_max']}）｜
样本外 {rb['date_min']} ～ {rb['date_max']}｜LightGBM 双头（排序头 + 概率头）+ 季度 walk-forward 20 折｜
特征 41 个，全部点位内可得｜代码与中间结果见 <code>t1overnight/</code>｜本报告为量化研究记录，不构成投资建议。</footer>
</div></body></html>"""

    fp = OUT / "T1隔夜模型报告.html"
    fp.write_text(html, encoding="utf-8")
    print(f"[saved] {fp} ({fp.stat().st_size/1024:.0f}KB)")
    return fp


if __name__ == "__main__":
    build()
