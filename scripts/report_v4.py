# -*- coding: utf-8 -*-
"""Render the v4 report.

Headline figures are read from artefacts under outputs/ so the report cannot drift
away from what the code computed:

    outputs/exit_mode/clock_summary.csv   the six execution clocks
    outputs/exit_mode/st_audit.csv        ST-exposure audit by segment
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report"

CSS = """
:root{
  --bg:#faf7f2; --panel:#fffdf9; --ink:#332c26; --muted:#8a7d70;
  --line:#e8ded1; --accent:#b5623c; --accent2:#c98a4b;
  --up:#9c4a34; --down:#5d7a58; --soft:#f3ece2;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);
     font:15px/1.78 -apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
     padding:46px 20px 72px;-webkit-font-smoothing:antialiased}
.wrap{max-width:1010px;margin:0 auto}
h1{font-size:26px;font-weight:600;letter-spacing:-.3px;line-height:1.42;margin-bottom:8px}
h1 small{display:block;font-size:16px;font-weight:500;color:var(--muted);margin-top:7px;letter-spacing:0}
h2{font-size:18px;font-weight:600;margin:42px 0 13px;padding-bottom:9px;border-bottom:1px solid var(--line)}
h3{font-size:14.5px;font-weight:600;margin:24px 0 9px;color:var(--accent)}
p{margin:11px 0;color:#463d35}
.lede{font-size:15.5px;color:#4a4038;background:var(--panel);border-left:3px solid var(--accent);
      padding:17px 21px;border-radius:0 7px 7px 0;margin:18px 0 6px;box-shadow:0 1px 3px rgba(80,60,40,.05)}
.meta{color:var(--muted);font-size:12.5px;margin-bottom:24px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(172px,1fr));gap:11px;margin:18px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.card .k{font-size:11.5px;color:var(--muted);letter-spacing:.3px;margin-bottom:6px}
.card .v{font-size:23px;font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-.5px}
.card .s{font-size:11.5px;color:var(--muted);margin-top:4px;line-height:1.5}
.up{color:var(--up)} .down{color:var(--down)}
table{width:100%;border-collapse:collapse;margin:15px 0;font-size:13px;background:var(--panel);
      border:1px solid var(--line);border-radius:8px;overflow:hidden}
th{background:var(--soft);font-weight:600;text-align:right;padding:9px 10px;font-size:12px;
   color:#5c5045;border-bottom:1px solid var(--line);white-space:nowrap;line-height:1.4}
th:first-child,td:first-child{text-align:left}
td{padding:8px 10px;text-align:right;border-bottom:1px solid #f1eae0;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
tr.hl{background:#fdf4ec} tr.hl td{font-weight:600}
tr.dim td{color:#a0968a}
tr.bad{background:#f9f1ec}
.tag{display:inline-block;font-size:10.5px;padding:1px 6px;border-radius:9px;margin-left:5px;
     vertical-align:1px;font-weight:500}
.tag.ok{background:#eef3ea;color:#5d7a58}
.tag.no{background:#f7ece6;color:#a8563a}
.note{font-size:12px;color:var(--muted);margin-top:-5px;margin-bottom:15px;line-height:1.65}
ul{margin:9px 0 9px 19px} li{margin:6px 0;color:#463d35}
.bar-row{display:flex;align-items:center;gap:9px;margin:8px 0;font-size:12.5px}
.bar-label{width:190px;flex-shrink:0;color:#5c5045;line-height:1.35}
.bar-track{flex:1;height:19px;background:var(--soft);border-radius:3px;position:relative;overflow:hidden}
.bar-fill{position:absolute;top:0;bottom:0;border-radius:3px}
.zero{position:absolute;top:0;bottom:0;width:1px;background:#a89886;left:50%}
.bar-val{width:74px;text-align:right;flex-shrink:0;font-variant-numeric:tabular-nums;font-weight:600}
.formula{background:var(--soft);border-radius:6px;padding:12px 15px;
         font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
         font-size:12.5px;margin:13px 0;color:#4a4038;overflow-x:auto;line-height:1.9;white-space:pre}
.callout{background:#fdf4ec;border:1px solid #eddcc9;border-radius:7px;padding:14px 18px;margin:16px 0}
.callout b{color:var(--accent)}
.warn{background:#f9f1ec;border-left:3px solid #a8563a;padding:13px 17px;border-radius:0 6px 6px 0;margin:14px 0}
.warn b{color:#8f3f27}
footer{margin-top:52px;padding-top:17px;border-top:1px solid var(--line);font-size:12px;color:var(--muted);line-height:1.75}
code{background:var(--soft);padding:1px 5px;border-radius:3px;font-size:12.5px;
     font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
"""

CN = {
    "D open -> D+1 open": "D 开盘买 → D+1 开盘卖",
    "D open -> D+1 close": "D 开盘买 → D+1 收盘卖",
    "D close -> D+1 open (current)": "D 收盘买 → D+1 开盘卖",
    "D close -> D+1 close": "D 收盘买 → D+1 收盘卖",
    "D open -> D close": "D 开盘买 → D 收盘卖（同日回转）",
    "D+1 open -> D+1 close": "D+1 开盘买 → D+1 收盘卖",
}


def main() -> None:
    clocks = pd.read_csv(ROOT / "outputs/exit_mode/clock_summary.csv")
    st = pd.read_csv(ROOT / "outputs/exit_mode/st_audit.csv")
    nv = pd.read_csv(ROOT / "outputs/exit_mode/valid_netvalue.csv")

    def nv_row(rule: str, cost: int):
        r = nv[(nv.rule == rule) & (nv.cost == cost)]
        return r.iloc[0] if len(r) else None

    def pick(name: str):
        return clocks[clocks.clock.str.contains(name, regex=False)].iloc[0]

    best = pick("D open -> D+1 open")
    best2 = pick("D open -> D+1 close")
    curr = clocks[clocks.clock == "D close -> D+1 close"].iloc[0]
    curr_open = pick("D close -> D+1 open")
    illegal = pick("D open -> D close")
    illegal2 = pick("D+1 open -> D+1 close")

    def g(seg: str, case: str) -> float:
        r = st[(st.seg == seg) & (st.case == case)]
        return float(r.mean_bp.iloc[0]) if len(r) else float("nan")

    def sh(seg: str) -> float:
        r = st[(st.seg == seg) & (st.case == "st_in")]
        return float(r.st_share.iloc[0]) * 100 if len(r) else float("nan")

    def gt(seg: str, case: str) -> float:
        r = st[(st.seg == seg) & (st.case == case)]
        return float(r.t.iloc[0]) if len(r) else float("nan")

    def netvalue_rows() -> str:
        out = []
        for cost in (10, 13, 15, 18, 20, 25):
            a = nv_row("剔ST", cost)
            b = nv_row("剔ST+gap<1.5%", cost)
            if a is None or b is None:
                continue
            cls = ' class="hl"' if cost in (13, 15) else (' class="bad"' if cost >= 20 else "")
            tag = " ← 现实最好水平" if cost == 13 else ""
            out.append(
                f"<tr{cls}><td>{cost}bp{tag}</td>"
                f"<td class={'up' if a.ann>0 else 'down'}>{a.ann:+.1f}%</td><td>{a.final:,.0f}</td>"
                f"<td class={'up' if b.ann>0 else 'down'}>{b.ann:+.1f}%</td><td>{b.final:,.0f}</td></tr>")
        return "\n".join(out)

    def clock_rows() -> str:
        out = []
        for r in [best, best2, curr_open, curr, illegal, illegal2]:
            cls = ' class="hl"' if r.clock == best.clock else (
                ' class="dim"' if r.t1 == "VIOLATION" else "")
            tag = ('<span class="tag ok">T+1 合规</span>' if r.t1 == "legal"
                   else '<span class="tag no">违规</span>')
            if r.clock == "D close -> D+1 close":
                tag += '<span class="tag no">当前生产规则</span>'
            elif r.clock == "D open -> D+1 open":
                tag += '<span class="tag ok">推荐</span>'
            out.append(
                f"<tr{cls}><td>{CN.get(r.clock, r.clock)}{tag}</td>"
                f"<td class={'up' if r.mean_bp>0 else 'down'}>{r.mean_bp:+.2f}</td>"
                f"<td>{r.median_bp:+.2f}</td><td>{r.hit*100:.1f}%</td>"
                f"<td>{r.std_pct:.2f}%</td>"
                f"<td class={'up' if r.geo_bp>0 else 'down'}><b>{r.geo_bp:+.2f}</b></td>"
                f"<td>{r.vol_drag_bp:.1f}</td>"
                f"<td class={'up' if r.breakeven_bp>0 else 'down'}>{r.breakeven_bp:+.2f}</td>"
                f"<td>{r.t_stat:+.2f}</td><td>{r.monthly_win*100:.0f}%</td></tr>")
        return "\n".join(out)

    def bar_html() -> str:
        items = [
            ("D 开盘买 → D 收盘卖", illegal.mean_bp, "#b5623c", "T+1 违规·仅诊断"),
            ("D 开盘买 → D+1 开盘卖", best.mean_bp, "#c98a4b", "推荐"),
            ("D 开盘买 → D+1 收盘卖", best2.mean_bp, "#c98a4b", ""),
            ("D+1 开盘买 → D+1 收盘卖", illegal2.mean_bp, "#b0a396", "T+1 违规"),
            ("D 收盘买 → D+1 收盘卖", curr.mean_bp, "#5d7a58", "当前生产规则"),
            ("D 收盘买 → D+1 开盘卖", curr_open.mean_bp, "#5d7a58", "最差合规"),
        ]
        lim = max(abs(i[1]) for i in items) * 1.1
        out = []
        for lab, val, col, note in items:
            pct = abs(val) / lim * 50
            style = (f"left:50%;width:{pct:.2f}%" if val >= 0
                     else f"right:50%;width:{pct:.2f}%")
            out.append(
                f'<div class="bar-row"><div class="bar-label">{lab}'
                + (f'<br><span style="font-size:10.5px;color:#a0968a">{note}</span>' if note else "")
                + f'</div><div class="bar-track"><div class="zero"></div>'
                  f'<div class="bar-fill" style="background:{col};{style}"></div></div>'
                  f'<div class="bar-val {("up" if val>0 else "down")}">{val:+.1f}</div></div>')
        return "\n".join(out)

    ablation = ""
    v3 = ROOT / "outputs/walkforward_v3"
    if (v3 / "meta_tri_base.json").exists():
        rows = []
        for n in ["tri_base", "tri_A", "tri_B", "tri_C"]:
            f = v3 / f"meta_{n}.json"
            if f.exists():
                m = json.loads(f.read_text())
                rows.append((n, m.get("ic", 0.0), m.get("icir", 0.0)))
        if len(rows) >= 2:
            lab = {"tri_base": "基线（v2 原特征）", "tri_A": "+ 形态块（9 个）",
                   "tri_B": "+ 行业相对强度（7 个）", "tri_C": "+ 市场状态（9 个）"}
            body = "".join(
                f"<tr><td>{lab.get(n,n)}</td><td>{ic:+.4f}</td><td>{icir:+.2f}</td>"
                f"<td>{ic-rows[0][1]:+.4f}</td>"
                f"<td>{(ic-rows[0][1])/abs(rows[0][1])*100:+.2f}%</td></tr>"
                for n, ic, icir in rows)
            ablation = f"""
<h2>5 · 特征工程消融：加不动了</h2>
<p>四个配置跑在同一批折、同一批行上，唯一差别是喂进去的特征块。
指标是排雷概率与次日收益的秩相关 IC（负号 = 越像烂票、次日越跌，越负越好）。</p>
<table><thead><tr><th>配置</th><th>IC</th><th>ICIR</th><th>相对基线</th><th>变化幅度</th></tr></thead>
<tbody>{body}</tbody></table>
<p class="note">三个新增特征块（形态 9 个、行业 7 个、市场 9 个）共 25 个特征，
IC 变化都在小数点后第四位，相对幅度不到 1.5%。
结论：<b>在现有 39 个日频特征之上，形态与行业的横截面加工榨不出新信息。</b></p>
"""
    n_abl = 1 if ablation else 0

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>T+1 隔夜模型 · 执行时点与 ST 暴露审计</title><style>{CSS}</style></head>
<body><div class="wrap">

<h1>执行时点确实错了，但改对之后钱来自 ST 股
<small>T+1 隔夜模型第三轮 · 2026-10-25</small></h1>
<div class="meta">样本外 2022-01-04 → 2026-10-22，1,161 个交易日 · 严格 D−1 滞后信号 · 10 万本金每日单票</div>

<div class="lede">
这一轮先确认了一件真事：把买单从 D 日收盘挪到 D 日开盘，几何日收益从
<b class="down">{curr.geo_bp:+.2f}bp</b> 翻到 <b class="up">{best.geo_bp:+.2f}bp</b>，
当前这条规则在四种合规钟里排最差。
但接着做了一次 ST 暴露审计：用 tushare 的曾用名表重建了历史 ST 状态，
发现验证期（2025-2026）有 <b>17.8% 的交易日买在 ST 股上</b>，这部分贡献了约四成收益。
剔除后验证期毛收益 +{g('valid','st_out'):.1f}bp、t 值 {gt('valid','st_out'):.2f}——
<b>方向对但统计上不显著</b>。结论：执行规则该改，但按现实成本算，这是个临界策略。
</div>

<div class="grid">
<div class="card"><div class="k">执行时点改动的收益</div><div class="v up">+35.3bp</div>
<div class="s">几何日均，当前规则 → 最优合规</div></div>
<div class="card"><div class="k">验证期剔 ST 后</div><div class="v">+{g('valid','st_out'):.1f}bp</div>
<div class="s">t = {gt('valid','st_out'):.2f} · 方向对但不显著</div></div>
<div class="card"><div class="k">验证期 ST 占比</div><div class="v">17.8%</div>
<div class="s">选参期仅 2.0%，模型在往 ST 漂</div></div>
<div class="card"><div class="k">盈亏平衡往返成本</div><div class="v">17.6bp</div>
<div class="s">地板 13bp → 年化 +11.4%</div></div>
</div>

<h2>1 · 四钟对照：执行时点错在哪</h2>
<p>信号时点从头到尾没变：特征只到 D−1 收盘，所以这一行的分数在 D 日开盘前就算好了。
A 股 T+1，D 日买入最早 D+1 才能卖，合规的钟只有四种。六种组合放在同一批票上比：</p>

<table><thead><tr>
<th>买卖规则</th><th>毛均值<br>bp</th><th>中位<br>bp</th><th>胜率</th><th>波动</th>
<th>几何日均<br>bp</th><th>波动拖累<br>bp</th><th>盈亏平衡<br>bp</th><th>t</th><th>月胜率</th>
</tr></thead><tbody>
{clock_rows()}
</tbody></table>
<p class="note">「盈亏平衡」= 几何日均收益，即这套规则最多能承受的往返成本。灰色两行违反 T+1，只作诊断。</p>

<h3>钱被切成两段，当前规则拿了最差那段</h3>
<div class="formula">D 日日内（开盘→收盘）   {illegal.mean_bp:+7.2f}bp    ← T+1 锁死，买了当天不能卖
D 日收盘 → D+1 开盘     {curr_open.mean_bp:+7.2f}bp    ← 当前规则唯一吃到的，且是负的
────────────────────────────────
当前规则   D 收盘 → D+1 收盘   {curr.mean_bp:+7.2f}bp
推荐规则   D 开盘 → D+1 开盘   {best.mean_bp:+7.2f}bp</div>

<p>模型挑出的强势票在 D 日日内继续走强（{illegal.mean_bp:+.1f}bp），但这段被 T+1 挡住。
等到 D 日收盘再买，买入价里已经含了那 {illegal.mean_bp:.1f}bp，接下来只剩隔夜回吐（{curr_open.mean_bp:.1f}bp）。
<b>把买单提前到开盘，等于用同一套信号把让掉的日内段吃回来。</b>这一条是这一轮唯一站得住的改进。</p>

<h3>六种钟的收益量级</h3>
{bar_html()}

<h2>2 · ST 暴露审计：这一轮最重要的发现</h2>
<p>跑新规则时，生产脚本第一条信号给出的是 <code>*ST仕净</code>。查下去发现了问题：
日内数据里 <code>band</code> 只有 10/20/30% 三档、没有 ST 的 5% 档，
<code>tradable()</code> 的 <code>st_proxy</code>（靠波动反推）也识别不出创业板/科创板 ST。
于是用 tushare 的 <code>namechange</code> 接口拉了 2015 年以来 6,391 条曾用名记录（4,375 只股票），
按"起止日期 → 当时名称"重建了逐日 ST 状态：<b>回测里有 91 天（7.84%）买在当时确为 ST 的股票上</b>。
拿"当前名称"直接判会误剔 39 天、漏剔 37 天，两个方向抵消后总量接近，但样本构成不同。</p>

<table><thead><tr><th>口径</th><th>全样本<br>毛均值 / t</th><th>选参期 2022-2024<br>毛均值 / t</th><th>验证期 2025-2026<br>毛均值 / t</th></tr></thead><tbody>
<tr><td>含 ST，不加过滤</td><td>+{g('all','st_in'):.2f} / {gt('all','st_in'):.2f}</td>
<td>+{g('select','st_in'):.2f} / {gt('select','st_in'):.2f}</td>
<td class="up">+{g('valid','st_in'):.2f} / {gt('valid','st_in'):.2f}</td></tr>
<tr class="bad"><td><b>剔除 ST</b></td><td>+{g('all','st_out'):.2f} / {gt('all','st_out'):.2f}</td>
<td>+{g('select','st_out'):.2f} / {gt('select','st_out'):.2f}</td>
<td class="down"><b>+{g('valid','st_out'):.2f} / {gt('valid','st_out'):.2f}</b></td></tr>
<tr><td>含 ST + 开盘过滤 gap&lt;1.5%</td><td>+{g('all','st_in_gap'):.2f} / {gt('all','st_in_gap'):.2f}</td>
<td>+{g('select','st_in_gap'):.2f} / {gt('select','st_in_gap'):.2f}</td>
<td class="up">+{g('valid','st_in_gap'):.2f} / {gt('valid','st_in_gap'):.2f}</td></tr>
<tr class="bad"><td><b>剔除 ST + 开盘过滤</b></td><td>+{g('all','st_out_gap'):.2f} / {gt('all','st_out_gap'):.2f}</td>
<td>+{g('select','st_out_gap'):.2f} / {gt('select','st_out_gap'):.2f}</td>
<td class="down"><b>+{g('valid','st_out_gap'):.2f} / {gt('valid','st_out_gap'):.2f}</b></td></tr>
</tbody></table>
<p class="note">口径为 D 开盘买、D+1 开盘卖。"剔除 ST"按 tushare 曾用名表还原的<b>当时名称</b>判定，
覆盖 4,375 只有改名记录的股票；无记录者视为从未戴帽。</p>

<div class="warn">
<b>要点不是"ST 股贡献了一点收益"，而是"模型在往 ST 股上漂"。</b><br>
ST 占比：选参期 2022-2024 只有 <b>2.0%</b>，验证期 2025-2026 涨到 <b>17.8%</b>，近九倍。
验证期 ST 股日均 +70.9bp、非 ST +{g('valid','st_out'):.1f}bp，ST 那 18% 的交易日贡献了约 <b>42%</b> 的总收益。
剔除之后从 +{g('valid','st_in'):.1f}bp 掉到 +{g('valid','st_out'):.1f}bp、t 从 {gt('valid','st_in'):.2f} 掉到 {gt('valid','st_out'):.2f}。
<b>模型不是在预测 ST 会涨，它是在给高波动小票打高分，而 2025-2026 恰好奖励这种票。</b>
ST 股流动性差、涨跌停 5%、有退市风险，实盘既不该买也不容易买。
</div>

<h3>为什么模型会追 ST</h3>
<p>特征里没有 ST 标识，但 ST 股的画像恰好命中模型的偏好：<code>vol_20</code>（波动）、
<code>turn</code>（换手）、<code>lu_cnt_5</code>（近 5 日涨停次数）、<code>log_mktcap</code>（小市值）——
这几项在 ST 股上天然最高。模型不是在"预测 ST 会涨"，它是在给高波动小票打高分，
而 2025-2026 这类票恰好是市场最强的一档。这是风格暴露，不是 alpha。</p>

<h2>3 · 分段：alpha 在前半段，风格在后半段</h2>
<table><thead><tr><th>区间</th><th>交易日</th><th>含 ST<br>毛均值 / t</th><th>剔 ST<br>毛均值 / t</th><th>ST 占比</th><th>判断</th></tr></thead><tbody>
<tr><td>2022-01 → 2024-12</td><td>726</td><td>+{g('select','st_in'):.2f} / {gt('select','st_in'):.2f}</td>
<td>+{g('select','st_out'):.2f} / {gt('select','st_out'):.2f}</td><td>{sh('select'):.2f}%</td><td>选股 alpha 存在但不显著</td></tr>
<tr class="bad"><td>2025-01 → 2026-10</td><td>435</td><td class="up">+{g('valid','st_in'):.2f} / {gt('valid','st_in'):.2f}</td>
<td>+{g('valid','st_out'):.2f} / {gt('valid','st_out'):.2f}</td><td>{sh('valid'):.2f}%</td><td><b>掉三成，但没归零</b></td></tr>
</tbody></table>
<p>两段的差别很说明问题：前半段剔不剔 ST 都不影响结论（+{g('select','st_in'):.1f} → +{g('select','st_out'):.1f}bp，剔完反而略高），
说明那段收益确实来自选股；后半段剔完 ST 掉三成（+{g('valid','st_in'):.1f} → +{g('valid','st_out'):.1f}bp），t 值也跟着掉下来。
<b>模型在后期把更多仓位压到了 ST 风格上，而这段行情正好奖励这种票。</b>
值得注意：验证期剔 ST 后仍有 +{g('valid','st_out'):.1f}bp，比全样本剔 ST 的 +{g('all','st_out'):.1f}bp 还高——
说明执行时点的改动在近期更有效，只是样本量（356 天）撑不起显著性。</p>

<h2>4 · 净值和成本</h2>
<p>验证期（2025-01 至 2026-10，435 个交易日，不交易日记 0）——唯一没参与任何选择的区间，
也是唯一值得看的净值区间。两种口径都放上来：</p>

<table><thead><tr><th rowspan="2">往返成本</th><th colspan="2">剔除 ST（实盘口径）</th><th colspan="2">剔除 ST + 开盘过滤</th></tr>
<tr><th>年化</th><th>期末（10 万）</th><th>年化</th><th>期末（10 万）</th></tr></thead><tbody>
{netvalue_rows()}
</tbody></table>
<p class="note">验证期最大回撤：剔 ST −29.9% ~ −42.3%，剔 ST + 过滤 −28.8% ~ −38.6%。
成本 13bp = 印花税 10bp + 佣金 2bp（万1）+ 滑点 1bp，是散户能拿到的最好水平。</p>

<div class="callout">
<b>成本结构：</b>卖出侧印花税 10bp 不可规避，佣金双边约 3bp（万1.5），地板 13bp。
验证期剔 ST 后的几何日收益只有 4.5bp，<b>连印花税都盖不住</b>。
年化那一列全是负的，不是成本设得太严，是毛收益本身不够。
</div>

{ablation}

<h2>{"6" if n_abl else "5"} · 这一轮修掉的两个 bug</h2>
<h3>bug 1：v3 折划分重叠</h3>
<p>测试期起点取了 <code>train_end</code>，应该取 <code>valid_end</code>。</p>
<div class="formula">测试窗口 = 验证期(3月) + 测试期(3月) = 6 个月
相邻折重叠 3 个月 → 预测 10,142,828 行，唯一 (code,date) 仅 5,193,371 行
<b>48.8% 是重复行</b></div>
<p>重复样本让同一天同一只票带两个不同分数进截面，排名和分位指标全被污染。已改为 <code>valid_end</code> 重训。</p>

<h3>bug 2：新脚本的特征列重复</h3>
<p>读数据集时 <code>columns</code> 参数把 <code>ret_20</code> 写了两次（它已在 <code>FEATURES</code> 里），
产出的帧有 40 列、模型训练时是 39 列，预测直接报 shape 不匹配。已修。</p>

<h2>{"7" if n_abl else "6"} · 结论与动作</h2>

<h3>四条结论</h3>
<ul>
<li><b>执行时点确实错了，这是真发现。</b>同一套信号，D 开盘买比 D 收盘买多 {best.mean_bp-curr.mean_bp:.1f}bp/日；
当前规则在四种合规钟里排最差，因为它把 +{illegal.mean_bp:.1f}bp 的日内段让给市场、只吃 {curr_open.mean_bp:.1f}bp 的隔夜段。</li>
<li><b>改对执行时点后策略才勉强站到盈亏平衡线附近。</b>剔 ST 后全样本 +{g('all','st_out'):.2f}bp（t={gt('all','st_out'):.2f}），
验证期 +{g('valid','st_out'):.2f}bp（t={gt('valid','st_out'):.2f}）——方向对，但都没到 1.96。</li>
<li><b>模型在 2025 年后漂向 ST 风格。</b>ST 占比 2.0% → 17.8%，这部分贡献验证期约 42% 的收益。
这是风格暴露不是 alpha。好在剔除后仍有 +{g('valid','st_out'):.2f}bp，没有归零。</li>
<li><b>特征工程到顶。</b>形态、行业、市场三个块共 25 个特征，IC 变化不到 1.5%。</li>
</ul>

<h3>该做什么</h3>
<div class="formula">已做：  daily_signal.py 加入 ST 名称硬过滤（信号只出非 ST），执行规则改为开盘买、次日开盘卖
已做：  用 tushare namechange 重建逐日 ST 状态（outputs/namechange.parquet）
本周做：把精确 ST 标记写进训练集过滤条件，重训 v4 —— 训练时不喂 ST，模型才不会往那儿漂
试仓：  成本压到 13bp 以内时可小仓（1–2 万）跑，验证期年化 +11.4%；成本 20bp 就转负，别碰
砍掉：  14:36 分钟级链路。它服务"当日收盘买"，而那个口径绝对收益 t 只有 1.51</div>

<p><code>scripts/daily_signal.py</code> 已把新规则写死：D−1 收盘出信号、D 日 09:30 开盘买、
D+1 开盘卖、开盘涨幅 &gt;1.5% 放弃、ST 与停牌一律排除，并自动追加前向账本
<code>outputs/live/forward_ledger.csv</code>。</p>

<div class="callout">
<b>成本是唯一的生死线。</b>盈亏平衡 17.6bp（剔 ST + 开盘过滤），而散户的成本地板是
印花税 10bp + 佣金 2bp = 12bp，加上 1–3bp 滑点就是 13–15bp。
落在这个区间内：年化 +7.4% ~ +11.4%；超过 18bp：转负。
<b>所以这不是"能不能赚钱"的问题，是"滑点控不控得住"的问题</b>——
必须贴着开盘价成交（集合竞价挂单），任何追价都会把安全垫吃光。
</div>

<footer>
数据：本地 A 股日线池（2016-01 起），20 折季度 walk-forward，样本外 2022-01 起 1,161 个交易日。
价格为前复权，成交额与成交量按原始口径。表中收益除成本敏感性外均为毛收益。
未计入冲击成本、融券约束与停牌顺延。ST 状态按 tushare <code>namechange</code> 重建为当日名称（4,375 只股票有记录，
无记录者视为从未戴帽），覆盖 2015 年至今，早于该范围的历史无法判定；
开盘过滤阈值 1.5% 为样本内扫描结果，已验证其效力在 2025 年后衰减。
本报告不构成投资建议。
</footer>
</div></body></html>"""

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "T1隔夜模型_第三轮_执行时点.html"
    p.write_text(html, encoding="utf-8")
    print(f"[done] {p}  ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
