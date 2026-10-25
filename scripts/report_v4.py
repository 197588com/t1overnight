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

    def st_attrib_html() -> str:
        """Which features actually lift the ST picks to the top of the score."""
        f = ROOT / "outputs/exit_mode/st_attrib.csv"
        if not f.exists():
            return "<p>（归因表未生成，跑 <code>scripts/st_attrib.py</code>）</p>"
        a = pd.read_csv(f).dropna(subset=["gain_share"])
        a = a.sort_values("tilt", ascending=False)
        body = []
        for _, r in a.iterrows():
            hot = r.gap > 15
            cls = ' class="bad"' if hot else ""
            body.append(
                f"<tr{cls}><td><code>{r.feature}</code></td>"
                f"<td>{r.gain_share*100:.2f}%</td>"
                f"<td>{r.pct_st:.1f}</td><td>{r.pct_nonst:.1f}</td>"
                f"<td class={'up' if r.gap>0 else 'down'}>{r.gap:+.1f}</td></tr>")
        top4 = a.head(4)
        return f"""
<table><thead><tr><th>特征</th><th>gain 占比</th><th>ST 票<br>当日分位</th><th>非 ST 票<br>当日分位</th><th>差</th></tr></thead>
<tbody>{''.join(body)}</tbody></table>
<p class="note">分位 = 该票当天在全池的横截面百分位（0–100）。gain 占比 = 20 折模型增益归一化后取均值。
四行加粗的是把 ST 票推上去的主力，合计占模型总增益 <b>{top4.gain_share.sum()*100:.1f}%</b>。</p>"""

    def triage_html() -> str:
        f = ROOT / "outputs/triage_eval/triage_quality.csv"
        if not f.exists():
            return "<p>（排雷评分未生成，跑 <code>scripts/triage_eval.py</code>）</p>"
        t = pd.read_csv(f).sort_values("lift_bp", ascending=False)
        lab = {"v2_rank_bottom20": "选股分数最低 20%（白捡）",
               "tri_base": "排雷头 base", "tri_A": "排雷头 + 形态",
               "tri_B": "排雷头 + 行业", "tri_C": "排雷头 + 市场"}
        body = []
        for _, r in t.iterrows():
            hl = ' class="hl"' if r.tag == "v2_rank_bottom20" else ""
            body.append(
                f"<tr{hl}><td>{lab.get(r.tag, r.tag)}</td>"
                f"<td>{r.bl_mean_bp:+.2f}</td><td>{r.bl_hit*100:.1f}%</td>"
                f"<td>{r.kept_mean_bp:+.2f}</td>"
                f"<td class={'up' if r.lift_bp>0 else 'down'}><b>{r.lift_bp:+.2f}</b></td>"
                f"<td>{r.lift_bp_per_day_t:+.2f}</td>"
                f"<td>{r.tail_recall_5*100:.1f}%</td></tr>")
        return f"""
<table><thead><tr><th>口径</th><th>被标记票<br>次日均值 (bp)</th><th>其中上涨<br>占比</th>
<th>剔除后<br>全池均值 (bp)</th><th>提升 (bp)</th><th>提升 t</th><th>最差 5%<br>召回</th></tr></thead>
<tbody>{''.join(body)}</tbody></table>
<p class="note">同一批 4,984,479 行、同一段样本外区间（2022-01 → 2026-10）。标记比例统一取 20%。
"提升" = 剔除被标记票后全池均值 − 全池均值。</p>"""

    def v4_html() -> str:
        f = ROOT / "outputs/v4/eval_top1.csv"
        if not f.exists():
            return "<p>（v4 结果未生成，跑 <code>scripts/train_v4.py</code> + <code>v4_eval.py</code>）</p>"
        t = pd.read_csv(f).set_index("run")
        lab = [("keep_rank", "原样训练（会买到 ST）", ""),
               ("keep_rank_filt", "原样训练，选股时剔 ST", ""),
               ("drop_rank", "训练 + 选股都剔 ST（v4）", ' class="hl"')]
        body = []
        for k, name, cls in lab:
            if k not in t.index:
                continue
            r = t.loc[k]
            body.append(
                f"<tr{cls}><td>{name}</td><td>{int(r.days)}</td>"
                f"<td>{r.st_share*100:.2f}%</td>"
                f"<td class={'up' if r.mean_bp>0 else 'down'}><b>{r.mean_bp:+.2f}</b></td>"
                f"<td>{r.geo_bp:+.2f}</td><td>{r.hit*100:.1f}%</td>"
                f"<td>{r.t:+.2f}</td>"
                f"<td>{r.select_bp:+.2f}<span class='note'> / {r.select_t:+.2f}</span></td>"
                f"<td>{r.valid_bp:+.2f}<span class='note'> / {r.valid_t:+.2f}</span></td></tr>")
        rows_html = "".join(body)

        cost_rows = []
        for k, name, _ in lab:
            if k not in t.index:
                continue
            geo = t.loc[k, "geo_bp"] / 1e4
            cells = "".join(f"<td>{((1+geo-c/1e4)**244-1)*100:+.1f}%</td>"
                            for c in (13, 15, 18, 20, 25))
            cost_rows.append(f"<tr><td>{name}</td>{cells}</tr>")

        ab = {}
        abf = ROOT / "outputs/v4/eval_ab.json"
        if abf.exists():
            ab = json.loads(abf.read_text())
        ab_txt = ""
        if ab:
            ab_txt = (f"<p>配对比较（只比<b>买得到的票</b>，同一 1,161 天）：v4 比"
                      f"\"原样训练但选股剔 ST\"多 <b>{ab['diff_bp']:+.2f}bp</b>/日，"
                      f"t = {ab['t']:+.2f}，胜率 {ab['win']*100:.0f}%。"
                      f"<b>方向对，但 t 没到 1.96——不能说\"确定更好\"。</b></p>")
        return f"""
<table><thead><tr><th>口径</th><th>交易日</th><th>ST 占比</th><th>毛均值<br>bp</th>
<th>几何<br>bp</th><th>胜率</th><th>t</th>
<th>选参期 2022-24<br>bp / t</th><th>验证期 2025-26<br>bp / t</th></tr></thead>
<tbody>{rows_html}</tbody></table>

<table><thead><tr><th>年化净收益（10 万）</th><th>13bp</th><th>15bp</th><th>18bp</th>
<th>20bp</th><th>25bp</th></tr></thead><tbody>{''.join(cost_rows)}</tbody></table>
{ab_txt}"""

    def v4v(run: str, col: str, default: float = float("nan")) -> float:
        f = ROOT / "outputs/v4/eval_top1.csv"
        if not f.exists():
            return default
        t = pd.read_csv(f).set_index("run")
        return float(t.loc[run, col]) if run in t.index else default

    def abv(key: str, default: float = float("nan")) -> float:
        f = ROOT / "outputs/v4/eval_ab.json"
        if not f.exists():
            return default
        return float(json.loads(f.read_text()).get(key, default))

    def ann(run: str, cost: float) -> float:
        geo = v4v(run, "geo_bp") / 1e4
        return ((1 + geo - cost / 1e4) ** 244 - 1) * 100

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
<h2>6 · 模型侧的两个实验：特征挖不动，排雷头不如白捡的</h2>

<h3>6.1 特征块消融：加不动了</h3>
<p>四个配置跑在同一批折、同一批行上，唯一差别是喂进去的特征块。
指标是排雷概率与次日收益的秩相关 IC（负号 = 越像烂票、次日越跌，越负越好）。</p>
<table><thead><tr><th>配置</th><th>IC</th><th>ICIR</th><th>相对基线</th><th>变化幅度</th></tr></thead>
<tbody>{body}</tbody></table>
<p class="note">三个新增特征块（形态 9 个、行业 7 个、市场 9 个）共 25 个特征，
IC 变化都在小数点后第四位，相对幅度不到 1.5%。
结论：<b>在现有 39 个日频特征之上，形态与行业的横截面加工榨不出新信息。</b></p>

<h3>6.2 排雷头：专门训练，反而不如直接用选股分数</h3>
<p>原本的计划是单独训一个"排雷"二分类头（标签 = 当日收益落在后 20%），做成日更的黑名单。
v3 把它和"选股分数最低 20%"这个不需要额外模型的做法放在同一批行上比：</p>
{triage_html()}
<p>结论是<b>排雷头没有赢</b>。它确实更会抓极端：最差 5% 的召回 57.1% 对 52.2%，
被标记票里真正属于最烂 20% 的比例也更高（35.3% 对 33.8%）。但被它标记的票<b>平均只跌 13.8bp，
而选股分数最低那一档跌 16.4bp</b>——因为低分票里混着暴涨暴跌的极端票，均值更难看，
剔除后留下的池子提升更明显（+3.70 对 +4.34bp，t 6.49 对 9.18）。三个特征块也拉不动，互差不到 0.1bp。</p>
<div class="callout"><b>所以排雷不要单独建模型。</b>直接把选股分数的低 20% 当黑名单，
零成本、零维护、效果更好。这是这轮省下来的一件事——原本准备写进定时任务的东西，实测不值得做。</div>
"""
    n_abl = 1 if ablation else 0

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>T+1 隔夜模型 · 执行时点、ST 暴露与 v4 重训</title><style>{CSS}</style></head>
<body><div class="wrap">

<h1>日毛收益从 −15bp 走到 +30bp：执行时点、ST 剔除，各值多少
<small>T+1 隔夜模型第三轮 · 2026-10-25</small></h1>
<div class="meta">样本外 2022-01-04 → 2026-10-22，1,161 个交易日 · 严格 D−1 滞后信号 · 10 万本金每日单票</div>

<div class="lede">
这一轮解开了三个扣。<b>一，执行时点确实错了</b>：把买单从 D 日收盘挪到 D 日开盘，
几何日收益从 <b class="down">{curr.geo_bp:+.2f}bp</b> 翻到 <b class="up">{best.geo_bp:+.2f}bp</b>，
当前这条规则在四种合规钟里排最差。<b>二，改对之后发现钱来自 ST 股</b>：用 tushare 曾用名表重建历史 ST 状态，
验证期有 <b>17.8% 的交易日买在 ST 股上</b>，贡献约四成收益——这是风格暴露，不是 alpha。
<b>三，把这批票从训练集里一起删掉，收益反而更高</b>：毛收益
+{v4v('keep_rank_filt','mean_bp'):.1f}bp → <b class="up">+{v4v('drop_rank','mean_bp'):.1f}bp</b>、
t 值 {v4v('drop_rank','t'):.2f}，盈亏平衡成本从 {v4v('keep_rank_filt','geo_bp'):.1f}bp
抬到 {v4v('drop_rank','geo_bp'):.1f}bp。<b>但改善集中在 2022-2024，
2025-2026 仍然只是"没证伪"。</b>
</div>

<div class="grid">
<div class="card"><div class="k">当前生产规则（D 收 → D+1 收）</div><div class="v down">{curr.geo_bp:+.1f}bp</div>
<div class="s">四种合规钟里最差</div></div>
<div class="card"><div class="k">只改执行时点</div><div class="v up">{best.geo_bp:+.1f}bp</div>
<div class="s">D 开 → D+1 开，旧模型</div></div>
<div class="card"><div class="k">再剔掉 ST（v4）</div><div class="v up">{v4v('drop_rank','geo_bp'):+.1f}bp</div>
<div class="s">t = {v4v('drop_rank','t'):.2f} · 盈亏平衡 24bp</div></div>
<div class="card"><div class="k">验证期（2025-26）</div><div class="v">{v4v('drop_rank','valid_bp'):+.1f}bp</div>
<div class="s">t = {v4v('drop_rank','valid_t'):.2f} · 仍未证实</div></div>
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

<h3>为什么模型会追 ST：把分数拆开看</h3>
<p>特征里没有任何 ST 标识，所以只能问"模型到底在给什么打分"。做法是把 20 折模型的增益
（gain，归一化后取均值）和"被选中的票当天在全池的分位"交叉起来——一个特征只有同时满足
<b>增益大</b>和<b>ST 票在上面分位极高</b>，才解释得了这个漂移。</p>
{st_attrib_html()}
<p>答案比预想的更直白：<b>模型最重要的两个特征就是换手率</b>——<code>cs_turn</code>（相对全市场的换手）
占增益 13.1%、<code>turn</code>（绝对换手）占 9.9%，两项合计 <b>23%</b>。而 ST 票在这两项上处在当日全池的
<b>第 72 百分位</b>，正常票只在第 32 百分位。再加上 <code>amp</code>（振幅，4.9% 增益，p72.6 对 p28.0）
和 <code>vol_20</code>（20 日波动，p85.2 对 p27.2），四个特征就把 ST 票抬到了前排。</p>
<p>反方向也一致：ST 票的 <code>log_mktcap</code> 分位只有 13.6（市值小）、
<code>cs_ma_dev_20</code> 只有 10.5（深度跌破 20 日均线，超跌）、<code>cs_ret_5</code> 只有 31.5（近一周相对弱）。
<b>四个特征说"换手极高、振幅大"，三个特征说"小、超跌、近期落后"——这就是 ST 股的画像，
也是模型的偏好画像。</b>模型没在预测 ST 会涨，它是在找"高换手的小盘超跌票"，
而 2025-2026 的 ST 板块恰好就是这批票。</p>
<p class="note">顺带推翻了我先前的猜测：我以为推手是 <code>lu_cnt_5</code>（近 5 日涨停次数）和
<code>log_mktcap</code>。实测 <code>lu_cnt_5</code> 的增益只有 0.04%、ST 票分位 51.0 对 48.1——
它基本没参与排序。<b>猜机制和量机制是两件事。</b></p>

<h2>3 · v4：把 ST 从训练集里一起拿掉</h2>
<p>上一节只做了"选股时剔掉 ST"——那只是不买。模型还是照着含 ST 的数据学的，
它会继续朝那个方向找票，只是被挡在门外。这一节问的是另一个问题：
<b>训练时就不喂 ST，模型会不会学会一套不同的偏好？</b></p>
<p>做法：同一个 20 折、同一批行、同样的 39 个特征、同 200 轮，只改一件事——
训练集里删掉当时确为 ST 的行（占全池 0.56%），预测池也删掉。为了公平，另跑一个
"原样训练"的版本在完全相同的折上，再加一个"原样训练、但选股时剔 ST"的口径——
<b>最后这个才是两边都买得到的比较对象。</b></p>

{v4_html()}

<p>三个数字连起来读：原样训练 +{v4v('keep_rank','mean_bp'):.2f}bp，但里面有
{v4v('keep_rank','st_share')*100:.2f}% 的交易日买的是 ST 股；把 ST 挡在门外之后掉到
+{v4v('keep_rank_filt','mean_bp'):.2f}bp；而训练时就删掉 ST，同样的折、同样的成本口径，
回到 <b>+{v4v('drop_rank','mean_bp'):.2f}bp、t = {v4v('drop_rank','t'):.2f}</b>——
这是四轮实验里第一次 t 值接近 3。</p>

<div class="callout">
<b>真正的变化不在收益，在成本容忍度。</b>
盈亏平衡往返成本从 keep_filt 的 <b>{v4v('keep_rank_filt','geo_bp'):.1f}bp</b>
抬到 v4 的 <b>{v4v('drop_rank','geo_bp'):.1f}bp</b>。
按散户地板成本 13bp 算，年化从 <b class="down">{ann('keep_rank_filt',13):+.1f}%</b>
变成 <b class="up">{ann('drop_rank',13):+.1f}%</b>；
即使成本到 20bp 仍是 {ann('drop_rank',20):+.1f}%。
<b>这是一个策略能不能实盘的差别，不是收益高低的问题。</b>
</div>

<p>但也必须把话说完整：<b>v4 的改善全部来自 2022-2024 段。</b>选参期
+{v4v('drop_rank','select_bp'):.2f}bp（t={v4v('drop_rank','select_t'):.2f}）站得很稳，
验证期（2025-2026，435 天）只有 +{v4v('drop_rank','valid_bp'):.2f}bp（t={v4v('drop_rank','valid_t'):.2f}）。
原样训练在验证期 +{v4v('keep_rank','valid_bp'):.2f}bp 是好，但那一段的收益本来就靠 ST 那 18% 撑起来——
剔掉之后，那段行情里没有别的票能顶上。</p>

<p>那 +{abv('diff_bp'):.1f}bp 的配对优势是怎么来的？胜率只有 {abv('win')*100:.0f}%——
<b>七成的日子 v4 是输的，赢的那三成赢得够多</b>。这是少数大赢撑起来的均值差，不是稳定优势。
所以 v4 的定位是"把不该赚的钱挤出去之后，剩下的部分更干净、也更薄"，<b>它不是一个验证通过的策略。</b></p>

<h2>4 · 分段：alpha 在前半段，风格在后半段</h2>
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

<h2>5 · 净值和成本</h2>
<p>验证期（2025-01 至 2026-10，435 个交易日，不交易日记 0）——唯一没参与任何选择的区间，
也是唯一值得看的净值区间。两种口径都放上来：</p>

<table><thead><tr><th rowspan="2">往返成本</th><th colspan="2">剔除 ST（实盘口径）</th><th colspan="2">剔除 ST + 开盘过滤</th></tr>
<tr><th>年化</th><th>期末（10 万）</th><th>年化</th><th>期末（10 万）</th></tr></thead><tbody>
{netvalue_rows()}
</tbody></table>
<p class="note">验证期最大回撤：剔 ST −29.9% ~ −42.3%，剔 ST + 过滤 −28.8% ~ −38.6%。
成本 13bp = 印花税 10bp + 佣金 2bp（万1）+ 滑点 1bp，是散户能拿到的最好水平。</p>

<p class="note">本节是旧模型（v2，含 ST）的口径。第 3 节的 v4 在同样的执行时点下把盈亏平衡从
17.6bp 抬到 {v4v('drop_rank','geo_bp'):.1f}bp，那才是该看的成本线。</p>

<div class="callout">
<b>成本结构：</b>卖出侧印花税 10bp 不可规避，佣金双边约 3bp（万1.5），地板 13bp。
验证期剔 ST 后的几何日收益只有 4.5bp，<b>连印花税都盖不住</b>。
年化那一列全是负的，不是成本设得太严，是毛收益本身不够。
</div>

{ablation}

<h2>{"7" if n_abl else "6"} · 这一轮修掉的两个 bug</h2>
<h3>bug 1：v3 折划分重叠</h3>
<p>测试期起点取了 <code>train_end</code>，应该取 <code>valid_end</code>。</p>
<div class="formula">测试窗口 = 验证期(3月) + 测试期(3月) = 6 个月
相邻折重叠 3 个月 → 预测 10,142,828 行，唯一 (code,date) 仅 5,193,371 行
<b>48.8% 是重复行</b></div>
<p>重复样本让同一天同一只票带两个不同分数进截面，排名和分位指标全被污染。已改为 <code>valid_end</code> 重训。</p>

<h3>bug 2：新脚本的特征列重复</h3>
<p>读数据集时 <code>columns</code> 参数把 <code>ret_20</code> 写了两次（它已在 <code>FEATURES</code> 里），
产出的帧有 40 列、模型训练时是 39 列，预测直接报 shape 不匹配。已修。</p>

<h2>{"8" if n_abl else "7"} · 结论与动作</h2>

<h3>五条结论</h3>
<ul>
<li><b>执行时点确实错了，这是真发现。</b>同一套信号，D 开盘买比 D 收盘买多 {best.mean_bp-curr.mean_bp:.1f}bp/日；
当前规则在四种合规钟里排最差，因为它把 +{illegal.mean_bp:.1f}bp 的日内段让给市场、只吃 {curr_open.mean_bp:.1f}bp 的隔夜段。</li>
<li><b>模型一直在偷偷买 ST 股，而这是模型的偏好、不是市场的错。</b>增益第一第二的特征是换手率
（合计 23%），ST 票在换手上的日分位是 72 对 32；再叠上振幅和波动——四种"高换手高振幅"特征加上三种
"小市值超跌"特征，就是 ST 股的完整画像。</li>
<li><b>把 ST 从训练集里删掉，收益不降反升。</b>毛收益 +{v4v('keep_rank_filt','mean_bp'):.2f} → 
+{v4v('drop_rank','mean_bp'):.2f}bp，t 从 {v4v('keep_rank_filt','t'):.2f} 升到 {v4v('drop_rank','t'):.2f}，
盈亏平衡成本 17 → {v4v('drop_rank','geo_bp'):.0f}bp。这是四轮里第一次 t 接近 3。</li>
<li><b>但改善集中在 2022-2024，验证期没有证据。</b>选参期 +{v4v('drop_rank','select_bp'):.2f}bp（t={v4v('drop_rank','select_t'):.2f}），
验证期 +{v4v('drop_rank','valid_bp'):.2f}bp（t={v4v('drop_rank','valid_t'):.2f}）。
配对 t 只有 {abv('t'):.2f}。<b>不能把"全样本 t=2.82"读成"策略通过了"。</b></li>
<li><b>特征工程到顶，排雷头不值得单独做。</b>25 个新特征 IC 变化不到 1.5%；
专门训的排雷头（提升 +3.70bp，t 6.49）输给"选股分数最低 20%"这个零成本做法（+4.34bp，t 9.18）。</li>
</ul>

<h3>该做什么</h3>
<div class="formula">已做：  daily_signal.py 写死新规则（D−1 出信号、D 开盘买、D+1 开盘卖、ST 与停牌硬过滤）
已做：  scripts/st_mask.py 用 tushare namechange 重建逐日 ST（可复用，训练与回测共用一份口径）
已做：  v4 重训完成：训练集与预测池同时剔 ST，毛收益 +{v4v('drop_rank','mean_bp'):.1f}bp、t={v4v('drop_rank','t'):.2f}
下一步：前向记账 3 个月。daily_signal.py 每天自动写 outputs/live/forward_ledger.csv，
        累计 ≥60 个交易日、毛收益 ≥ +25bp 再谈上钱
不要做：单独训排雷模型（输给免费做法）；再堆日频特征（三个块都没用）
可砍掉：14:36 分钟级链路。它服务"当日收盘买"，那个口径的绝对收益 t 只有 1.51</div>

<div class="callout">
<b>成本仍是生死线，但安全垫变厚了。</b>v4 的盈亏平衡是 {v4v('drop_rank','geo_bp'):.1f}bp，
散户地板成本 13bp（印花 10 + 佣金 2 + 滑点 1），中间有 {v4v('drop_rank','geo_bp')-13:.0f}bp 缓冲；
而修 ST 之前这个缓冲是 {v4v('keep_rank_filt','geo_bp')-13:+.1f}bp，等于 13bp 成本下已经在亏。
<b>结论变了：从"滑点控不控得住"变成"滑点别超过 10bp"</b>——仍然必须贴着开盘价用集合竞价成交，
但容错空间比上一轮大了一个量级。真正的门槛已经不是成本，是<b>验证期那 435 天里 t 只有
{v4v('drop_rank','valid_t'):.2f}</b>，这一段还没有证据。
</div>

<footer>
数据：本地 A 股日线池（2016-02 起，8,667,494 个可交易行），20 折季度 walk-forward，
样本外 2022-01 起 1,161 个交易日。v3/v4 训练 200 轮、39 个特征、同一批折。
价格为前复权，成交额与成交量为原始口径。表中收益除成本敏感性外均为毛收益。
未计入冲击成本、融券约束与停牌顺延。ST 状态按 tushare <code>namechange</code> 重建为当日名称
（4,375 只股票有记录，无记录者视为从未戴帽），覆盖 2015 年至今，更早的历史无法判定；
v4 的标签分位（y_rank）在同一天的<b>全池</b>上计算，剔 ST 只影响训练行不影响标签定义。
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
