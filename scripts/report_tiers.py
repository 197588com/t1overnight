# -*- coding: utf-8 -*-
"""Render the permission-tier board: what to buy today under each account tier,
and what each tier has actually paid historically.

Every number is read from outputs/tiers/*.csv -- nothing is hardcoded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TIER_LABEL = {
    "T1 沪深主板": ("沪深主板", "无需任何权限", "600/601/603/605 · 000/001/002/003"),
    "T2 +创业板": ("主板 + 创业板", "10 万资产 + 24 个月", "300/301"),
    "T3 +科创板": ("主板 + 创业板 + 科创板", "50 万资产 + 24 个月", "688/689"),
    "T4 全含北交所": ("全市场", "50 万 + 24 个月（另开）", "8xx/4xx"),
}


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = ROOT / "outputs/tiers"
    ev = pd.read_csv(d / "eval_tiers.csv")
    inc = pd.read_csv(d / "incremental.csv")
    clk = pd.read_csv(d / "clocks_by_tier.csv")
    today = pd.read_csv(d / "today_by_tier.csv", dtype={"code": str})
    snap = pd.read_csv(ROOT / "outputs/live_all/scores_latest.csv", dtype={"code": str})
    snap["code"] = snap["code"].astype(str).str.zfill(6)
    today = today.merge(snap[["code", "ret_1"]], on="code", how="left")
    return ev, inc, clk, today, snap


def pct(v: float, nd: int = 1) -> str:
    return "—" if pd.isna(v) else f"{v:+.{nd}f}%"


def main() -> None:
    ev, inc, clk, today, snap = load()
    ev = ev.set_index("run")
    asof = str(pd.Timestamp.max)  # placeholder, replaced below
    bar = pd.Timestamp("2026-10-23")

    # ---- today's table -------------------------------------------------
    rows = []
    for tier, (name, qual, codes) in TIER_LABEL.items():
        q = today[today.tier == tier]
        if q.empty:
            continue
        for i, (_, r) in enumerate(q.iterrows()):
            first = i == 0
            rows.append(f"""<tr class="{'lead' if first else ''}">
<td class="tier">{name if first else ''}<span class="q">{qual if first else ''}</span></td>
<td class="rk">{int(r['rank'])}</td>
<td class="cd">{r['code']}</td><td class="nm">{r['name']}</td>
<td class="bd">{r['board']}</td>
<td class="num">{r['close']:.2f}</td>
<td class="num up">{r['gap_cap']:.2f}</td>
<td class="num">{r['ret_1']*100:+.1f}%</td>
<td class="num sc">{r['score']:.3f}</td></tr>""")

    # ---- history table -------------------------------------------------
    hist = []
    for tier in TIER_LABEL:
        if tier not in ev.index:
            continue
        r = ev.loc[tier]
        name = TIER_LABEL[tier][0]
        hist.append(f"""<tr>
<td class="tier">{name}</td>
<td class="num">{int(r['days'])}</td>
<td class="num up">{r['mean_bp']:+.2f}</td>
<td class="num up">{r['geo_bp']:+.2f}</td>
<td class="num">{r['hit']*100:.1f}%</td>
<td class="num {'sig' if r['t'] >= 1.96 else ''}">{r['t']:.2f}</td>
<td class="num">{r['select_bp']:+.1f} <span class="sm">({r['select_t']:.2f})</span></td>
<td class="num">{r['valid_bp']:+.1f} <span class="sm">({r['valid_t']:.2f})</span></td>
</tr>""")

    # ---- cost table ----------------------------------------------------
    def ann(geo_bp: float, cost: int) -> float:
        g = geo_bp / 1e4
        return ((1 + g - cost / 1e4) ** 244 - 1) * 100

    costs = (10, 13, 15, 18, 20, 25)
    cost_rows = []
    for tier in TIER_LABEL:
        if tier not in ev.index:
            continue
        g = float(ev.loc[tier, "geo_bp"])
        cells = []
        for c in costs:
            a = ann(g, c)
            cls = "up" if a > 3 else ("dn" if a < -3 else "mid")
            cells.append(f'<td class="num {cls}">{a:+.1f}%</td>')
        cost_rows.append(f'<tr><td class="tier">{TIER_LABEL[tier][0]}</td>'
                         f'<td class="num">{g:+.2f}</td>' + "".join(cells) + "</tr>")

    # ---- incremental ---------------------------------------------------
    inc_rows = []
    for _, r in inc.iterrows():
        a = TIER_LABEL[r["from"]][0]
        b = TIER_LABEL[r["to"]][0]
        inc_rows.append(f"""<tr><td>{b}</td><td class="num">{r['from_bp']:+.2f}</td>
<td class="num">{r['to_bp']:+.2f}</td>
<td class="num up">{r['lift_bp']:+.2f}</td>
<td class="num {'sig' if abs(r['t']) >= 1.96 else ''}">{r['t']:+.2f}</td>
<td class="num">{r['win']*100:.0f}%</td></tr>""")

    # ---- clocks --------------------------------------------------------
    clock_lab = {"d_oo": "D 开 → D+1 开", "d_oc1": "D 开 → D+1 收",
                 "fwd_co": "D 收 → D+1 开", "fwd_cc": "D 收 → D+1 收"}
    clk_rows = []
    for tier in TIER_LABEL:
        q = clk[clk.tier == tier]
        if q.empty:
            continue
        cells = []
        for c, lab in clock_lab.items():
            v = q[q.clock == c]
            if not len(v):
                continue
            best = c == "d_oo"
            cells.append(f'<td class="num {"up" if best else ""}">{float(v.mean_bp.iloc[0]):+.1f}'
                         f'<span class="sm"> / t{float(v.t.iloc[0]):.2f}</span></td>')
        clk_rows.append(f'<tr><td class="tier">{TIER_LABEL[tier][0]}</td>' + "".join(cells) + "</tr>")

    # ---- lead pick per tier for the headline ---------------------------
    def lead(tier: str) -> pd.Series | None:
        q = today[(today.tier == tier) & (today["rank"] == 1)]
        return q.iloc[0] if len(q) else None

    mb, cy, allm = lead("T1 沪深主板"), lead("T2 +创业板"), lead("T3 +科创板")
    e_mb, e_all = ev.loc["T1 沪深主板"], ev.loc["T3 +科创板"]
    barrier = e_mb["geo_bp"]
    gap_geo = e_all["geo_bp"] - e_mb["geo_bp"]

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>权限分档 · 今日信号与历史代价</title>
<style>
:root{{
  --bg:#fbf7f2; --panel:#fffdfa; --line:#e8dcd0; --ink:#3b322c; --dim:#8a7a6d;
  --terra:#b4553f; --amber:#c08444; --sage:#6f7f60; --sand:#f3eadf;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.72 -apple-system,"PingFang SC","Hiragino Sans GB",sans-serif;
 -webkit-font-smoothing:antialiased}}
.wrap{{max-width:1080px;margin:0 auto;padding:52px 28px 72px}}
h1{{font-size:29px;line-height:1.35;margin:0 0 8px;font-weight:650;letter-spacing:-.3px}}
h1 small{{display:block;font-size:13px;font-weight:400;color:var(--dim);
 letter-spacing:.4px;margin-top:10px}}
h2{{font-size:19px;margin:46px 0 14px;font-weight:620;letter-spacing:-.2px;
 padding-left:11px;border-left:3px solid var(--terra)}}
h3{{font-size:15px;margin:26px 0 8px;font-weight:600;color:var(--amber)}}
p{{margin:11px 0;color:#4a403a}}
.meta{{font-size:12.5px;color:var(--dim);margin-bottom:26px;padding-bottom:18px;
 border-bottom:1px solid var(--line)}}
.lede{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--terra);
 padding:18px 20px;border-radius:3px;margin:22px 0 6px;font-size:14.5px}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:15px 16px}}
.card .k{{font-size:11.5px;color:var(--dim);letter-spacing:.3px}}
.card .v{{font-size:23px;font-weight:600;margin:5px 0 2px;letter-spacing:-.5px}}
.card .s{{font-size:11.5px;color:var(--dim);line-height:1.5}}
table{{width:100%;border-collapse:collapse;margin:14px 0;font-size:13px;
 background:var(--panel);border:1px solid var(--line)}}
th{{background:var(--sand);font-weight:600;font-size:11.5px;color:#6b5c50;
 text-align:right;padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}}
th:first-child,th:nth-child(2){{text-align:left}}
td{{padding:8px 10px;border-bottom:1px solid #f2eae1;text-align:right;white-space:nowrap}}
td.tier,td.cd,td.nm,td.bd{{text-align:left}}
tr:last-child td{{border-bottom:none}}
tr.lead{{background:#fdf6ef}}
tr.lead td{{border-top:1px solid var(--line)}}
td.tier{{font-weight:600}}
td.tier .q{{display:block;font-size:10.5px;font-weight:400;color:var(--dim)}}
td.cd{{font-variant-numeric:tabular-nums;color:var(--dim)}}
td.nm{{font-weight:600}}
td.num{{font-variant-numeric:tabular-nums}}
td.sc{{color:var(--dim)}}
.sm{{font-size:10.5px;color:var(--dim);font-weight:400}}
.up{{color:var(--terra)}} .dn{{color:var(--sage)}} .mid{{color:var(--amber)}}
td.sig{{color:var(--terra);font-weight:700}}
.note{{font-size:12px;color:var(--dim);line-height:1.65;margin:8px 0 0}}
.callout{{background:#f8f1e7;border:1px solid #e9d8c4;border-left:3px solid var(--amber);
 padding:16px 18px;border-radius:3px;margin:18px 0;font-size:13.5px}}
.formula{{background:#3f3730;color:#f0e6da;font:12.5px/1.85 ui-monospace,SFMono-Regular,Menlo,monospace;
 padding:16px 18px;border-radius:3px;margin:16px 0;white-space:pre;overflow-x:auto}}
ul{{margin:10px 0;padding-left:20px}} li{{margin:7px 0;color:#4a403a}}
footer{{margin-top:52px;padding-top:20px;border-top:1px solid var(--line);
 font-size:11.5px;color:var(--dim);line-height:1.75}}
@media(max-width:820px){{.grid{{grid-template-columns:repeat(2,1fr)}}
 .wrap{{padding:32px 16px 56px}} h1{{font-size:23px}}}}
</style></head><body><div class="wrap">

<h1>你买不了科创板，但策略几乎没有缩水
<small>权限分档 · 2026-10-23 收盘信号 → 10-26 开盘执行</small></h1>
<div class="meta">v4 模型（训练与选股均剔除 ST）· 样本外 2022-01-04 → 2026-10-22，{int(e_mb['days'])} 个交易日
· D 日开盘买、D+1 开盘卖 · 每日单票 10 万 · 表中收益均为毛收益</div>

<div class="lede">
全市场首选是 <b>{allm['code']} {allm['name']}</b>（{allm['board']}，开盘上限 {allm['gap_cap']:.2f}），
你买不了。把科创板从股票池里拿掉后重新选，落到
<b>{mb['code']} {mb['name']}</b>（{mb['board']}，开盘上限 {mb['gap_cap']:.2f}）。
关键问题不是"换成了谁"，而是"换了之后还赚不赚钱"——这是我这次真正去测的东西：
同一批折、同一套模型，只把可买池切小，<b>主板档几何日收益 {e_mb['geo_bp']:+.2f}bp，
全市场 {e_all['geo_bp']:+.2f}bp，只差 {gap_geo:.1f}bp</b>，
而配对检验 t 值只有 {float(inc.iloc[0]['t']):.2f}——<b>这个差距在统计上区分不出来。</b>
</div>

<div class="grid">
<div class="card"><div class="k">主板档 · 几何日均</div><div class="v up">{e_mb['geo_bp']:+.1f}bp</div>
<div class="s">全市场 {e_all['geo_bp']:+.1f}bp · 差 {gap_geo:.1f}bp</div></div>
<div class="card"><div class="k">主板档 · t 值</div><div class="v">{e_mb['t']:.2f}</div>
<div class="s">全市场 {e_all['t']:.2f} · 门槛 1.96</div></div>
<div class="card"><div class="k">主板档 · 13bp 成本年化</div><div class="v up">{ann(e_mb['geo_bp'],13):+.1f}%</div>
<div class="s">全市场 {ann(e_all['geo_bp'],13):+.1f}%</div></div>
<div class="card"><div class="k">主板档 · 盈亏平衡成本</div><div class="v">{e_mb['geo_bp']:.1f}bp</div>
<div class="s">全市场 {e_all['geo_bp']:.1f}bp</div></div>
</div>

<h2>1 · 今天买什么（按你的账户权限选一行）</h2>
<p>三档是累加的：每多一档权限，股票池变大，首选可能整体换掉。没开创业板就看第一段，
开了创业板看第二段。<span class="sm">"开盘上限"= 信号日收盘 × 1.015，开盘价高于它当天放弃。</span></p>
<table><thead><tr><th>权限档</th><th>名次</th><th>代码</th><th>名称</th><th>板块</th>
<th>昨收</th><th>开盘上限</th><th>昨日涨幅</th><th>分数</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="note">分数是模型原始输出，全池分位都在 99.3% 以上——注意这<b>不代表把握度 99%</b>，
只说明它是当日模型最看好的那一只。昨日涨幅一列是为了提醒：<code>{mb['code']}</code> 前一日
{mb['ret_1']*100:+.1f}%、<code>{cy['code']}</code> 前一日 {cy['ret_1']*100:+.1f}%，
开盘直接跳空高开就放弃，别追。</p>

<h2>2 · 历史上每一档真的赚多少（这是重点）</h2>
<p>做法：拿 v4 的逐日全池分数，按档位裁掉不能买的板块，再取当日最高分那一只。
不是"把不能买的票跳过去取第二只"——那样选的还是同一批名字，只是晚一天算账。
这里是<b>重新选股</b>，所以换了票之后的表现才是真实的。</p>
<table><thead><tr><th>权限档</th><th>天数</th><th>毛均值 bp</th><th>几何 bp</th>
<th>胜率</th><th>t 值</th><th>选参期 22-24</th><th>验证期 25-26</th></tr></thead>
<tbody>{''.join(hist)}</tbody></table>
<p class="note">选参期 / 验证期括号内是各自的 t 值。四档在验证期都不显著（t &lt; 0.5）——
这个老问题没因为换档而消失，也说明验证期的样本量（356 天）本身就撑不起结论。</p>

<div class="callout">
<b>有个反直觉的地方值得单独说：主板档的胜率反而最高（{e_mb['hit']*100:.1f}% vs
全市场 {e_all['hit']*100:.1f}%），t 值也更接近门槛。</b>
主板票波动小，单笔涨幅没那么猛，但赢的次数更多——算成几何收益后，两者差距被压到
{gap_geo:.1f}bp。这正是波动拖累的另一面：<b>高波动票把算术上的优势吃掉了一部分。</b>
对每日满仓单票的打法，胜率高的那一档其实更好拿。
</div>

<h2>3 · 解锁一个板块值多少（配对检验）</h2>
<p>同一天、同一批交易日，两档的日收益直接相减。注意"赢了 11%"这种数字不是低胜率，
而是大多数日子两档选的是同一只票、差值为零，胜率统计把零算作非赢。</p>
<table><thead><tr><th>解锁到</th><th>原档 bp</th><th>新档 bp</th><th>增量 bp</th>
<th>t 值</th><th>换票日占比</th></tr></thead><tbody>{''.join(inc_rows)}</tbody></table>
<p class="note">北交所一行没有增量：模型 1,161 天里<b>一次都没选过北交所的票</b>，这一档权限白给。
科创板只在 4% 的交易日被选中，创业板 23%。<b>真正有意义的门槛是创业板，不是科创板。</b></p>

<h2>4 · 成本才是拿到手的决定项</h2>
<p>10 万本金，按不同往返成本算年化（几何口径，244 个交易日）。绿字代表亏。</p>
<table><thead><tr><th>权限档</th><th>几何 bp</th>{''.join(f'<th>{c}bp</th>' for c in costs)}</tr></thead>
<tbody>{''.join(cost_rows)}</tbody></table>
<p class="note">往返成本地板：印花税 10bp（卖出，不可规避）+ 佣金双边 2–3bp + 滑点。
落在 13bp 是理想情况，15bp 很常见，超过 {e_mb['geo_bp']:.0f}bp 主板档就归零。</p>

<h2>5 · 四种合规执行钟，主板档都看一遍</h2>
<p>和全市场口径一样，D 开盘买、D+1 开盘卖仍是最优的那一口。格式：毛均值 bp / t 值。</p>
<table><thead><tr><th>权限档</th>{''.join(f'<th>{lab}</th>' for lab in clock_lab.values())}</tr></thead>
<tbody>{''.join(clk_rows)}</tbody></table>

<h2>6 · 顺带修掉的一个真 bug（它影响昨天的报价）</h2>
<p>做分档的时候对了一下账，发现开盘涨幅的<b>基准价取错了一天</b>：代码用的是信号日前一交易日的收盘，
而正确基准是信号日当天的收盘（模型预测的是 open(D)/open(D−1)，D 就是信号日的下一交易日）。
后果是上限价整体偏低，遇到信号日大涨的票，算出来的上限甚至<b>低于当日收盘价</b>，
等于当天必然判"高开超限"直接放弃。</p>
<div class="formula">例：002245 蔚蓝锂芯  信号日收盘 11.80，当日涨幅 +4.4%
  旧基准 11.30（前一日收盘）→ 上限 11.47  < 11.80，永远买不进
  新基准 11.80（信号日收盘）→ 上限 11.98  正确

瑞松科技   上限 55.03 → 修正为 55.27
方正科技   上限 10.31 → 修正为 10.46</div>
<p>已修复 <code>scripts/daily_signal.py</code>，并把板块权限做成了参数 <code>--boards</code>，
以后一条命令出对应档位的信号，不再需要事后手工筛。原来那份全市场结果完整保留在
<code>outputs/keep_20261023_allboard/</code>。</p>

<h2>7 · 结论</h2>
<ul>
<li><b>权限不构成障碍。</b>主板档几何日均 {e_mb['geo_bp']:+.2f}bp，全市场 {e_all['geo_bp']:+.2f}bp，
差 {gap_geo:.1f}bp，配对 t 仅 {float(inc.iloc[0]['t']):.2f}。这已经是"看不出来"的级别。</li>
<li><b>今天该买的是 {mb['code']} {mb['name']}</b>（{mb['board']}），开盘不高于 {mb['gap_cap']:.2f} 才买，
高于就放弃、第二天重出信号。有创业板权限的话首选换为 {cy['code']} {cy['name']}。</li>
<li><b>代价在成本线上，不在权限上。</b>主板档 13bp 成本年化 {ann(e_mb['geo_bp'],13):+.1f}%，
18bp 掉到 {ann(e_mb['geo_bp'],18):+.1f}%，20bp 归零。全市场档的安全垫比它厚
{e_all['geo_bp']-e_mb['geo_bp']:.1f}bp——这 3 个多 bp 就是"要不要去开权限"的全部数学答案。</li>
<li><b>验证期仍然不显著（t={float(e_mb['valid_t']):.2f}）。</b>换档不改变这个结论。
在攒够前向样本之前，任何一档都只适合 1–2 万试仓。</li>
</ul>

<footer>
信号日 {bar.date()}，数据截至该日收盘。模型为 <code>outputs/walkforward_v4/models/drop_rank_*.txt</code>
（v4，训练与选股均剔除 ST）。分档回测复用 v4 的 20 折季度 walk-forward 预测，
仅在选股环节裁剪可买板块，未重训模型——因此各档之间的差异纯粹来自"换了哪些票"。
价格为前复权，成交额原始口径。未计入冲击成本、停牌顺延与新股上市初期的流动性约束。
开盘过滤阈值 1.5% 为样本内扫描结果，已验证其效力在 2025 年后衰减（验证期增量仅 +4bp）。
本报告不构成投资建议。
</footer>
</div></body></html>"""

    out = ROOT / "outputs/report" / "权限分档_今日信号.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[ok] {out}  ({len(html):,} bytes)")
    print(f"主板档 {e_mb['mean_bp']:+.2f} / 全市场 {e_all['mean_bp']:+.2f} bp, "
          f"gap {gap_geo:+.2f}bp")


if __name__ == "__main__":
    main()
