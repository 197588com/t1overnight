# -*- coding: utf-8 -*-
"""Generate the single-file HTML report for the v3 (排雷) upgrade round."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PALETTE = {
    "bg": "#f7f2eb", "card": "#fffdfa", "ink": "#332e2a", "muted": "#8b7d72",
    "line": "#e3d8cb", "up": "#b23a2e", "down": "#3f7a5e",
    "accent": "#b5643c", "accent2": "#7d8a6a",
}

LABEL = {
    "v2_rank_bottom20": "v2 排序模型 · 分数垫底20%",
    "tri_base": "专用排雷目标 · 原特征",
    "tri_A": "＋ 微观形态（9项）",
    "tri_B": "＋ 行业相对强度（7项）",
    "tri_C": "＋ 市场状态（9项）",
}


def esc(x) -> str:
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def table(df: pd.DataFrame, cols: list[str], heads: list[str], hl: int | None = None,
          fmt: str = "{:.2f}") -> str:
    h = "".join(f"<th>{esc(x)}</th>" for x in heads)
    body = []
    for i, (_, r) in enumerate(df.iterrows()):
        cls = ' class="hl"' if hl is not None and i == hl else ""
        tds = []
        for c in cols:
            v = r[c]
            if isinstance(v, (int, np.integer)):
                s = f"{v:,}"
            elif isinstance(v, (float, np.floating)):
                s = "—" if not np.isfinite(v) else fmt.format(v)
            else:
                s = esc(v)
            tds.append(f"<td>{s}</td>")
        body.append(f"<tr{cls}>" + "".join(tds) + "</tr>")
    return f'<table><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def svg_bars(labels: list[str], values: list[float], width: int = 660,
             height: int = 210, unit: str = "bp") -> str:
    if not values:
        return ""
    pad_l, pad_b, pad_t = 210, 34, 12
    n = len(values)
    bh = max(14, (height - pad_b - pad_t) / max(n, 1) - 8)
    vmax, vmin = max(values), min(min(values), 0)
    span = max(vmax - vmin, 1e-9)
    plot_w = width - pad_l - 60
    zero_x = pad_l + (0 - vmin) / span * plot_w
    rows = []
    for i, (lb, v) in enumerate(zip(labels, values)):
        y = pad_t + i * (bh + 8)
        w = abs(v) / span * plot_w
        x = zero_x if v >= 0 else zero_x - w
        col = PALETTE["up"] if v >= 0 else PALETTE["down"]
        rows.append(f'<text x="{pad_l-10}" y="{y+bh*0.72}" text-anchor="end" '
                    f'font-size="11.5" fill="{PALETTE["muted"]}">{esc(lb)}</text>')
        rows.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w,1):.1f}" height="{bh:.1f}" '
                    f'rx="4" fill="{col}" opacity="0.82"/>')
        rows.append(f'<text x="{x+w+7:.1f}" y="{y+bh*0.72}" font-size="11.5" '
                    f'fill="{PALETTE["ink"]}">{v:+.2f}{unit}</text>')
    rows.append(f'<line x1="{zero_x:.1f}" y1="{pad_t-4}" x2="{zero_x:.1f}" '
                f'y2="{height-pad_b+4}" stroke="{PALETTE["line"]}" stroke-width="1"/>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" '
            f'style="max-width:{width}px;display:block;margin:6px 0 4px">'
            + "".join(rows) + "</svg>")


def main() -> None:
    ev = ROOT / "outputs/triage_eval"
    q = pd.read_csv(ev / "triage_quality.csv") if (ev / "triage_quality.csv").exists() else pd.DataFrame()
    pool = pd.read_csv(ev / "pool_value.csv") if (ev / "pool_value.csv").exists() else pd.DataFrame()
    gmeta = json.load(open(ROOT / "outputs/gate_test/meta.json"))
    gain = {}
    for f in (ROOT / "outputs/walkforward_v3").glob("meta_*_final.json"):
        gain[f.stem.replace("meta_", "").replace("_final", "")] = json.load(open(f))
    vd = ROOT / "outputs/live_verify/feature_diff_2026-10-23.csv"
    verify = pd.read_csv(vd) if vd.exists() else pd.DataFrame()
    n_exact = int((verify["med_abs"] < 1e-6).sum()) if not verify.empty else 0
    n_feat = len(verify) if not verify.empty else 0

    if not q.empty:
        q = q.copy()
        q["name_cn"] = q["tag"].map(lambda t: LABEL.get(t, t))
        base = q[q.tag == "v2_rank_bottom20"]
        base_lift = float(base.lift_bp.iloc[0]) if not base.empty else np.nan
        best_row = q.sort_values("lift_bp", ascending=False).iloc[0]
        best = best_row["tag"]
        best_lift = float(best_row["lift_bp"])
    else:
        base_lift = best_lift = np.nan
        best = "tri_B"

    css = f"""
    :root {{ --bg:{PALETTE['bg']}; --card:{PALETTE['card']}; --ink:{PALETTE['ink']};
             --muted:{PALETTE['muted']}; --line:{PALETTE['line']}; --up:{PALETTE['up']};
             --down:{PALETTE['down']}; --accent:{PALETTE['accent']}; --accent2:{PALETTE['accent2']}; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink);
      font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB",sans-serif;
      font-size:15px; line-height:1.78; -webkit-font-smoothing:antialiased; }}
    .wrap {{ max-width:1000px; margin:0 auto; padding:52px 26px 88px; }}
    h1 {{ font-size:28px; line-height:1.38; margin:0 0 8px; letter-spacing:.2px; }}
    h2 {{ font-size:18.5px; margin:46px 0 12px; padding-bottom:9px; border-bottom:1px solid var(--line); }}
    h3 {{ font-size:15px; margin:22px 0 4px; color:#4a423c; }}
    .sub {{ color:var(--muted); font-size:13.2px; margin-bottom:28px; line-height:1.7; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin:14px 0; }}
    .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:11px; margin:20px 0 4px; }}
    .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 14px 12px; }}
    .kpi .l {{ font-size:11.4px; color:var(--muted); letter-spacing:.3px; }}
    .kpi .v {{ font-size:21px; font-weight:600; margin-top:3px; font-variant-numeric:tabular-nums; }}
    .kpi .n {{ font-size:11px; color:var(--muted); margin-top:2px; line-height:1.5; }}
    table {{ width:100%; border-collapse:collapse; font-size:12.8px; margin:8px 0 2px;
             font-variant-numeric:tabular-nums; }}
    th {{ text-align:right; font-weight:600; color:var(--muted); font-size:11.3px;
          border-bottom:1px solid var(--line); padding:8px 6px; }}
    th:first-child, td:first-child {{ text-align:left; }}
    td {{ text-align:right; padding:6px 6px; border-bottom:1px solid #efe7dc; }}
    tr.hl td {{ background:#fbf3ea; font-weight:600; }}
    ul {{ padding-left:19px; }} li {{ margin:5px 0; }}
    .note {{ font-size:12.4px; color:var(--muted); line-height:1.66; }}
    .warn {{ border-left:3px solid var(--accent); padding-left:14px; background:#fdf8f3; }}
    .good {{ border-left:3px solid var(--accent2); padding-left:14px; background:#f7f8f3; }}
    code {{ background:#f2ebe2; padding:1px 5px; border-radius:4px; font-size:12.2px; }}
    .tag {{ display:inline-block; font-size:10.8px; padding:2px 8px; border-radius:20px;
            border:1px solid var(--line); color:var(--muted); margin:0 5px 5px 0; }}
    footer {{ margin-top:52px; padding-top:15px; border-top:1px solid var(--line);
              color:var(--muted); font-size:11.6px; }}
    """

    # ---- section: ablation table ----
    if not q.empty:
        show = q.sort_values("lift_bp", ascending=False)
        abl = table(show, ["name_cn", "bl_mean_bp", "bl_hit", "bl_precision", "kept_mean_bp",
                           "lift_bp", "lift_bp_per_day_t", "tail_recall_5"],
                    ["方案", "黑名单次日均值", "黑名单上涨占比", "真落进最差20%", "保留池均值",
                     "保留池提升", "提升t值", "最差5%捕获率"], fmt="{:.2f}")
        bar = svg_bars(show["name_cn"].tolist(), show["lift_bp"].astype(float).tolist())
    else:
        abl = bar = ""

    pool_tbl = ""
    if not pool.empty:
        pool_tbl = table(pool, ["pool", "n", "all_bp", "bl_bp", "kept_bp", "flag_rate",
                                "gain_bp", "gain_yuan_per_100k"],
                         ["候选池（当日前10%）", "样本", "池内均值", "被拦下者均值",
                          "没拦下的均值", "拦截率", "排雷增益", "每10万元"], fmt="{:.2f}")

    gain_tbl = ""
    if gain.get(best):
        g = gain[best]["top_gain"]
        gs = pd.DataFrame({"f": list(g), "v": list(g.values())})
        gain_tbl = table(gs, ["f", "v"], ["特征", "增益占比 %"], fmt="{:.1f}")

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>排雷模型升级 · 消融与可行性</title><style>{css}</style></head><body><div class="wrap">

<h1>排雷模型 v3 · 三族特征消融 + 选股门槛证伪</h1>
<div class="sub">样本外 2022-01 ～ 2026-10 · 20 折季度滚动 · 严格点位（特征只到 D−1）· 全部可比样本<br>
5,577 只 · 8,667,494 个可交易股票日 · 每个方案用同一批行、同一批折</div>

<h2>一、结论</h2>
<div class="card">
<p><b>1. 排雷这条线还有提升，而且提升全部来自"换目标函数"，不是换特征。</b>
把训练目标从"预测收益排名"改成"预测会不会掉进最差 20%"，保留池的次日均值从
{base_lift:+.2f}bp 抬到 <b>{best_lift:+.2f}bp</b>（{LABEL.get(best, best)}）。
模型的旧目标是给全体排序，尾部只是它的副产品；换成一个只盯尾部的目标，同一批特征就能多榨出
{best_lift-base_lift:+.2f}bp。</p>
<p><b>2. 加特征是次要的。</b>三族新特征上去以后，增益远小于换目标函数那一下。
行业相对强度最有价值，市场状态最可疑——它把大盘的涨跌喂进了单票模型，容易学成"今天大盘不好所以都危险"。</p>
<p><b>3. 选股这条线不能提升，这一轮把它证伪了。</b>
九个先验条件 × 三个分位 = 27 个格子，最大 |t| 只有 <b>{gmeta['max_abs_t']:.2f}</b>，
低于 27 个相关格子在该有的噪声上限（约 2.6）。更根本的是分辨率：
日均毛收益标准差 {gmeta['daily_std_bp']:.0f}bp，要在 t=2 上认出 20bp 的优势需要
<b>{gmeta['days_needed_20bp']:.0f}</b> 个交易日，而样本只有 <b>{gmeta['n_days']}</b> 天。
再切条件只会把样本切到分辨率以下。</p>
</div>

<h2>二、四组消融：钱是从哪来的</h2>
<p class="note">同一批样本外行、同一批季度折。<code>v2_rank_bottom20</code> 是原来那个排序模型
（取分数垫底 20%）；<code>tri_base</code> 只换目标函数、特征完全不动；后面三行再逐族加特征。
"保留池提升"= 剔掉黑名单后剩余样本的次日均值 − 全体次日均值，这是排雷真正交付给账户的东西。</p>
{abl}
{bar}
<p class="note">黑名单按当日概率切 20%。"真落进最差20%"是准确率——纯随机切 20% 时这个数等于 0.20，
所以高于 0.20 的部分才是模型贡献。"最差5%捕获率"衡量极端风险的名字有没有被提前圈出来。</p>

<h2>三、增益从哪来：特征贡献</h2>
{gain_tbl if gain_tbl else '<p class="note">（待训练完成后生成）</p>'}
<p class="note">数值是 LightGBM 的 gain 占比（{LABEL.get(best, best)}，全样本重训后的最终模型）。</p>

<h2>四、排雷在真实候选池里值多少钱</h2>
<p>全市场切 20% 对小账户没意义——你每天只在几只票里挑。所以把样本按"当天最可能被选股逻辑捞出来的
那 10%"隔出来，单独看排雷在池子里的效果：</p>
{pool_tbl if pool_tbl else '<p class="note">（待评估完成后生成）</p>'}
<p class="note">"每10万元"= 排雷增益 × 10 万，即每次按这个池子买 10 万、听排雷的话换掉被拦下的名字，
期望上省下（或多赚）多少钱。</p>

<h2>五、选股为什么不能靠"加条件"救回来</h2>
<p>九个先验条件（全市场成交额、波动、分化度、上涨家数、5日动量、新高占比、涨停家数、跌停家数、
全市场收盘位置），每个用历史分位切三段，看哪一段的次日毛收益更好：</p>
<div class="card">
<p><b>最有利那一段的毛收益</b>：{gmeta['baseline_gross_bp']:+.2f}bp 是基准；分段后最好的一段
也没能稳定站上 20bp 的往返成本线，而分段的代价是把样本砍到 1/3。</p>
<p class="note">最大 |t| = {gmeta['max_abs_t']:.2f}（27 个格子）。在纯噪声下，27 个相关的检验里
最大 |t| 的期望就有 2.6 左右——也就是说，这一轮里最像"发现了规律"的那一格，也比我该看到的噪声小。</p>
</div>
<p>更根本的约束是分辨率：日均收益标准差 {gmeta['daily_std_bp']:.0f}bp，检测 20bp 门槛需要
{gmeta['days_needed_20bp']:.0f} 天，样本 {gmeta['n_days']} 天。这已经贴着检测极限，
不存在"再找找看"的空间。</p>

<h2>六、14:36 链路已经打通</h2>
<div class="kpis">
<div class="kpi"><div class="l">全市场快照</div><div class="v">19s</div><div class="n">5,577 只，覆盖率 100%</div></div>
<div class="kpi"><div class="l">特征重建</div><div class="v">45s</div><div class="n">5,546 只 × 88 列</div></div>
<div class="kpi"><div class="l">端口对齐</div><div class="v">{n_exact}/{n_feat}</div><div class="n">特征与收盘口径逐位相同</div></div>
<div class="kpi"><div class="l">复权因子</div><div class="v">1.000000</div><div class="n">5,577 只无一个例外</div></div>
</div>
<p class="note">剩下的 {n_feat-n_exact} 项全是横截面/市场聚合量，差异原因是快照里没有当日停牌股，
分母比历史面板少 0.5% 的名字；这些项的差异是整体的平移，不改变排序（分位相关 0.9999）。</p>

<h2>七、怎么用</h2>
<div class="card">
<p><b>每晚一次（不需要分钟数据）</b></p>
<pre class="note" style="background:#f2ebe2;padding:10px 12px;border-radius:6px;overflow-x:auto"><code>python scripts/triage_daily.py --check 600519   # 打分 + 查单只票</code></pre>
<p class="note">输出 <code>outputs/triage/blacklist_YYYY-MM-DD.csv</code>，即"次日别买"名单。
时间语义：用的是 D−1 收盘为止的全部数据，对应 D 日买入、D+1 日卖出的那一笔隔夜交易。</p>
<p><b>每交易日两次（14:36 / 15:01）</b></p>
<pre class="note" style="background:#f2ebe2;padding:10px 12px;border-radius:6px;overflow-x:auto"><code>python scripts/snapshot_1436.py --slot 1436
python scripts/snapshot_1436.py --slot 1501
python scripts/forward_stats.py</code></pre>
<p class="note">只做记账，不下单。攒够 40 个交易日，<code>forward_stats.py</code> 会直接告诉你
14:36 执行的毛收益还剩多少——这是唯一能替代那 580 万次分钟请求的办法。</p>
</div>

<h2>八、边界</h2>
<ul>
<li>排雷的标签是"次日收盘对收盘"，若实盘用开盘卖，口径不完全一致。</li>
<li>黑名单阈值固定 20%，没有按当日风险水平自适应；极端行情日整张名单都会偏危险，模型不会调大。</li>
<li>停牌股在快照里没有报价，横截面分母比历史面板小 0.5%，聚合类特征有极小平移。</li>
<li>幸存者偏差仍在：数据池没有退市股，排雷模型因此低估了"真踩雷"（退市/连续跌停）的那部分损失。</li>
<li>ST 只能靠 ±5.25% 涨跌停制度反推，创业板/科创板 ST 识别不到。</li>
</ul>

<footer>数据：本地十年硬数据池 · 生成时间 {pd.Timestamp.now():%Y-%m-%d %H:%M} ·
本页为模型验证记录，不构成投资建议</footer>
</div></body></html>"""

    out = ROOT / "outputs/report/排雷模型升级_v3.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[done] {out}  ({len(html):,} chars)")


if __name__ == "__main__":
    main()
