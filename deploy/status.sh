#!/bin/sh
# 一眼看清"这套东西今天到底跑没跑、出没出信号"。
#
# 分开报告两件事，因为它们不同：
#   心跳   —— 调度器有没有触发（没有 = 电脑/调度器的问题）
#   信号   —— 数据有没有到、模型有没有出票（有 = 可以照做）
#
# 只有这两条分开看，"没看到消息" 才能被解释成结论，而不是猜。
#
# usage: status.sh

set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
T1_ROOT="$(cd "$HERE/.." && pwd)"
if [ ! -f "$HERE/config.env" ]; then
  echo "[fatal] 缺少 $HERE/config.env —— 先执行：" >&2
  echo "        cp deploy/config.env.example deploy/config.env  # 再按注释改路径" >&2
  exit 78
fi
. "$HERE/config.env"

ROOT="$T1_ROOT"
PY="$T1_PY"
UID_N="$(id -u)"
cd "$ROOT" || exit 1

echo "════ 调度器 ════"
LOADED=0
for L in com.t1overnight.evening com.t1overnight.morning; do
  if launchctl print "gui/$UID_N/$L" >/dev/null 2>&1; then
    R=$(launchctl print "gui/$UID_N/$L" 2>/dev/null | awk -F'= ' '/runs =/{print $2}')
    E=$(launchctl print "gui/$UID_N/$L" 2>/dev/null | awk -F'= ' '/last exit code/{print $2}')
    printf '  %-26s 已加载  runs=%s  last_rc=%s\n' "$L" "${R:-0}" "${E:-–}"
    LOADED=1
  fi
done
if [ "$LOADED" -eq 0 ]; then
  echo "  单一触发器：仅 WorkBuddy 定时（${T1_EVENING_AT} / ${T1_MORNING_AT}）"
  echo "  launchd 备用层未启用；需要时 sh deploy/install_launchd.sh install"
fi

# 僵尸调度器：本机还加载着指向其它项目的隔夜任务，它们不产出本项目的信号，
# 但会在排查"到底谁在跑"时把水搅浑。只报告，不擅自卸载。
for L in $(launchctl list 2>/dev/null | awk '/overnight/{print $3}'); do
  case "$L" in
    com.t1overnight.*) continue ;;
  esac
  P=$(launchctl print "gui/$UID_N/$L" 2>/dev/null | awk -F'= ' '/working directory =/{print $2}' | head -1)
  echo "  [无关任务] $L 仍已加载 · 目录=${P:-?}"
done

echo
echo "════ 心跳（最近 8 次触发）════"
if [ -s outputs/live/heartbeat.log ]; then
  tail -8 outputs/live/heartbeat.log | sed 's/^/  /'
  # A beat list only says what happened, not whether the list stopped growing.
  # The whole point of the heartbeat is to catch a scheduler that died quietly,
  # so judge the age here instead of leaving it to the reader to subtract dates.
  "$PY" - <<'PY'
import re
import numpy as np
import pandas as pd

lines = [x for x in open("outputs/live/heartbeat.log", encoding="utf-8").read().splitlines() if x.strip()]
last = lines[-1]
# Split on any whitespace: a hand-edited or older-format line must not take the
# whole board down with a parse error, which is worse than the line itself.
tok = re.split(r"\s+", last.strip())
try:
    ts = pd.Timestamp(f"{tok[0]} {tok[1]}")
except (IndexError, ValueError):
    print(f"  最后一行格式不认识，跳过判定：{last!r}")
else:
    rc = re.search(r"rc=(-?\d+)", last)
    rc = rc.group(1) if rc else "?"
    now = pd.Timestamp.now()
    age_h = (now - ts).total_seconds() / 3600
    # Weekend gaps are normal (Fri 17:20 -> Mon 07:50), so age alone misleads.
    # Count skipped weekdays instead: one is a slow start, two or more means the
    # schedule stopped -- which is exactly what happened on this machine for 16 days.
    missed = int(np.busday_count(ts.date(), now.date()))
    print(f"  上次心跳 {ts:%Y-%m-%d %H:%M}（{age_h:.1f} 小时前）")
    if missed >= 2:
        print(f"  ⚠ 中间跳过 {missed} 个工作日 —— 调度器可能已静默死亡，"
              f"先确认 WorkBuddy 那两个 17:20 / 07:50 任务是否还在")
    if rc == "2":
        print("  ⚠ 最后一次 rc=2：数据不全已挂起，等 07:50 晨补")
    elif rc == "3":
        print("  rc=3：最近一次触发落在非交易日，正常跳过")
    elif rc != "0":
        print(f"  ⚠ 最后一次退出码 rc={rc}，要看原因")
PY
else
  echo "  还没有心跳记录（说明定时器一次都没跑过）"
fi

# 实盘台账（positions.csv / real_fills.csv / position_book.py）已于 2026-10-28 退役。
# 运行模式自 10-27 起是纯影子前测：不跟踪、不提醒、不评价实盘持仓，
# 账面唯一口径是 forward_ledger.csv。原来的「持仓台账」节会持续输出
# 实盘浮亏、仓位缺口、逾期未平仓告警，把"该买哪只"淹在后面 —— 一并删除。
echo
echo "════ 今日该做什么 ════"
TODAY=$(date +%Y-%m-%d)
if [ -f "outputs/live/NO_SIGNAL_${TODAY}.md" ]; then
  echo "  ⚠ 今日已声明空仓：outputs/live/NO_SIGNAL_${TODAY}.md"
  grep -E "原因|取数行数" "outputs/live/NO_SIGNAL_${TODAY}.md" | sed 's/^/    /'
fi

if [ -f outputs/live/forward_ledger.csv ]; then
  # Column order differs between a freshly created ledger and one that grew by
  # concat, so address columns by name rather than by position.
"$PY" - "$TODAY" <<'PY'
import os
import sys
import pandas as pd

today = sys.argv[1]
d = pd.read_csv("outputs/live/forward_ledger.csv", dtype={"code": str})
for c in ("gross_ret", "net_ret", "entry_price", "exit_price", "gap_cap"):
    d[c] = pd.to_numeric(d.get(c), errors="coerce")
d["traded"] = d.get("traded", "").fillna("").astype(str)
d["signal_bar"] = d["signal_bar"].astype(str)
d["trade_date"] = d["trade_date"].astype(str)

# Two different questions, and the morning answer is not the evening one:
#   09:15 你要知道的是"今天开盘买哪只" -> trade_date == today
#   17:30 你要知道的是"刚才算出哪只"   -> signal_bar == today
# Showing only the second one hid the actionable pick every morning.
low = d["traded"].str.lower()
real = d[~low.str.startswith("void") & (low != "no_pick")]


POS_PCT = float(os.environ.get("T1_POSITION_PCT", "1.0") or 1.0)
CAP = float(os.environ.get("T1_CAPITAL", "100000"))


def plan_qty(price):
    """按目标仓位算手数并取整百股。这里算，是为了让汇报引脚本而不是让人心算。"""
    if price is None or pd.isna(price) or float(price) <= 0:
        return 0
    return int(CAP * POS_PCT / float(price) / 100) * 100


def line(r, tag):
    if pd.notna(r["exit_price"]):
        state = (f"已了结 {r['entry_price']:.2f}→{r['exit_price']:.2f} "
                 f"{r['net_ret']*1e4:+.0f}bp")
    elif pd.notna(r["entry_price"]):
        state = f"已入场 {r['entry_price']:.2f}，等次日开盘卖"
    elif r["traded"] == "no_fill_gap":
        state = "跳空放弃（开盘高于上限）"
    else:
        state = "未开盘，上限内才买"
    q = plan_qty(r["gap_cap"])
    qty_txt = f"建议 {q:,} 股（{POS_PCT:.0%}仓位） · " if q else ""
    return (f"  [{tag}] {r['code']} {r['name']}（{r['board']}）"
            f"上限 {r['gap_cap']:.2f} · {qty_txt}{state}")


buy_today = real[real["trade_date"] == today]
sig_today = real[real["signal_bar"] == today]

print("  今天开盘执行（今天买、明天开盘卖）：")
if len(buy_today):
    for _, r in buy_today.iterrows():
        print(line(r, "今日执行"))
else:
    print("    无 —— 今天空仓，不动手")

print("  今天收盘后新出（明天开盘执行）：")
if len(sig_today):
    for _, r in sig_today.iterrows():
        print(line(r, "明日执行"))
else:
    print(f"    无（17:20 之后再看；账本最后一次信号日 {real['signal_bar'].max() if len(real) else '—'}）")

# 卖出侧单独成段：买入信号容易被看见，卖出日期却要靠人记，漏掉一次就变成
# 被动多拿一天 —— 隔夜策略里多拿一天等于换了一套策略，收益口径全变。
print("  卖出策略（固定口径，不设止损止盈）：")
_unclosed = real[real["entry_price"].notna() & real["exit_price"].isna()]
if len(_unclosed):
    for _, r in _unclosed.iterrows():
        print(f"    [待卖] {r['code']} {r['name']} 入场 {r['entry_price']:.2f} → 下一交易日开盘卖出")
for _, r in sig_today.iterrows():
    if pd.isna(r["entry_price"]) and r["traded"] != "no_fill_gap":
        print(f"    [今晚新票] {r['code']} {r['name']} 明日开盘买入成功后 → 后一交易日开盘卖出，持有当日不动")
print("    · 到点就卖，不看盘中浮盈浮亏；开盘价高于上限则当日不买、不顺延")

graded = real[real["gross_ret"].notna()]
pending = real[real["entry_price"].notna() & real["exit_price"].isna()]
awaiting = real[real["entry_price"].isna() & (low[real.index] != "no_fill_gap")]
gap = real[real["traded"] == "no_fill_gap"]
nopick = int((low == "no_pick").sum())

print()
# 门槛数值来自 deploy/config.env，不再写死在报表文案里 —— 否则改了门槛，
# 报表还在报旧标准，看上去一切正常。
_gate_days = int(os.environ.get("T1_GATE_DAYS", "60"))
_gate_bp = int(os.environ.get("T1_GATE_BP", "25"))
print(f"════ 前向记账（上钱门槛：{_gate_days} 个交易日且毛均 ≥ +{_gate_bp}bp）════")
print(f"  已结算 {len(graded)} 笔 / 已入场待结算 {len(pending)} 笔 / "
      f"待开盘 {len(awaiting)} 笔 / 跳空放弃 {len(gap)} 笔"
      + (f" / 空仓日 {nopick} 笔" if nopick else ""))
if len(graded):
    print(f"  毛均 {graded['gross_ret'].mean()*1e4:+.2f}bp  "
          f"净均 {graded['net_ret'].mean()*1e4:+.2f}bp  "
          f"胜率 {(graded['gross_ret'] > 0).mean():.1%}")
    # 单笔均值看不出"这套策略到现在赚了多少"，必须给累计。
    # 口径：每笔按固定仓位比例投入、盈亏叠加复利，与建议手数同一仓位参数。
    _pos = float(os.environ.get("T1_POSITION_PCT", "1.0") or 1.0)
    _cap = float(os.environ.get("T1_CAPITAL", "100000"))
    _g = graded.sort_values("signal_bar")
    _eq = float((1 + _g["net_ret"] * _pos).prod())
    print(f"  累计收益 {(_eq - 1) * 100:+.2f}%（{_pos:.0%} 仓位复利）  "
          f"本金 ¥{_cap:,.0f} → ¥{_cap * _eq:,.0f}")
    print(f"  单笔最好 {_g['net_ret'].max() * _pos * 100:+.2f}%  "
          f"最差 {_g['net_ret'].min() * _pos * 100:+.2f}%（已折仓位）")
    need = _gate_days
    left = max(0, need - len(graded))
    print(f"  门槛进度 {len(graded)}/{need}"
          + (f"，还差 {left} 笔" if left else "，样本量已满足"))
PY
else
  echo "  账本不存在"
fi

echo
echo "════ 最后状态写入 ════"
if [ -f outputs/live/run_state.json ]; then
  "$PY" - <<'PY'
import json
s = json.load(open("outputs/live/run_state.json"))
for k in sorted(s)[-3:]:
    v = s[k]
    print(f"  {k}  {v.get('status','?'):8s} 源={v.get('source','?'):16s} "
          f"{v.get('code','')} {v.get('name','')}  {v.get('at','')}")
PY
fi

# 配置一致性单列一节：上面的心跳/信号回答"跑了没、出了没"，这一节回答
# "跑的还是不是你以为的那套配置"。两者分开，才不会用"有信号"掩盖"配错了"。
echo
echo "════ 配置一致性（只报 WARN/FAIL）════"
if [ -f "$HERE/config_check.py" ]; then
  "$PY" "$HERE/config_check.py" --quiet | grep -E '^\[(WARN|FAIL)\]' || echo "  全部通过"
else
  echo "  config_check.py 不存在"
fi
