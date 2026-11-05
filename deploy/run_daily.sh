#!/bin/sh
# Daily driver invoked by launchd (see com.t1overnight.*.plist).
#
# Why a wrapper instead of pointing launchd straight at python
# ------------------------------------------------------------
# launchd starts jobs with a minimal environment (PATH=/usr/bin:/bin, no shell
# rc) and appends every run's stdout to one fixed file.  Three things are needed
# that a bare <ProgramArguments> cannot give us:
#
#   1. a per-day log, so a failure three weeks ago is still readable
#   2. an absolute interpreter path, because launchd will not resolve `python`
#   3. a heartbeat line that is written even when the run fails, so "did the
#      scheduler fire at all?" and "did the data arrive?" stay distinguishable
#
# The heartbeat matters more than it sounds: the local hermes cron on this
# machine sat dead for 16 days with every job still showing `active`, and there
# was no way to tell that apart from "the job ran and found nothing".
#
# usage: run_daily.sh evening|morning

set -u

# 自定位：从本脚本位置推导项目根，迁移工作区目录后无需改任何一行。
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
MODE="${1:-evening}"

case "$MODE" in
  evening) FLAG="";          LABEL="当晚 $T1_EVENING_AT" ;;
  morning) FLAG="--morning"; LABEL="晨补 $T1_MORNING_AT" ;;
  *) echo "usage: $0 evening|morning" >&2; exit 64 ;;
esac

# 失败告警：只在本层真失败时说话（T1_ALERT=1 才开，见 config.env 第 7 节）。
# 主调度（AI 自动化）有自己的对话汇报通道，本函数服务的是"WorkBuddy 没跑起来、
# 只剩 launchd 这条腿"的场景——那时除了这里，没有任何东西会告诉你今天没信号。
# 定义放在前置守卫之前，是因为解释器/项目根缺失是最该被告知的一种失败。
alert() {
  # $1 标题  $2 正文  $3 退出码
  [ "${T1_ALERT:-0}" = "1" ] || return 0
  # 正文可能带配置里的路径，转义双引号，避免苹果脚本语法被路径打断。
  _body=$(printf '%s' "$2" | sed 's/"/\\"/g')
  /usr/bin/osascript -e "display notification \"$_body\" with title \"$1\" sound name \"Basso\"" \
      >/dev/null 2>&1 || true
  _alog="$ROOT/outputs/live/alerts.log"
  mkdir -p "$ROOT/outputs/live" 2>/dev/null || true
  printf '%s\tmode=%s\trc=%s\t%s\t%s\n' \
      "$(date '+%F %T')" "$MODE" "$3" "$1" "$2" >>"$_alog" 2>/dev/null || true
}

# 前置守卫：解释器或项目根缺失时立刻失败并说清原因，而不是让 python
# 报一个 "No such file" 之后日志里只剩 traceback。
[ -x "$PY" ]    || { echo "[fatal] 解释器不存在: $PY" >&2
                     alert "A股隔夜短线·任务无法启动 (rc=78)" "解释器不存在，今日未出信号：$PY" 78
                     exit 78; }
[ -d "$ROOT" ]  || { echo "[fatal] 项目根不存在: $ROOT" >&2
                     alert "A股隔夜短线·任务无法启动 (rc=78)" "项目根不存在，今日未出信号：$ROOT" 78
                     exit 78; }

cd "$ROOT" || exit 1
LOG_DIR="$ROOT/outputs/live/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${MODE}_$(date +%Y%m%d).log"

TMPOUT="$(mktemp -t t1run)"
{
  echo "=== $MODE ($LABEL) start $(date '+%F %T') ==="
  "$PY" scripts/daily_run.py $FLAG >"$TMPOUT" 2>&1
  RC=$?
  cat "$TMPOUT"
  echo "=== $MODE end $(date '+%F %T') rc=$RC ==="
} >>"$LOG" 2>&1

# The heartbeat line is written by daily_run.py itself (so it is produced no
# matter which scheduler started the run); this wrapper only owns the log file.

# 失败告警。口径与 daily_run.py 的退出码严格对应，不多报也不少报：
#   rc=0  正常出信号 / 主调度已成功、本层 skip  -> 静默
#   rc=3  非交易日，是正确结果不是故障          -> 静默
#   rc=2  数据未就绪，当日空仓                  -> 告警
#   其他  脚本异常（含 78 前置守卫）             -> 告警
LOG_NAME=$(basename "$LOG")
case "$RC" in
  0)
    # 推送只在「本次真的新出了信号」时发。主调度已成功、本层 skip 的那次
    # 不推，否则同一天会收到两条一样的消息。
    if grep -q '\[skip\]' "$TMPOUT" 2>/dev/null; then
      echo "[push] 本层 skip（主调度已出信号），不重复推送" >>"$LOG" 2>&1
    elif [ "${T1_WECOM_PUSH:-0}" = "1" ]; then
      "$PY" scripts/push_wecom.py >>"$LOG" 2>&1 || \
        echo "[push] 推送失败，见上方 stderr（不改变本次 rc）" >>"$LOG" 2>&1
    fi
    ;;
  2) alert "A股隔夜短线·今日空仓 (rc=2)" \
           "${MODE} 数据未就绪，未出信号，今天不做。日志：$LOG_NAME" 2 ;;
  *) alert "A股隔夜短线·任务异常 (rc=$RC)" \
           "${MODE} 异常退出，信号可能未生成。日志：$LOG_NAME" "$RC" ;;
esac
rm -f "$TMPOUT" 2>/dev/null || true

# keep N days of logs; they are small but not free.  launchd 自己的
# StandardOutPath/StandardErrorPath 也落在同一目录，一并裁剪，否则
# launchd_*.err 会无限增长（它只追加、从不轮转）。
find "$LOG_DIR" \( -name '*.log' -o -name 'launchd_*.out' -o -name 'launchd_*.err' \) \
     -mtime +"$T1_LOG_RETAIN_DAYS" -delete 2>/dev/null

exit $RC
