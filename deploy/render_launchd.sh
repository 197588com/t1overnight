#!/bin/sh
# 从 deploy/config.env 渲染 launchd plist。
#
# 为什么不再手写 plist
# --------------------
# 手写的 plist 把项目绝对路径重复了 4 次、把触发时间又重复了 5 次（周一至
# 周五各一条 StartCalendarInterval）。改一次部署位置要改 8 个地方，漏一个
# 的结果是定时器指向旧目录 —— 它仍会"成功"触发，只是跑的是别的东西，
# 日志里看不出异常。渲染成单一来源后，位置和时间只有一处可改。
#
# 输出是幂等的：同样的 config.env 渲染两次，字节级一致，git 不会反复抖动。
#
# usage: render_launchd.sh [--check]
#        --check  只渲染到临时目录并比对，不写盘；有差异则退出 1

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
T1_ROOT="$(cd "$HERE/.." && pwd)"
if [ ! -f "$HERE/config.env" ]; then
  echo "[fatal] 缺少 $HERE/config.env —— 先执行：" >&2
  echo "        cp deploy/config.env.example deploy/config.env  # 再按注释改路径" >&2
  exit 78
fi
. "$HERE/config.env"

OUT_DIR="$HERE/launchd"
UID_N="$(id -u)"
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

# HH:MM + N 分钟，跨小时/跨日都正确（跨日只影响显示，launchd 用 0-23 小时）
add_min() {
  _h="${1%%:*}"
  _m="${1##*:}"
  _t=$(( _h * 60 + _m + $2 ))
  printf '%02d:%02d' $(( (_t / 60) % 24 )) $(( _t % 60 ))
}

EVENING_AT="$(add_min "$T1_EVENING_AT" "$T1_LAUNCHD_OFFSET_MIN")"
MORNING_AT="$(add_min "$T1_MORNING_AT" "$T1_LAUNCHD_OFFSET_MIN")"

# 单条 plist 模板。$1=label  $2=hh:mm  $3=脚本参数  $4=说明
render_one() {
  _label="$1"
  _hh="${2%%:*}"
  _mm="${2##*:}"
  _arg="$3"
  _note="$4"

  cat <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${_label}</string>

  <!-- 指向 wrapper 而不是 python：launchd 不加载 shell rc，也没有可用的 PATH。 -->
  <key>ProgramArguments</key>
  <array>
    <string>/bin/sh</string>
    <string>${T1_ROOT}/deploy/run_daily.sh</string>
    <string>${_arg}</string>
  </array>

  <key>WorkingDirectory</key>
  <string>${T1_ROOT}</string>

  <!-- ${_note}
       本文件由 deploy/render_launchd.sh 从 deploy/config.env 生成，不要手改。 -->
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>${_hh}</integer><key>Minute</key><integer>${_mm}</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>${_hh}</integer><key>Minute</key><integer>${_mm}</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>${_hh}</integer><key>Minute</key><integer>${_mm}</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>${_hh}</integer><key>Minute</key><integer>${_mm}</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>${_hh}</integer><key>Minute</key><integer>${_mm}</integer></dict>
  </array>

  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>${HOME}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <!-- 刻意不设 HTTP_PROXY：三个数据源已验证可直连，继承应用代理只会多一个必须在
         触发时刻活着的东西。 -->
  </dict>

  <key>StandardOutPath</key><string>${T1_ROOT}/${T1_OUT_DIR}/logs/launchd_${_arg}.out</string>
  <key>StandardErrorPath</key><string>${T1_ROOT}/${T1_OUT_DIR}/logs/launchd_${_arg}.err</string>

  <key>RunAtLoad</key><false/>
</dict>
</plist>
PLIST
}

render_all() {
  _dir="$1"
  mkdir -p "$_dir"
  render_one com.t1overnight.evening "$EVENING_AT" evening \
    "${EVENING_AT} 周一至周五，比主调度器（${T1_EVENING_AT}）晚 ${T1_LAUNCHD_OFFSET_MIN} 分钟。" \
    >"$_dir/com.t1overnight.evening.plist"
  render_one com.t1overnight.morning "$MORNING_AT" morning \
    "${MORNING_AT} 周一至周五，比主调度器（${T1_MORNING_AT}）晚 ${T1_LAUNCHD_OFFSET_MIN} 分钟。" \
    >"$_dir/com.t1overnight.morning.plist"
}

if [ "$CHECK" -eq 1 ]; then
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  render_all "$TMP"
  RC=0
  for L in com.t1overnight.evening com.t1overnight.morning; do
    if ! cmp -s "$TMP/$L.plist" "$OUT_DIR/$L.plist" 2>/dev/null; then
      echo "[drift] $L.plist 与 config.env 不一致" >&2
      RC=1
    fi
  done
  [ "$RC" -eq 0 ] && echo "[ok] plist 与 config.env 一致"
  exit "$RC"
fi

render_all "$OUT_DIR"

for L in com.t1overnight.evening com.t1overnight.morning; do
  if plutil -lint "$OUT_DIR/$L.plist" >/dev/null 2>&1; then
    echo "  [rendered] $L.plist"
  else
    echo "  [invalid] $L.plist 未能通过 plutil -lint" >&2
    exit 1
  fi
done

echo "触发时间：evening ${EVENING_AT} · morning ${MORNING_AT}（主调度 ${T1_EVENING_AT}/${T1_MORNING_AT}）"
