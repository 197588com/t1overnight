#!/bin/sh
# Install (or remove) the two launchd timers that make this pipeline independent
# of any application being open.
#
# Why this exists
# ---------------
# The assistant's own scheduler only fires while the assistant is running.  That
# is fine most days and useless on the day it is not.  launchd is the OS
# scheduler: it fires whether or not any app is open, and it survives reboots.
# On this machine `crontab` is provably dead (macOS 26 runs no user cron at all,
# see the note in the old crontab), but LaunchAgents demonstrably fire -- the
# probe job used to design this ran at 30s intervals with 0s drift.
#
# usage: install_launchd.sh [install|remove|status]

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
AGENTS="$HOME/Library/LaunchAgents"
LABELS="com.t1overnight.evening com.t1overnight.morning"
UID_N="$(id -u)"

case "${1:-install}" in
  install)
    # 先按 config.env 重新渲染，再装载。手写的 plist 不再被信任 —— 它可能是
    # 上一处部署位置留下的副本，装上去会"成功"跑到旧目录去。
    /bin/sh "$HERE/render_launchd.sh" || { echo "渲染失败，未安装" >&2; exit 1; }
    echo
    mkdir -p "$AGENTS"
    for L in $LABELS; do
      plutil -lint "$HERE/launchd/$L.plist" >/dev/null || { echo "plist 语法错误: $L" >&2; exit 1; }
      cp "$HERE/launchd/$L.plist" "$AGENTS/$L.plist"
      launchctl bootout "gui/$UID_N/$L" 2>/dev/null
      launchctl bootstrap "gui/$UID_N" "$AGENTS/$L.plist" || { echo "加载失败: $L" >&2; exit 1; }
      echo "已加载 $L"
    done
    echo
    echo "下一次触发:"
    for L in $LABELS; do
      launchctl print "gui/$UID_N/$L" 2>/dev/null | grep -E "next fire|state =" | sed "s/^/  $L: /"
    done
    ;;

  remove)
    for L in $LABELS; do
      launchctl bootout "gui/$UID_N/$L" 2>/dev/null && echo "已卸载 $L" || echo "$L 未加载"
      rm -f "$AGENTS/$L.plist"
    done
    ;;

  status)
    exec /bin/sh "$HERE/status.sh"
    ;;

  *)
    echo "usage: $0 [install|remove|status]" >&2; exit 64 ;;
esac
