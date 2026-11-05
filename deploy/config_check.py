#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置体检：确认"配置里写的"和"机器上真在跑的"是同一件事。

为什么需要它
------------
这套 pipeline 的失败方式不是报错，是**静默跑偏**：

  * 迁移了项目目录，plist 还指向旧路径 —— 定时器照常触发、照常退出 0，
    只是跑的是另一个目录里的旧代码，甚至是一个不存在的目录；
  * 改了 config.env 的门槛，status.sh 的报表还在用旧的写死数字；
  * launchd 备用层从未安装，于是"唯一的调度器"其实是那个只在应用开着时
    才响的定时任务，某天应用没开，当天就无声地没有信号。

这些都留不下 traceback，只在事后看账本时表现为"某几天没交易"。所以把
每一处"两处配置必须相等"的假设写成断言，定期跑，让它先响。

usage:
    <T1_PY> deploy/config_check.py [--json] [--quiet]
退出码: 0 全部通过 / 1 有告警 / 2 有硬失败
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "deploy" / "config.env"

OK, WARN, FAIL = "ok", "warn", "fail"
RANK = {OK: 0, WARN: 1, FAIL: 2}
rows: list[tuple[str, str, str]] = []      # (level, item, detail)

# 已知的、确认后可接受的"无关调度任务"。放进这里不是忽略，而是把结论固化：
# 每天报同一条已经判过的告警，等于训练人忽略告警本身。不在表里的照常报 WARN。
KNOWN_OTHERS = {
    "com.qiyin.overnight.update":
        "旧项目「寻找利润点和操作时间节点」自用缓存更新（19:35），写自己的 data/ 目录，与"
        " t1overnight 数据池无关；保留。",
    "com.qiyin.overnight.sell":
        "已于 2026-10-26 停用（连续 7 个交易日取不到当日竞价数据）；plist 归档在"
        " ~/Library/LaunchAgents/disabled/。若重新出现，说明被恢复或重装。",
}


def rec(level: str, item: str, detail: str) -> None:
    rows.append((level, item, detail))


def rel(p: Path) -> str:
    """路径可能是项目外的（T1_OUT_DIR 允许指向别处），relative_to 会抛错。"""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------- #
# 读取 config.env：用真的 sh 去 source，而不是自己解析
# --------------------------------------------------------------------------- #
def load_config() -> dict[str, str]:
    """跑 `sh -c '. config.env; env'` 取实际生效的值。

    自己写 KEY=VALUE 解析器会漏掉 ${VAR:-默认} 和末尾的补值逻辑，而"文件能
    不能真的被 sh 加载"本身就是最该验证的一点。
    """
    if not CFG.exists():
        rec(FAIL, "config.env", f"不存在：{CFG}")
        return {}
    out = subprocess.run(
        ["/bin/sh", "-c", f'. "{CFG}" && env'],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if out.returncode != 0:
        rec(FAIL, "config.env", f"无法被 sh 加载：{out.stderr.strip()[:200]}")
        return {}
    env: dict[str, str] = {}
    for line in out.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    rec(OK, "config.env", f"可加载，导出 {sum(k.startswith('T1_') for k in env)} 个 T1_* 变量")
    return env


def check_paths(env: dict[str, str]) -> None:
    root = env.get("T1_ROOT", "")
    if not root:
        rec(FAIL, "T1_ROOT", "未设置")
    elif Path(root).resolve() != ROOT:
        rec(FAIL, "T1_ROOT", f"配置指向 {root}，实际项目根 {ROOT} —— 迁移后未改配置")
    else:
        rec(OK, "T1_ROOT", root)

    py = env.get("T1_PY", "")
    if not py:
        rec(FAIL, "T1_PY", "未设置")
    elif not os.access(py, os.X_OK):
        rec(FAIL, "T1_PY", f"不可执行：{py}")
    else:
        ver = subprocess.run([py, "-V"], capture_output=True, text=True).stdout.strip()
        rec(OK, "T1_PY", f"{py} ({ver})")

    pool = Path(env.get("T1_DATAPOOL", ""))
    if not pool.is_dir():
        rec(FAIL, "数据池", f"目录不存在：{pool}")
    else:
        n = len(list(pool.glob("*.parquet")))
        rec(OK, "数据池", f"{pool} · {n} 个 parquet")

    uni = Path(env.get("T1_UNIVERSE", ""))
    if not uni.is_file():
        rec(FAIL, "T1_UNIVERSE", f"文件不存在：{uni}")
    elif uni.stat().st_size == 0:
        rec(FAIL, "T1_UNIVERSE", f"文件为空：{uni}")
    else:
        rec(OK, "T1_UNIVERSE", f"{uni.name} · {uni.stat().st_size / 1024:.0f} KB")

    for key in ("T1_DATASET", "T1_APPEND_DIR", "T1_OUT_DIR", "T1_MODELS"):
        p = ROOT / env.get(key, "")
        base = p.parent if key != "T1_DATASET" else p.parent
        if base.is_dir():
            rec(OK, key, str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p))
        else:
            rec(WARN, key, f"父目录不存在（首次运行前可忽略）：{base}")


# --------------------------------------------------------------------------- #
# 脚本内嵌的缺省值 必须与 config.env 相等
# --------------------------------------------------------------------------- #
def check_script_defaults(env: dict[str, str]) -> None:
    probes = [
        ("scripts/daily_signal.py", r'os\.environ\.get\("T1_GAP_MAX",\s*"([^"]+)"\)', "T1_GAP_MAX"),
        ("scripts/daily_run.py", r'os\.environ\.get\("T1_BOARDS",\s*"([^"]+)"\)', "T1_BOARDS"),
        ("scripts/daily_run.py", r'os\.environ\.get\("T1_TIER",\s*"([^"]+)"\)', "T1_TIER"),
        ("scripts/daily_run.py", r'os\.environ\.get\(\s*"T1_DATAPOOL",\s*"([^"]+)"\)', "T1_DATAPOOL"),
    ]
    for rel, pat, key in probes:
        f = ROOT / rel
        if not f.exists():
            rec(WARN, f"{rel} → {key}", "文件不存在，跳过")
            continue
        m = re.search(pat, f.read_text(encoding="utf-8"))
        if not m:
            rec(WARN, f"{rel} → {key}", "未找到 os.environ.get 读取点（可能仍为硬编码）")
            continue
        code_default, cfg_value = m.group(1), env.get(key, "")
        if code_default == cfg_value:
            rec(OK, f"{rel} → {key}", f"缺省值一致 = {cfg_value}")
        else:
            rec(FAIL, f"{rel} → {key}",
                f"代码缺省 {code_default!r} ≠ 配置 {cfg_value!r} —— 未加载 config.env 时会跑出不同行为")


# --------------------------------------------------------------------------- #
# plist 必须与 config.env 渲染结果一致
# --------------------------------------------------------------------------- #
def check_plists() -> None:
    render = ROOT / "deploy" / "render_launchd.sh"
    if not render.exists():
        rec(WARN, "launchd plist", "render_launchd.sh 不存在，跳过漂移检查")
        return
    r = subprocess.run(["/bin/sh", str(render), "--check"], capture_output=True, text=True)
    if r.returncode == 0:
        rec(OK, "launchd plist", "与 config.env 一致")
    else:
        rec(FAIL, "launchd plist", (r.stderr or r.stdout).strip().splitlines()[0][:160])

    for name in ("com.t1overnight.evening.plist", "com.t1overnight.morning.plist"):
        p = ROOT / "deploy" / "launchd" / name
        if not p.exists():
            rec(FAIL, name, "文件缺失")
            continue
        lint = subprocess.run(["plutil", "-lint", str(p)], capture_output=True, text=True)
        if lint.returncode == 0:
            rec(OK, name, "plutil -lint 通过")
        else:
            rec(FAIL, name, lint.stdout.strip()[:160])


# --------------------------------------------------------------------------- #
# 调度器：谁真的在跑
# --------------------------------------------------------------------------- #
def check_schedulers() -> None:
    uid = os.getuid()
    if not shutil.which("launchctl"):
        rec(WARN, "launchd", "非 macOS 或无 launchctl，跳过")
        return

    loaded, zombies = [], []
    try:
        listing = subprocess.run(["launchctl", "list"], capture_output=True, text=True).stdout
    except OSError:
        listing = ""
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) >= 3 and "overnight" in parts[2]:
            (loaded if parts[2].startswith("com.t1overnight.") else zombies).append(parts[2])

    if loaded:
        rec(OK, "launchd 备用层", f"已加载 {len(loaded)} 个：{', '.join(sorted(loaded))}")
    else:
        rec(WARN, "launchd 备用层",
            "未加载 —— 当前只有 WorkBuddy 定时在跑；应用未运行时当天不会有信号。"
            "启用：sh deploy/install_launchd.sh install")

    for z in sorted(zombies):
        if z in KNOWN_OTHERS:
            rec(OK, "无关调度任务", f"{z} 已判定可接受：{KNOWN_OTHERS[z]}")
            continue
        d = subprocess.run(["launchctl", "print", f"gui/{uid}/{z}"],
                           capture_output=True, text=True).stdout
        m = re.search(r"working directory = (.+)", d)
        rec(WARN, "无关调度任务",
            f"{z} 仍已加载（目录 {m.group(1).strip() if m else '?'}）—— 不产出本项目信号，但会干扰排查")

    # 心跳：调度器"触发过"与"跑成功"必须分开看
    hb = ROOT / os.environ.get("T1_OUT_DIR", "outputs/live") / "heartbeat.log"
    if not hb.exists():
        rec(WARN, "心跳", f"不存在：{hb}（从未被调度器触发过？）")
    else:
        lines = [x for x in hb.read_text(encoding="utf-8").splitlines() if x.strip()]
        rec(OK, "心跳", f"{len(lines)} 条记录，最后一条：{lines[-1].split(chr(9))[0]}")


# --------------------------------------------------------------------------- #
# 产物与日志
# --------------------------------------------------------------------------- #
def check_artifacts(env: dict[str, str]) -> None:
    out = ROOT / env.get("T1_OUT_DIR", "outputs/live")
    led = out / "forward_ledger.csv"
    if led.exists():
        rec(OK, "前向账本", f"{rel(led)} · {led.stat().st_size / 1024:.0f} KB")
    else:
        rec(WARN, "前向账本", f"不存在：{rel(led)}")

    logs = out / "logs"
    if logs.is_dir():
        files = list(logs.glob("*"))
        size = sum(f.stat().st_size for f in files) / 1024 / 1024
        err = logs / "launchd_evening.err"
        rec(OK, "日志目录", f"{len(files)} 个文件 · {size:.1f} MB")
        if err.exists() and err.stat().st_size > 5 * 1024 * 1024:
            rec(WARN, "launchd_evening.err", "超过 5 MB —— 轮转只在 run_daily.sh 跑成功时裁剪")
    else:
        rec(WARN, "日志目录", f"不存在：{rel(logs)}（首次运行后生成）")


def main() -> int:
    as_json = "--json" in sys.argv
    quiet = "--quiet" in sys.argv

    env = load_config()
    if env:
        check_paths(env)
        check_script_defaults(env)
        check_plists()
    check_schedulers()
    if env:
        check_artifacts(env)
        # check_positions() 已退役：实盘台账不属纯影子前测流程（2026-10-28）

    worst = max((RANK[l] for l, _, _ in rows), default=0)
    code = 0 if worst == 0 else (1 if worst == 1 else 2)

    if as_json:
        print(json.dumps({
            "root": str(ROOT),
            "exit_code": code,
            "fail": sum(l == FAIL for l, _, _ in rows),
            "warn": sum(l == WARN for l, _, _ in rows),
            "ok": sum(l == OK for l, _, _ in rows),
            "items": [{"level": l, "item": i, "detail": d} for l, i, d in rows],
        }, ensure_ascii=False, indent=2))
        return code

    mark = {OK: " OK ", WARN: "WARN", FAIL: "FAIL"}
    if not quiet:
        print(f"配置体检 · {ROOT}")
        print("=" * 72)
    for level, item, detail in rows:
        print(f"[{mark[level]}] {item:<24} {detail}")
    if not quiet:
        f = sum(l == FAIL for l, _, _ in rows)
        w = sum(l == WARN for l, _, _ in rows)
        print("=" * 72)
        print(f"结论：{f} 项失败 / {w} 项告警 → "
              + {0: "配置一致，可以按现状运行", 1: "可用，但有不一致需要看",
                 2: "存在硬失败，先修再跑"}[code])
    return code


if __name__ == "__main__":
    sys.exit(main())
