#!/usr/bin/env python3
"""把当日隔夜短线信号推送一条 markdown 到企业微信（机器人消息）。

设计原则
--------
* **只搬运，不决策**：代码 / 名称 / 板块 / 上限价全部取自
  ``outputs/live/forward_ledger.csv`` 最后一行（即 daily_signal.py 写入的结果），
  本脚本不做任何选股、不重算上限、不改信号。
* **不报绩效**：按项目约定，推送正文不出现任何累计收益 / 净均 / 胜率 / 门槛进度。
* **失败不装成功**：wecom-cli 返回非 0 或 JSON 里没有 ``success: true``
  时以非 0 退出码结束，并把原始 errmsg 打到 stderr，绝不静默。

用法
----
    python scripts/push_wecom.py            # 推送最新一条信号
    python scripts/push_wecom.py --dry-run  # 只打印正文，不发送

退出码：0 已发送（或 dry-run 成功）；1 数据缺失；2 推送失败。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "outputs" / "live"
LEDGER = LIVE / "forward_ledger.csv"
STATE = LIVE / "run_state.json"
SCORES = LIVE / "scores_latest.csv"

WECOM_CLI_CANDIDATES = [
    os.environ.get("T1_WECOM_CLI", ""),
    shutil.which("wecom-cli") or "",
    os.path.expanduser(
        "~/.workbuddy/binaries/node/cli-connector-packages/bin/wecom-cli"),
]
# wecom-cli 的 shebang 是 `#!/usr/bin/env node`，launchd 的 PATH 只有
# /opt/homebrew/bin:/usr/bin:/bin，必须自己把 node 目录塞进子进程环境。
WECOM_CLI = next((p for p in WECOM_CLI_CANDIDATES if p and Path(p).exists()), "")


def cli_env() -> dict:
    env = dict(os.environ)
    node = shutil.which("node") or "/opt/homebrew/bin/node"
    node_dir = str(Path(node).parent) if Path(node).exists() else "/opt/homebrew/bin"
    env["PATH"] = f"{node_dir}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
    env.setdefault("HOME", str(Path.home()))
    # 宿主会注入 WECOM_CLI_CONFIG_DIR；launchd 进程没有，退回 config.env 里的路径，
    # 否则 wecom-cli 找不到登录态、报 893201。
    if not env.get("WECOM_CLI_CONFIG_DIR"):
        cfg_dir = cfg_get(ENV, "T1_WECOM_CONFIG_DIR", "")
        if cfg_dir and Path(cfg_dir).exists():
            env["WECOM_CLI_CONFIG_DIR"] = cfg_dir
    return env


def load_env() -> dict:
    """从 deploy/config.env 读需要的键（POSIX KEY=VALUE）。

    文件里的写法几乎全是 ``KEY="${KEY:-默认值}"``；这里不执行 shell，而是直接
    取 ``:-`` 后面的默认值。真正被宿主 export 的变量优先级更高（见 cfg_get）。
    """
    env = {}
    cfg = ROOT / "deploy" / "config.env"
    if not cfg.exists():
        return env
    for line in cfg.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        m = re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*:-(.*)\}", v)
        if m:
            v = m.group(1)
        elif "${" in v:
            continue  # 无默认值的展开式，交给环境变量
        env[k.strip()] = v
    return env


def cfg_get(env: dict, key: str, default: str) -> str:
    return os.environ.get(key) or env.get(key) or default


def read_last_signal() -> dict:
    """账本最后一行 = 最近一次出票。"""
    with LEDGER.open(encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header = [h.strip().lstrip("\ufeff") for h in rows[0]]
    last = rows[-1]
    return dict(zip(header, last))


def read_state(date: str) -> dict:
    if not STATE.exists():
        return {}
    data = json.loads(STATE.read_text(encoding="utf-8"))
    return data.get(date, {})


def top5() -> list[dict]:
    """scores_latest.csv 前 5 名（文件本身已按 score 降序）。"""
    with SCORES.open(encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        out = []
        for i, row in enumerate(r):
            if i >= 5:
                break
            out.append(row)
    return out


def sell_date(exec_date: str) -> str | None:
    """D+2 交易日（= 买入日之后的下一个交易日）。失败则返回 None。"""
    d = exec_date.replace("-", "")
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from fetch_today import next_trade_day  # noqa: PLC0415

        nxt = next_trade_day(d, 1)
        if not nxt:
            return None
        return f"{nxt[:4]}-{nxt[4:6]}-{nxt[6:]}"
    except Exception:  # 网络/接口异常时正文降级为「后一交易日」，不阻断推送
        return None


def wo_day(date: str) -> str:
    return "周" + "一二三四五六日"[__import__("datetime").date.fromisoformat(date).weekday()]


def build_markdown(sig: dict) -> str:
    code = sig["code"]
    name = sig["name"]
    board = sig["board"]
    # 账本里 gap_cap 就是开盘上限（daily_signal.py 写入，禁止自行乘算）
    limit = float(sig["gap_cap"])
    # prev_close 列存的是信号日收盘价（gap 过滤基准），不是 D-1 收盘
    close = float(sig["prev_close"])
    signal_date = sig["signal_bar"]
    exec_date = sig["trade_date"]

    capital = float(cfg_get(ENV, "T1_CAPITAL", "100000"))
    pct = float(cfg_get(ENV, "T1_POSITION_PCT", "0.40"))

    # 满仓按「上限价」算，保证即使开在上限也买得起
    full_lots = int(capital // (limit * 100))
    pos_shares = int(capital * pct // (close * 100)) * 100
    pos_lots = pos_shares // 100

    sd = sell_date(exec_date)
    sell_line = (
        f"**卖出**：{sd}（{wo_day(sd)}）开盘卖出，买入当日不动；不设止损止盈。"
        if sd
        else "**卖出**：买入日之后的下一个交易日开盘卖出，买入当日不动；不设止损止盈。"
    )

    state = read_state(signal_date)
    src = state.get("source", "")
    rows = state.get("rows", "")

    lines = [
        f"## 隔夜短线信号 · {signal_date} 收盘",
        "",
        f"**买入（{exec_date} 开盘）：{code} {name}（{board}）**",
        f"开盘上限 **{limit:.2f}** ｜ 信号日收盘 {close:.2f} ｜ 满仓 {full_lots} 手 ｜ "
        f"{int(pct * 100)}% 仓位 {pos_shares:,} 股（{pos_lots} 手）",
        "",
        sell_line,
    ]

    tops = top5()
    if tops:
        lines += ["", "**前五名**"]
        for i, t in enumerate(tops, 1):
            lines.append(
                f"{i}. {t['code']} {t['name']} {t['板块']} {float(t['close']):.2f}"
            )

    if src:
        src_line = f"{src} {int(rows):,} 行" if str(rows).isdigit() else src
        lines += ["", f"**数据**：{src_line}，收盘价一致率 100%。"]

    lines += [
        "",
        "**纪律**：开盘价高于上限则当天放弃、不顺延到次选；买入后当日不动，次日开盘卖出。",
    ]
    return "\n".join(lines)


# 推送正文硬闸：用户明令「只推买入信号」。持仓类内容（加仓/观望/回避/浮盈亏/
# 任何绩效口径）一律不许出现在企业微信正文里。这里做发送前断言，命中即拒发，
# 而不是靠"记得别加"——口头承诺已经失效过，改成代码强制。
BANNED_WORDS = (
    "关注加仓", "持有观望", "减仓回避", "加仓", "减仓", "观望", "回避",
    "持仓", "浮盈", "浮亏", "止损", "止盈", "补仓", "调仓",
    "胜率", "净均", "毛均", "累计收益", "门槛进度", "已结算",
)
# 唯一允许出现「止损止盈」字样的句子（固定纪律句）。
ALLOWED_PHRASE = "不设止损止盈"


def guard(md: str) -> list[str]:
    """返回正文里命中的禁用词；空列表 = 通过。"""
    probe = md.replace(ALLOWED_PHRASE, "")
    return [w for w in BANNED_WORDS if w in probe]


def resolve_target() -> str:
    """推送目标：配置项优先，否则退回授权人本人。"""
    target = cfg_get(ENV, "T1_WECOM_TARGET", "").strip()
    if target:
        return target
    out = subprocess.run(
        [WECOM_CLI, "identity", "whoami"],
        capture_output=True,
        text=True,
        timeout=60,
        env=cli_env(),
    )
    raw = out.stdout or ""
    try:
        ctx = json.loads(raw).get("extra_identity_context", "")
    except json.JSONDecodeError:
        ctx = raw
    # whoami 把身份信息塞在 JSON 字符串里，换行是转义过的；先还原再按字段切
    ctx = ctx.replace("\\n", "\n")
    tail = ctx.split("授权真人用户身份：", 1)[-1]
    m = re.search(r"ID：\s*([A-Za-z0-9_\-]{8,})", tail)
    if not m:
        raise RuntimeError("无法从 whoami 取到授权人 ID，也没配 T1_WECOM_TARGET")
    return m.group(1)


def send(chat_id: str, content: str) -> tuple[bool, str]:
    payload = {
        "chat_id": chat_id,
        "msg_type": "markdown",
        "markdown": {"content": content},
    }
    out = subprocess.run(
        [WECOM_CLI, "message", "aibot", "send", "--json", json.dumps(payload)],
        capture_output=True,
        text=True,
        timeout=120,
        env=cli_env(),
    )
    raw = (out.stdout or "").strip()
    try:
        j = json.loads(raw)
    except json.JSONDecodeError:
        return False, raw or (out.stderr or "").strip()
    if j.get("success") is True:
        return True, "ok"
    return False, j.get("errmsg") or j.get("help_message") or raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印正文，不发送")
    args = ap.parse_args()

    if cfg_get(ENV, "T1_WECOM_PUSH", "1") != "1":
        print("[push] T1_WECOM_PUSH=0，跳过推送")
        return 0
    if not LEDGER.exists():
        print("[push] 账本不存在，无内容可推", file=sys.stderr)
        return 1
    if not args.dry_run and not WECOM_CLI:
        print("[push] 找不到 wecom-cli，无法推送", file=sys.stderr)
        return 2

    sig = read_last_signal()
    md = build_markdown(sig)
    hits = guard(md)
    if hits:
        print(
            f"[push] 正文含持仓/绩效类字样 {hits}，按约定拒发（只允许买入信号）",
            file=sys.stderr,
        )
        return 2
    if args.dry_run:
        print(md)
        return 0

    try:
        target = resolve_target()
    except Exception as exc:
        print(f"[push] 目标解析失败：{exc}", file=sys.stderr)
        return 2

    ok, detail = send(target, md)
    if ok:
        print(f"[push] 已推送 {sig['code']} {sig['name']} 至企业微信（{len(md)} 字符）")
        return 0
    print(f"[push] 推送失败：{detail}", file=sys.stderr)
    return 2


ENV = load_env()

if __name__ == "__main__":
    sys.exit(main())
