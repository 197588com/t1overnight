# -*- coding: utf-8 -*-
"""One-shot daily production run: fetch today's bar, rebuild, then emit the signal.

Why the fetch step exists
-------------------------
The strategy scores the *next* session's open from the bar of the *last completed
session*, so a run whose data stops one day short does not fail loudly -- it
re-emits yesterday's pick under today's date.  That is the single most dangerous
failure mode here, so this script refuses to emit anything unless the dataset's
last bar is today.

Timing policy (why the deadline is 21:00 and not "wait forever")
---------------------------------------------------------------
tushare settles ``daily`` 15:00-16:00 and ``daily_basic`` 15:00-17:00 in batches,
so 17:20 covers the normal case with room to spare.  The signal is only needed at
the *next* session's 09:15 auction, which leaves an 18-hour window -- there is
nothing to gain from finishing early, only risk.

    evening run   17:20 -> retry until 21:00, then park the day as pending
    morning run   08:00 -> retry until 08:40, the last chance before the auction

If the morning run still cannot get a complete frame the day is *abandoned*:
``NO_SIGNAL_<date>.md`` is written and no signal is produced.  Falling back to
the previous session's bars would silently turn a data problem into a wrong
trade, which is far worse than sitting out one day.

    python scripts/daily_run.py                 # evening, deadline 21:00
    python scripts/daily_run.py --morning       # next-morning retry, deadline 08:40
    python scripts/daily_run.py --date 20261023 # replay a specific session
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

PY = sys.executable
# Overridable so the same code can run against a copy of the pool on another
# machine (see deploy/).  The default keeps this machine working untouched.
DATAPOOL = os.environ.get(
    "T1_DATAPOOL", os.path.expanduser("~/datapool/parquet"))

EXIT_OK = 0
EXIT_PENDING = 2       # data incomplete, no signal produced
EXIT_HOLIDAY = 3       # not a trading day

# Where the one-line "this run happened" record goes.  main() points it at
# --out-dir so that a replay into a sandbox directory does not leave a heartbeat
# in the production log -- a fake beat there reads exactly like a real run and
# defeats the whole reason the line exists.
HEARTBEAT = ROOT / "outputs" / "live" / "heartbeat.log"


# --------------------------------------------------------------------------- #
# run state: one small json so the morning run knows what the evening run did
# --------------------------------------------------------------------------- #
def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = sorted(state.items())[-60:]          # a quarter of history is plenty
    path.write_text(json.dumps(dict(keep), ensure_ascii=False, indent=1),
                    encoding="utf-8")


def newest_bar(dataset: Path) -> pd.Timestamp:
    t = pd.read_parquet(dataset, columns=["date"])
    return pd.Timestamp(t["date"].max())


def acquire_lock(path: Path):
    """Hold an exclusive lock for the life of the process, or return None.

    More than one scheduler is allowed to fire this script (launchd and the
    assistant's own timer), and the evening run can legitimately sit in its
    retry loop for hours.  Without a lock the second copy would start fetching
    into the same append file and write the same ledger row twice.

    ``flock`` is used rather than a pid file because the kernel drops it when
    the process dies -- a crashed run can never leave a stale lock behind and
    block the next day.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # "a+" not "w": opening with "w" truncates before the lock attempt, so a
    # loser would erase the winner's pid and then report an empty holder.
    fh = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid={os.getpid()} at={pd.Timestamp.now():%Y-%m-%d %H:%M:%S}\n")
    fh.flush()
    return fh


def run(cmd: list[str]) -> int:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd], cwd=str(ROOT))


def write_no_signal(out_dir: Path, day: pd.Timestamp, reason: str,
                    rows: int, attempts: int) -> Path:
    """Written only when the last chance is gone.  Humans read this, not machines."""
    p = out_dir / f"NO_SIGNAL_{day.strftime('%Y-%m-%d')}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"# {day.date()} 未出信号 —— 今日空仓\n\n"
        f"**不要买入任何股票。** 不是\"买别的\"，是今天不做。\n\n"
        f"- 原因：{reason}\n"
        f"- 取数行数：{rows:,}\n"
        f"- 重试次数：{attempts}\n"
        f"- 判定时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}\n\n"
        f"## 为什么不降级用旧数据\n\n"
        f"数据缺一天时脚本仍能跑通，只是会把上一个交易日的选股当成今天的发出去。\n"
        f"那种错误在账本上看起来完全正常，但买入的是一只与当日市场无关的票。\n"
        f"所以宁可空仓一天。\n\n"
        f"## 手工排查\n\n"
        f"1. `python scripts/fetch_today.py --date {day.strftime('%Y%m%d')}` 看是否已补齐\n"
        f"2. 若已补齐：`python scripts/daily_run.py --date {day.strftime('%Y%m%d')} --skip-fetch`\n"
        f"3. 若开盘前仍补不齐：今日不做，等下一个信号\n",
        encoding="utf-8")
    return p


def abort_pending(state: dict, state_path: Path, out_dir: Path, day: pd.Timestamp,
                  reason: str, rows: int = 0, attempts: int = 0,
                  morning: bool = False) -> "NoReturn":
    """Park the day as pending and exit 2.  Every data-side failure lands here.

    ``SystemExit("text")`` would exit 1 and print an unlabelled traceback-ish
    message, which the scheduler cannot tell apart from a code bug -- so the
    message is printed here and the exit code is set explicitly.
    """
    key = day.strftime("%Y-%m-%d")
    state[key] = {"status": "pending", "reason": reason, "rows": int(rows),
                  "attempts": int(attempts),
                  "last_try": pd.Timestamp.now().isoformat(timespec="seconds"),
                  "final": bool(morning)}
    save_state(state_path, state)

    msg = (f"当日（{day.date()}）数据未就绪，未出信号。\n"
           f"原因：{reason}\n取数 {rows:,} 行 · 重试 {attempts} 次")
    if morning:
        p = write_no_signal(out_dir, day, reason, rows, attempts)
        msg += (f"\n\n这是开盘前最后一次机会，已宣告今日空仓：{p}\n"
                f"不要用旧数据补，也不要改买别的票——今天不做。")
    else:
        msg += "\n\n当晚窗口已关闭，已挂起，等待次日早间补跑。"
    print(msg, file=sys.stderr, flush=True)
    raise SystemExit(EXIT_PENDING)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DATAPOOL)
    ap.add_argument("--dataset", default="outputs/t1_dataset.parquet")
    ap.add_argument("--append-dir", default="outputs/append")
    ap.add_argument("--boards", default=os.environ.get("T1_BOARDS", "0,1,2"))
    ap.add_argument("--tier", default=os.environ.get("T1_TIER", "主板+创业板"))
    ap.add_argument("--out-dir", default="outputs/live")
    ap.add_argument("--date", default=None,
                    help="session to produce a signal for; default = today")
    ap.add_argument("--until", default=None,
                    help="absolute HH:MM retry deadline; default 21:00 evening / 08:40 morning")
    ap.add_argument("--morning", action="store_true",
                    help="next-morning retry: shorter deadline, announces a no-signal day "
                         "if this last chance fails")
    ap.add_argument("--force", action="store_true",
                    help="re-emit even if this session already produced a signal")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-settle", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=200,
                    help="debug: cap the retry loop (default 200)")
    args = ap.parse_args()

    until = args.until or ("08:40" if args.morning else "21:00")
    ds = ROOT / args.dataset
    out_dir = ROOT / args.out_dir
    global HEARTBEAT
    HEARTBEAT = out_dir / "heartbeat.log"      # absolute when --out-dir is absolute
    state_path = out_dir / "run_state.json"
    day = pd.Timestamp(args.date) if args.date else pd.Timestamp.now().normalize()
    if args.morning and args.date is None:
        # The morning run exists to recover *yesterday's* session that the evening
        # run could not read.  Without this shift it used to ask the sources for
        # today -- a bar that cannot exist before the open -- so every morning
        # ended in a false "no trade today" declaration.
        from fetch_today import prev_trade_day                 # noqa: E402
        raw = day.strftime("%Y%m%d")
        try:
            pv = prev_trade_day(raw)
        except Exception:                                      # noqa: BLE001
            pv = None
        if not pv:                                             # offline fallback
            pv = raw
            for _ in range(10):
                pv = (pd.Timestamp(pv) - pd.Timedelta(days=1)).strftime("%Y%m%d")
                if pd.Timestamp(pv).weekday() < 5:
                    break
        print(f"[cal] 晨补目标：上一交易日 {pd.Timestamp(pv).date()}"
              f"（今天是 {day.date()}）", flush=True)
        day = pd.Timestamp(pv)
    stamp = day.strftime("%Y%m%d")
    key = day.strftime("%Y-%m-%d")
    t0 = time.time()

    # -- 0a. only one runner at a time ---------------------------------------
    lock = acquire_lock(out_dir / ".run.lock")
    if lock is None:
        holder = ""
        try:
            holder = (out_dir / ".run.lock").read_text(encoding="utf-8").strip()
        except OSError:
            pass
        print(f"[lock] 已有实例在跑（{holder}），本次让位退出。", flush=True)
        raise SystemExit(EXIT_OK)

    state = load_state(state_path)

    # -- 0. already done? ----------------------------------------------------
    prev = state.get(key, {})
    if prev.get("status") == "ok" and not args.force:
        print(f"[skip] {key} 已出过信号（{prev.get('at', '?')} 生成，"
              f"{prev.get('code', '?')}），本次不重复执行。", flush=True)
        print(f"[skip] 如需重跑：--force", flush=True)
        raise SystemExit(EXIT_OK)

    # -- 1. trading calendar -------------------------------------------------
    from fetch_today import is_trade_day, wait_until      # noqa: E402

    trade_day = is_trade_day(stamp)
    print(f"[cal] {day.date()} 交易日={trade_day} | 模式={'次日晨补跑' if args.morning else '当晚'} "
          f"| 重试截止 {until}", flush=True)
    if not trade_day:
        print("[skip] 非交易日，不出信号。", flush=True)
        raise SystemExit(EXIT_HOLIDAY)

    # -- 2. today's bar ------------------------------------------------------
    src_used = "unknown"
    if not args.skip_fetch:
        info = wait_until(stamp, until, max_attempts=args.max_attempts)
        if not info["ok"]:
            abort_pending(state, state_path, out_dir, day,
                          info.get("reason", "未知"),
                          int(info.get("rows", 0)), int(info.get("attempts", 0)),
                          morning=args.morning)
        src_used = info.get("source", "unknown")
        note = info.get("agreement", "")
        # A frame corroborated by a second provider is a different thing from one
        # that arrived alone.  Say which, every time, rather than letting a
        # degraded source blend into a normal-looking run.
        if "互证" in src_used:
            print(f"[source] {src_used}｜{note}", flush=True)
        else:
            print(f"[source!] 单源 {src_used}，无第二源互证｜{note}"
                  f"\n          决策仍可用（已验证换源不影响头部排名），但这天的账要留意。", flush=True)

    # -- 3. rebuild ----------------------------------------------------------
    if not args.skip_build:
        if not Path(args.source).exists():
            abort_pending(state, state_path, out_dir, day,
                          f"数据池不存在：{args.source}", morning=args.morning)
        rc = run([PY, "scripts/build_dataset.py", "--source", args.source,
                  "--out", args.dataset, "--append-dir", args.append_dir])
        if rc != 0:
            abort_pending(state, state_path, out_dir, day,
                          f"build_dataset 失败 (rc={rc})", morning=args.morning)
        print(f"[build] done in {(time.time()-t0)/60:.1f} min", flush=True)

    if not ds.exists():
        abort_pending(state, state_path, out_dir, day,
                      f"数据集缺失：{ds}", morning=args.morning)

    # -- 4. freshness gate ---------------------------------------------------
    bar = newest_bar(ds).normalize()
    print(f"\n[check] dataset last bar = {bar.date()} | session = {day.date()}")
    if bar != day:
        if args.skip_build or args.skip_fetch:
            print(f"[warn] 调试模式：数据集最后一天 {bar.date()} != {day.date()}，"
                  f"信号基准日仍为 {bar.date()}", flush=True)
        else:
            abort_pending(state, state_path, out_dir, day,
                          f"数据集最后一天 {bar.date()} != {day.date()}，"
                          f"append 层未带入当日行情", morning=args.morning)

    # -- 4b. did the pool or our own fetch win today? -------------------------
    # build_dataset keeps the *pool* row on a (code, date) collision and only
    # fills dates the pool is missing.  That is the right default (the pool is
    # the cleaned long history), but it means a half-settled row that someone
    # saved into the pool earlier in the afternoon would silently replace the
    # frame this run just verified with three sources -- and the freshness gate
    # below only looks at the date, so it would pass.  Say it out loud instead.
    ap_file = ROOT / args.append_dir / f"{stamp}.parquet"
    if not args.skip_fetch and ap_file.exists():
        try:
            fresh = pd.read_parquet(ap_file, columns=["code", "close"])
            fresh["code"] = fresh["code"].astype(str).str.zfill(6)
            built = pd.read_parquet(ds, columns=["date", "code", "close"])
            built = built[pd.to_datetime(built["date"]).dt.normalize() == day]
            built = built.assign(code=built["code"].astype(str).str.zfill(6))
            m = built.merge(fresh, on="code", suffixes=("_pool", "_fetch"))
            if not len(m):
                print(f"[pool!] 数据池当日 {len(built)} 行与本次抓取 {len(fresh)} 行"
                      f"没有一只重合 —— 当天行情来源异常，请人工看一眼", flush=True)
            else:
                same = (m["close_pool"] - m["close_fetch"]).abs() / m["close_fetch"] <= 0.002
                print(f"[pool] 当日 池 {len(built)} 行 / 抓 {len(fresh)} 行 / "
                      f"重合 {len(m)} 只，收盘价一致 {same.mean():.2%}", flush=True)
                if len(m) < 0.95 * len(fresh) or same.mean() < 0.98:
                    print(f"[pool!] 数据池当日的行数或收盘价与本次抓取不一致 —— "
                          f"数据集用的是池里的旧数，不是本次验证过的三源快照", flush=True)
        except Exception as exc:                            # noqa: BLE001
            print(f"[pool] 池/抓取一致性检查跳过（{type(exc).__name__}: {exc}）", flush=True)

    # -- 5. signal -----------------------------------------------------------
    # Grade earlier rows first: the next day's open is what closes a trade, and
    # it only exists once this rebuild has run.  Doing it before emitting today's
    # signal keeps the 60-day forward gate moving even on a day with no signal.
    if not args.skip_settle:
        run([PY, "scripts/settle_ledger.py",
             "--ledger", str(out_dir / "forward_ledger.csv"),
             "--dataset", args.dataset])

    rc = run([PY, "scripts/daily_signal.py",
              "--dataset", args.dataset,
              "--boards", args.boards,
              "--tier", args.tier,
              "--out-dir", args.out_dir])
    if rc != 0:
        abort_pending(state, state_path, out_dir, day,
                      f"daily_signal 失败 (rc={rc})", morning=args.morning)

    # -- 6. record success ---------------------------------------------------
    code = name = ""
    try:
        led = pd.read_csv(out_dir / "forward_ledger.csv", dtype={"code": str})
        hit = led[(led["signal_bar"] == key) & (led["tier"] == args.tier)]
        if len(hit):
            code, name = str(hit.iloc[-1]["code"]), str(hit.iloc[-1]["name"])
    except (OSError, KeyError, IndexError):
        pass

    state[key] = {
        "status": "ok",
        "signal_bar": bar.strftime("%Y-%m-%d"),
        "source": src_used,
        "rows": int(len(pd.read_parquet(ROOT / args.append_dir / f"{stamp}.parquet",
                                        columns=["code"])))
        if (ROOT / args.append_dir / f"{stamp}.parquet").exists() else 0,
        "code": code, "name": name,
        "at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    save_state(state_path, state)

    print(f"\n[summary] 信号基准日 {bar.date()} · 权限档 {args.tier} (boards {args.boards})"
          f" · 输出 {args.out_dir} · 耗时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"[state] {key} -> ok ({code} {name})")
    raise SystemExit(EXIT_OK)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        # One line per invocation, whatever the outcome and whoever started it.
        # Keeping this here rather than in a scheduler wrapper means the record
        # survives a change of scheduler: "the job fired and found nothing" and
        # "the job never fired" must never look the same in the log.
        try:
            code = exc.code if isinstance(exc.code, int) else 1
            beat = HEARTBEAT
            beat.parent.mkdir(parents=True, exist_ok=True)
            mode = "morning" if "--morning" in sys.argv else "evening"
            with open(beat, "a", encoding="utf-8") as fh:
                fh.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}\t{mode}\trc={code}\n")
        except OSError:
            pass
        raise
