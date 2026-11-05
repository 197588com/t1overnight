# -*- coding: utf-8 -*-
"""Pull one trading day's full-market snapshot into an append layer.

Two independent sources
-----------------------
tushare is the reference series (every backtest was calibrated on it) but it is
no longer the only one.  ``fetch_sina`` reads the same market through a different
operator, and each source's blind spot is exactly what the other one sees:

    tushare   settles in batches -> HTTP 200 carrying only part of the market
    sina      has no date field  -> can return the wrong session, fully formed

Every attempt therefore queries both and compares them name-by-name.  A partial
tushare frame shows up as a row-count gap; a wrong-day Sina frame shows up as a
close-price disagreement.  Verified on 2026-10-23: 5,571 names in common, prices
agreed to 2.0e-08 relative, zero disagreements above 0.1%.

What the swap costs: Sina's turnover is rounded a few ppm differently, which
LightGBM's step functions amplify.  Measured against tushare on the same session,
scores for rank 1-31 were *identical*, and only 58 of the other 3,468 names moved
by more than 1e-3.  The top of the book -- the only part that gets traded -- does
not move.

Why the datapool is not enough
------------------------------
The local datapool at ``$T1_DATAPOOL`` (default ``~/datapool/parquet``) is
refreshed in bulk, not on every trading day -- its last write landed on a
Saturday mid-day.  Building the dataset straight from it on a weekday evening
therefore risks silently re-using yesterday's bars and mistaking them for today.

tushare settles ``daily`` between 15:00 and 16:00 and ``daily_basic`` between
15:00 and 17:00 (official docs), and the rows arrive in batches.  Anything
pulled before roughly 17:00 can come back *partial*: a valid HTTP 200, a
plausible-looking frame, but only two thirds of the market.  That failure mode
is worse than an error, so ``snapshot()`` grades completeness explicitly and the
caller is expected to honour the verdict.

Units (verified row-by-row against daily_k.parquet for 2026-10-23, 5,572 names):
    daily.vol            hand   -> volume       x 100      (shares)
    daily.amount         k yuan -> amount       x 1000     (yuan)
    daily_basic.turnover_rate  %    -> turnover   / 100     (fraction)
    daily_basic.float_share   万 shares -> outstanding_share x 10000
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_sina                                           # noqa: E402
import fetch_tencent                                        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APPEND_DIR = ROOT / "outputs" / "append"
DATAPOOL = Path(os.environ.get("T1_DATAPOOL", os.path.expanduser("~/datapool/parquet")))

VOL_TO_SHARES = 100.0
AMT_TO_YUAN = 1000.0
TURN_PCT_TO_FRAC = 0.01
WAN_TO_SHARES = 10000.0

MIN_NAMES = 5000          # A-share market is ~5,570 listed names
MIN_COVERAGE = 0.95       # versus the previous trading day's row count

# cross-source agreement: measured at 2.0e-08 relative on 5,571 names, so both
# thresholds sit orders of magnitude away from float noise
CROSS_TOL = 1e-3          # per-name close mismatch tolerance (0.1%)
CROSS_MIN_AGREE = 0.995   # share of names that must agree


def _token() -> str:
    """Environment first, then the shell rc.

    The env var is what a container or a remote host can supply; the rc file is
    the interactive-machine path.  Reading the rc file first would have meant a
    scheduled run could never be given its own credential.
    """
    tok = os.environ.get("TUSHARE_TOKEN", "").strip()
    if tok:
        return tok
    for line in open(os.path.expanduser("~/.zshrc")):
        if "TUSHARE_TOKEN" in line:
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("TUSHARE_TOKEN not found in env or ~/.zshrc")


def api(name: str, params: dict, fields: str = "", retries: int = 2) -> pd.DataFrame:
    """Plain HTTP call to tushare; retries on transport errors only.

    Timeout is deliberately short: the caller (``wait_until``) already runs a
    multi-minute retry loop, so a 60s inner timeout only makes one stalled call
    eat three minutes of that budget.
    """
    tok = _token()
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.post("http://api.tushare.pro",
                              json={"api_name": name, "token": tok,
                                    "params": params, "fields": fields},
                              timeout=25)
            j = r.json()
            if j.get("code") != 0:
                raise RuntimeError(f"tushare {name}: {j.get('msg')}")
            d = j["data"]
            return pd.DataFrame(d["items"], columns=d["fields"])
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt == retries - 1:
                raise
            time.sleep(5)
    raise RuntimeError(str(last))


def is_trade_day(day: str) -> bool:
    """True when ``day`` (YYYYMMDD) is an A-share trading day."""
    d = api("trade_cal", {"exchange": "SSE", "start_date": day, "end_date": day})
    if d.empty:
        return False
    return str(d.iloc[0]["is_open"]) == "1"


def prev_trade_day(day: str) -> str | None:
    """The session immediately before ``day``.

    tushare returns ``trade_cal`` in *descending* order, so taking the last match
    silently yields the oldest day in the window instead of the nearest one.  The
    response also carries ``pretrade_date`` on every row, which is authoritative
    and needs no sorting at all -- prefer it, and keep an explicitly sorted
    fallback for the case where that column is absent.
    """
    d = api("trade_cal", {"exchange": "SSE",
                          "start_date": (pd.Timestamp(day) - pd.Timedelta(days=20)).strftime("%Y%m%d"),
                          "end_date": day})
    if d.empty:
        return None
    if "pretrade_date" in d.columns:
        hit = d[d["cal_date"].astype(str) == day]
        if len(hit):
            pv = str(hit.iloc[0]["pretrade_date"])
            if pv and pv.lower() not in ("none", "nan", "") and pv < day:
                return pv
    opens = sorted(x for x in d[d["is_open"].astype(str) == "1"]["cal_date"].astype(str)
                   if x < day)
    return opens[-1] if opens else None


def next_trade_day(day: str, n: int = 1) -> str | None:
    """The n-th session strictly after ``day`` (YYYYMMDD), or None if unknown.

    Weekday arithmetic is not good enough here.  A signal emitted on the last
    session before a holiday would get a trade_date that is not a session, and
    ``settle_ledger.py`` (which only grades rows whose trade_date exists in the
    dataset) would then never grade it -- the 60-day gate would quietly lose the
    trade instead of one day's data.  This has to come from the exchange
    calendar, not from ``BDay``.
    """
    end = (pd.Timestamp(day) + pd.Timedelta(days=15 + 10 * n)).strftime("%Y%m%d")
    d = api("trade_cal", {"exchange": "SSE", "start_date": day, "end_date": end})
    if d.empty:
        return None
    opens = sorted(x for x in d[d["is_open"].astype(str) == "1"]["cal_date"].astype(str)
                   if x > day)
    return opens[n - 1] if len(opens) >= n else None


def _float_share_fallback() -> pd.Series:
    """Float shares per code from the local pool, for names tushare omits today."""
    p = DATAPOOL / "daily_k.parquet"
    if not p.exists():
        return pd.Series(dtype="float64")
    d = pd.read_parquet(p, columns=["code", "date", "outstanding_share"])
    d["date"] = pd.to_datetime(d["date"])
    d["code"] = d["code"].astype(str).str.zfill(6)
    d = d.dropna(subset=["outstanding_share"])
    last = d.sort_values("date").groupby("code").tail(1)
    return last.set_index("code")["outstanding_share"].astype("float64")


def _pull_tushare(day: str) -> tuple[pd.DataFrame, dict]:
    """Fetch one day from tushare.  Never raises -- a failure is just a verdict."""
    info: dict = {"source": "tushare", "ok": False, "reason": "", "rows": 0}
    try:
        daily = api("daily", {"trade_date": day},
                    "ts_code,open,high,low,close,pre_close,vol,amount")
        if daily.empty:
            info["reason"] = "daily 无数据（尚未入库？）"
            return pd.DataFrame(), info
        for c in ("open", "high", "low", "close", "pre_close", "vol", "amount"):
            daily[c] = pd.to_numeric(daily[c], errors="coerce")

        # daily_basic can lag daily by up to an hour; a missing basic block is
        # recoverable (shares and turnover get filled from Sina downstream), so
        # it must not abort the whole pull
        try:
            basic = api("daily_basic", {"trade_date": day},
                        "ts_code,turnover_rate,float_share,total_mv")
            for c in ("turnover_rate", "float_share", "total_mv"):
                if c in basic:
                    basic[c] = pd.to_numeric(basic[c], errors="coerce")
        except Exception as exc:                              # noqa: BLE001
            info["basic_error"] = f"{type(exc).__name__}: {exc}"[:120]
            basic = pd.DataFrame(columns=["ts_code"])
    except Exception as exc:                                  # noqa: BLE001
        info["reason"] = f"tushare 取数异常 {type(exc).__name__}: {exc}"[:160]
        return pd.DataFrame(), info

    both = daily.merge(basic, on="ts_code", how="left")
    both["code"] = both["ts_code"].str.split(".").str[0].astype(str).str.zfill(6)
    both["date"] = pd.to_datetime(day, format="%Y%m%d")

    # BJ names share the 6-digit space with nothing else, so no prefix tricks needed
    df = pd.DataFrame({
        "code": both["code"],
        "date": both["date"],
        "open": both["open"].astype("float32"),
        "high": both["high"].astype("float32"),
        "low": both["low"].astype("float32"),
        "close": both["close"].astype("float32"),
        "volume": (both["vol"] * VOL_TO_SHARES).astype("float64"),
        "amount": (both["amount"] * AMT_TO_YUAN).astype("float64"),
        "outstanding_share": (both["float_share"] * WAN_TO_SHARES).astype("float64"),
        "turnover": (both["turnover_rate"] * TURN_PCT_TO_FRAC).astype("float64"),
    })

    # names suspended today (or missing from daily_basic) fall back to the last
    # known float share so that market-cap features stay defined
    if df["outstanding_share"].isna().any():
        fb = _float_share_fallback()
        if len(fb):
            need = df["outstanding_share"].isna()
            df.loc[need, "outstanding_share"] = df.loc[need, "code"].map(fb)

    df = df.drop_duplicates("code", keep="last").sort_values("code").reset_index(drop=True)
    info["rows"] = int(len(df))
    info["ok"] = bool(len(df) >= MIN_NAMES and df["outstanding_share"].isna().mean() <= 0.05)
    if not info["ok"]:
        info["reason"] = (f"tushare 仅 {len(df):,} 行或股本缺失 "
                          f"{df['outstanding_share'].isna().mean():.1%}")
    return df, info


def snapshot(day: str, out_dir: Path | None = None) -> dict:
    """Fetch and cache one day from every available source, then decide.

    No source is the default.  Each one is queried, and the winner is chosen by
    *completeness plus agreement*, not by seniority:

        tushare   the series the models were trained on, but documented to settle
                  as late as 17:00 for ``daily_basic`` -- the slowest of the three
        sina      independent operator, quotes stamped 15:00-15:36, 8.2s
        tencent   independent operator, 2.7s, carries its own date on every quote

    The failure mode each source cannot see on its own is what the others catch:

        tushare  settles in batches -> HTTP 200 carrying only part of the market
        sina     has no date field  -> can return the wrong session, fully formed
        tencent  turnover too coarse to score on, needs an external code list

    So: if any two agree on prices, that pair is believed and one of them is used
    (tushare preferred when it is in the pair, since its distribution is the one
    the models learned).  If the reference source disagrees with *everything*
    else while the others agree among themselves, the reference is the one that
    is wrong and is dropped -- that is the only case where three sources beat two.

    Cost when all is well: tushare (1s) + tencent (2.7s) = under 4s, and Sina is
    never called.  Sina only enters when the first two are incomplete or fail to
    agree, which is exactly when a third opinion is worth 8 seconds.
    """
    out_dir = APPEND_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{day}.parquet"
    info: dict = {"date": day, "ok": False, "reason": "", "rows": 0, "source": "",
                  "path": str(out), "fetched_at": pd.Timestamp.now().isoformat(timespec="seconds")}

    if not is_trade_day(day):
        info["reason"] = "非交易日"
        return info

    frames: dict[str, pd.DataFrame] = {}

    ts_df, ts_info = _pull_tushare(day)
    info["tushare"] = dict(ts_info)
    if ts_info["ok"]:
        frames["tushare"] = ts_df

    # Tencent can only pull a list it is handed; tushare's universe is preferred,
    # the local pool is the fallback (it can miss brand-new listings, which the
    # model would rarely pick anyway -- ``days_listed`` is tiny for them).
    codes = ts_df["code"].tolist() if ts_info["ok"] else fetch_tencent.default_codes()
    try:
        tx_info, tx_df = fetch_tencent.snapshot(day, codes, _shares_hint(day))
    except Exception as exc:                                  # noqa: BLE001
        tx_info, tx_df = {"ok": False, "rows": 0,
                          "reason": f"腾讯异常 {type(exc).__name__}: {exc}"[:160]}, pd.DataFrame()
    info["tencent"] = dict(tx_info)
    if tx_info.get("ok"):
        frames["tencent"] = tx_df

    # A third opinion only when the first two cannot settle it between them.
    sn_info: dict = {"ok": False, "reason": "未调用（前两源已互证）", "rows": 0}
    if len(frames) < 2:
        try:
            sn_info, sn_df = fetch_sina.snapshot(day, return_frame=True)
        except Exception as exc:                              # noqa: BLE001
            sn_info, sn_df = {"ok": False, "rows": 0,
                              "reason": f"新浪异常 {type(exc).__name__}: {exc}"[:160]}, pd.DataFrame()
        if sn_info.get("ok"):
            frames["sina"] = sn_df
    info["sina"] = dict(sn_info)

    if not frames:
        info["reason"] = (f"三源均不可用：tushare {ts_info.get('reason', '无')}"
                          f" / 腾讯 {tx_info.get('reason', '无')}"
                          f" / 新浪 {sn_info.get('reason', '无')}")
        return info

    chosen, note, fatal = _pick_source(frames)
    info["agreement"] = note
    if not chosen:
        info["reason"] = f"源之间无法互证：{note}"
        return info

    df = frames[chosen]
    info["source"] = chosen if len(frames) == 1 else f"{chosen}（{len(frames)}源互证）"
    info["reason"] = note
    info["rows"] = int(len(df))

    # --- completeness verdict -------------------------------------------------
    prev = _safe(lambda: prev_trade_day(day), None)
    prev_rows = 0
    if prev:
        p = out_dir / f"{prev}.parquet"
        if p.exists():
            prev_rows = len(pd.read_parquet(p, columns=["code"]))
        else:
            prev_rows = len(_safe(lambda: api("daily", {"trade_date": prev}, "ts_code"), pd.DataFrame()))
    info["prev_rows"] = int(prev_rows)
    info["stamped_out"] = int(df["close"].isna().sum())

    if len(df) < MIN_NAMES:
        info["ok"] = False
        info["reason"] = f"行数不足 {len(df)} < {MIN_NAMES}（数据可能只入库了一部分）"
    elif chosen == "tushare" and prev_rows and len(df) < prev_rows * MIN_COVERAGE:
        info["ok"] = False
        info["reason"] = (f"较前一交易日 {prev} 少 {1 - len(df) / prev_rows:.1%}"
                          f"（{len(df)} vs {prev_rows}），判定为不完整")
    elif df["outstanding_share"].isna().mean() > 0.05:
        info["ok"] = False
        info["reason"] = "流通股本缺失超过 5%，daily_basic 可能未落库"
    else:
        info["ok"] = True

    if info["ok"]:
        df.to_parquet(out, index=False)
    else:
        # keep whatever we got for diagnosis, but never under the canonical name
        bad = out_dir / f"{day}.incomplete.parquet"
        df.to_parquet(bad, index=False)
        info["incomplete_path"] = str(bad)
    return info


_SHARE_CACHE: dict[str, pd.Series] = {}


def _shares_hint(day: str) -> pd.Series:
    """Float shares as of the last session strictly before ``day``.

    Tencent's own market cap is only given to two decimals of 亿元, which is far
    too coarse to derive a turnover from for small caps.  Share counts move far
    more slowly than prices, so last session's figure reproduces today's turnover
    to a median relative error of 9.8e-06 -- the same order as Sina's published
    field.  Strictly ``< day``: using the same day would leak the answer.
    """
    key = day
    if key in _SHARE_CACHE:
        return _SHARE_CACHE[key]
    p = DATAPOOL / "daily_k.parquet"
    out = pd.Series(dtype="float64")
    if p.exists():
        d = pd.read_parquet(p, columns=["code", "date", "outstanding_share"])
        d["date"] = pd.to_datetime(d["date"])
        d["code"] = d["code"].astype(str).str.zfill(6)
        d = d[d["date"] < pd.Timestamp(day)].dropna(subset=["outstanding_share"])
        if not d.empty:
            last = d[d["date"] == d["date"].max()]
            out = last.set_index("code")["outstanding_share"].astype("float64")
    _SHARE_CACHE[key] = out
    return out


def _pick_source(frames: dict[str, pd.DataFrame]) -> tuple[str, str, bool]:
    """Choose a frame by agreement.  Returns ``(name, note, fatal)``.

    ``name`` is empty when nothing can be believed, which the caller treats as a
    hard stop -- emitting a signal from an unverifiable frame is worse than
    sitting the day out.
    """
    names = list(frames)
    if len(names) == 1:
        return names[0], f"仅 {names[0]} 一个源可用（无法互证）", False

    # reference = the source whose distribution the models were trained on
    ref = "tushare" if "tushare" in frames else ("sina" if "sina" in frames else names[0])
    others = [n for n in names if n != ref]
    agrees = []
    detail = []
    for n in others:
        ok, msg = _cross_check(frames[ref], frames[n])
        detail.append(f"{ref}~{n}: {msg}")
        if ok:
            agrees.append(n)
    if agrees:
        return ref, f"{ref} 与 {'/'.join(agrees)} 一致｜" + "；".join(detail), False

    # reference disagrees with everything -- check whether the others agree
    if len(others) >= 2:
        ok, msg = _cross_check(frames[others[0]], frames[others[1]])
        detail.append(f"{others[0]}~{others[1]}: {msg}")
        if ok:
            return others[0], (f"⚠ {ref} 与其余源全不一致，已弃用，改用 {others[0]}"
                               f"｜" + "；".join(detail)), False
    return "", "全部源互不一致｜" + "；".join(detail), True


def _safe(fn, default):
    try:
        return fn()
    except Exception:                                          # noqa: BLE001
        return default


def _cross_check(ts: pd.DataFrame, sn: pd.DataFrame) -> tuple[bool, str]:
    """Compare two frames name-by-name on close.  Far sharper than a row count."""
    if ts.empty or sn.empty:
        return False, "有一侧为空"
    a = ts.set_index("code")["close"]
    b = sn.set_index("code")["close"]
    j = pd.concat([a.rename("ts"), b.rename("sn")], axis=1, join="inner").dropna()
    if len(j) < MIN_NAMES:
        return False, f"可比对样本仅 {len(j):,} 只"
    rel = (j["ts"] - j["sn"]).abs() / j["sn"].abs().clip(lower=1e-9)
    agree = float((rel < CROSS_TOL).mean())
    n_bad = int((rel >= CROSS_TOL).sum())
    msg = f"收盘价一致率 {agree:.3%}（n={len(j):,}，不一致 {n_bad} 只）"
    return agree >= CROSS_MIN_AGREE, msg


def _next_gap(elapsed_min: float) -> float:
    """Retry cadence in minutes: dense while tushare is still settling, sparse after.

    tushare settles ``daily`` 15:00-16:00 and ``daily_basic`` 15:00-17:00, so the
    first half hour is where a partial frame is most likely to turn complete.
    Past that the wait is almost certainly a source-side outage, and hammering it
    every few minutes only risks getting throttled.
    """
    if elapsed_min < 30:
        return 3.0
    if elapsed_min < 90:
        return 10.0
    return 20.0


def wait_until(day: str, until: str | None, max_attempts: int = 200) -> dict:
    """Poll ``snapshot`` until it grades complete or the wall clock passes ``until``.

    ``until`` is an absolute HH:MM on the same calendar day as the run; ``None``
    means a single shot. The deadline is a *local clock* deadline rather than a
    duration so that a restarted or re-scheduled run lands on the same cut-off.
    """
    if until:
        hh, mm = (int(x) for x in until.split(":"))
        now = pd.Timestamp.now()
        dl = now.normalize() + pd.Timedelta(hours=hh, minutes=mm)
        if dl <= now:                      # already past it (e.g. a re-run at 22:45)
            dl = now + pd.Timedelta(minutes=1)
        deadline = dl.timestamp()
    else:
        deadline = time.time()

    t_start = time.time()
    info: dict = {"date": day, "ok": False, "reason": "未尝试", "rows": 0}
    for attempt in range(1, max_attempts + 1):
        # A raised error is just another kind of "not ready yet" -- it must not
        # unwind the whole run, because the caller still has to record the day as
        # pending and let the morning retry pick it up.
        try:
            info = snapshot(day)
        except Exception as exc:                      # noqa: BLE001
            info = {"date": day, "ok": False, "rows": 0,
                    "reason": f"取数异常 {type(exc).__name__}: {exc}"[:180]}
        info["attempts"] = attempt
        tag = "OK " if info["ok"] else "BAD"
        left = (deadline - time.time()) / 60
        print(f"[{tag}] {day} rows={info['rows']:,} prev={info.get('prev_rows', 0):,}"
              f" -> {info['reason']}  (第 {attempt} 次, 余 {max(0, left):.0f} 分钟)", flush=True)
        if info["ok"] or time.time() >= deadline or attempt >= max_attempts:
            break
        gap = _next_gap((time.time() - t_start) / 60)
        gap = min(gap, max(0.5, (deadline - time.time()) / 60))
        print(f"      {gap:.0f} 分钟后重试", flush=True)
        time.sleep(gap * 60)
    return info


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=pd.Timestamp.now().strftime("%Y%m%d"))
    ap.add_argument("--until", default=None,
                    help="absolute HH:MM deadline for retrying; omit for one shot")
    ap.add_argument("--wait", type=int, default=None,
                    help="legacy: minutes to retry (converted to a deadline)")
    args = ap.parse_args()

    until = args.until
    if until is None and args.wait:
        until = (pd.Timestamp.now() + pd.Timedelta(minutes=args.wait)).strftime("%H:%M")
    info = wait_until(args.date, until)
    raise SystemExit(0 if info["ok"] else 2)


if __name__ == "__main__":
    main()
