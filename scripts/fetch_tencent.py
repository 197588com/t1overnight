# -*- coding: utf-8 -*-
"""Third source: Tencent's batch quote endpoint.  Fast probe + price arbiter.

Where this fits
---------------
Not a drop-in replacement -- its turnover is too coarse (see below) -- but the
fastest and most explicit of the three sources, which makes it serve two jobs
the other two cannot:

1. **Probe.**  5,577 names in ~3.0s with 14 batched requests, versus 8.2s for
   Sina and a tushare call that is documented to settle as late as 17:00.  It is
   the cheapest way to answer "is today's close final yet, and is it today?".

2. **Arbiter.**  When tushare and Sina disagree on a price, two sources cannot
   decide which one is wrong.  A third independent operator can: majority wins.

The date stamp is the real prize
--------------------------------
Every quote carries ``[30]`` = ``20261023161458`` -- a full timestamp with the
date in it.  Sina has no date field at all (it needs a previous-close
fingerprint to verify), and tushare infers the date from the request.  Tencent
states it.  So a wrong-session frame is caught by reading one field.

Accuracy, measured against tushare on 2026-10-23 (5,572 names in common)
-----------------------------------------------------------------------
    open/high/low/close   median rel err 2.0e-08   0 names >1e-3
    amount                median rel err 0.0       0 names >1e-3
    outstanding_share     63 names >1e-3 (2-decimal 亿元 market cap)
    turnover              2,506 names >1e-3        <- coarsest of the three

``[38]`` carries turnover rounded to two decimals (``0.23``), so its relative
error is ~1e-3 rather than Sina's 1e-5.  Recomputing it as
``amount / float_market_cap`` is *worse* (median 3.2e-02) because ``[44]`` is
only given in 亿元 to two decimals.  Since ``turn``/``cs_turn`` are the two
highest-gain features in the model, a coarse turnover is disqualifying for
signal generation -- hence Tencent is used for OHLCV and verification only.

One unit trap
-------------
``[6]`` is 成交量 in 手 for every board *except* STAR (688***), where it is
already in shares.  Measured: 617 STAR names came back exactly 100x too large.
Rather than hard-coding the prefix rule (which a future change could break), the
parser cross-checks each name against ``amount / (volume * 100 * close)``: a
ratio near 0.01 means the figure was already in shares.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATAPOOL = Path(os.environ.get("T1_DATAPOOL", os.path.expanduser("~/datapool/parquet")))

URL = "http://qt.gtimg.cn/q="
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
}
CHUNK = 400               # measured: 400 codes -> 400 rows in 0.22s, no cap hit
MIN_NAMES = 5000


def _prefix(code: str) -> str:
    if code[0] == "6":
        return "sh"
    if code.startswith(("92", "43", "83", "87", "4", "8")):
        return "bj"
    return "sz"


def _num(x: str) -> float:
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return np.nan


def fetch_all(codes: list[str], chunk: int = CHUNK, retries: int = 3) -> pd.DataFrame:
    """Batch-pull the whole list.  ``codes`` must be supplied -- see module note."""
    rows: list[dict] = []
    for i in range(0, len(codes), chunk):
        part = codes[i:i + chunk]
        q = ",".join(_prefix(c) + c for c in part)
        for attempt in range(retries):
            try:
                r = requests.get(URL + q, headers=HEADERS, timeout=20)
                r.encoding = "gbk"
                for line in r.text.split(";"):
                    line = line.strip()
                    if not line.startswith("v_"):
                        continue
                    f = line.split('="', 1)[1].rstrip('"').split("~")
                    if len(f) < 49:
                        continue
                    amt = f[35].split("/")
                    rows.append({
                        "code": f[2], "name": f[1], "stamp": f[30],
                        "close": _num(f[3]), "pre_close": _num(f[4]), "open": _num(f[5]),
                        "high": _num(f[33]), "low": _num(f[34]),
                        "volume_raw": _num(f[6]),
                        "amount": _num(amt[2]) if len(amt) > 2 else np.nan,
                        "turnover_pct": _num(f[38]),
                        "float_mv_yi": _num(f[44]), "total_mv_yi": _num(f[45]),
                    })
                break
            except (requests.RequestException, ValueError, IndexError):
                if attempt == retries - 1:
                    raise
                time.sleep(2)
    if not rows:
        return pd.DataFrame()
    q = pd.DataFrame(rows)
    q["code"] = q["code"].astype(str).str.zfill(6)
    return q.drop_duplicates("code", keep="last").reset_index(drop=True)


def verify_date(q: pd.DataFrame, day: str) -> tuple[bool, str]:
    """Tencent stamps every quote with its own date -- just read it."""
    if q.empty or "stamp" not in q:
        return False, "无时间戳"
    stamp = q["stamp"].astype(str)
    got = stamp.str[:8]
    hit = float((got == day).mean())
    if hit < 0.98:
        common = got.value_counts().head(1)
        seen = f"{common.index[0]}（{common.iloc[0]} 只）" if len(common) else "未知"
        return False, f"时间戳日期不符：一致率 {hit:.2%}，实际多为 {seen}"
    lo = stamp.str[8:14]
    return True, f"时间戳日期一致率 {hit:.2%}，行情时钟 {lo.min()}–{lo.max()}"


def to_schema(q: pd.DataFrame, day: str,
              shares_hint: pd.Series | None = None) -> pd.DataFrame:
    """Map onto the project schema.

    ``shares_hint`` (code -> outstanding_share) is preferred over Tencent's
    2-decimal 亿元 float cap when available, because share counts move far more
    slowly than prices and the local pool's figure is much more precise.
    """
    px = q["close"].astype("float64")
    vol = q["volume_raw"].astype("float64")
    amt = q["amount"].astype("float64")

    # per-name unit self-check: amount / (vol*100*close) ~ 1 for 手, ~0.01 for shares
    ratio = amt / (vol * 100.0 * px.where(px > 0)).replace(0, np.nan)
    already_shares = (ratio < 0.1).fillna(False).to_numpy(dtype=bool)
    volume = np.where(already_shares, vol, vol * 100.0)

    shares = (q["float_mv_yi"].astype("float64") * 1e8 / px.where(px > 0)).astype("float64")
    if shares_hint is not None and len(shares_hint):
        hinted = q["code"].map(shares_hint).astype("float64")
        shares = hinted.where(hinted.notna() & (hinted > 0), shares)

    turnover = np.asarray(volume, dtype="float64") / shares.where(shares > 0)
    # fall back to the coarse published field when the recomputation is unusable
    turnover = pd.Series(turnover, index=q.index).where(
        lambda s: s.notna() & (s > 0), q["turnover_pct"].astype("float64") * 0.01)

    out = pd.DataFrame({
        "code": q["code"],
        "date": pd.to_datetime(day, format="%Y%m%d"),
        "open": q["open"].astype("float32"),
        "high": q["high"].astype("float32"),
        "low": q["low"].astype("float32"),
        "close": px.astype("float32"),
        "volume": volume,
        "amount": amt,
        "outstanding_share": shares,
        "turnover": turnover,
    })
    out.loc[out["volume"].fillna(0) <= 0, "turnover"] = np.nan
    return out


def default_codes() -> list[str]:
    """Local fallback universe when neither tushare nor Sina could supply one."""
    p = DATAPOOL / "daily_k.parquet"
    if not p.exists():
        return []
    d = pd.read_parquet(p, columns=["code"])
    return sorted(set(d["code"].astype(str).str.zfill(6)))


def snapshot(day: str, codes: list[str] | None = None,
             shares_hint: pd.Series | None = None) -> tuple[dict, pd.DataFrame]:
    """Returns ``(verdict, frame)``; caller must read ``verdict['ok']``."""
    info: dict = {"source": "tencent", "ok": False, "reason": "", "rows": 0}
    codes = list(codes) if codes else default_codes()
    if not codes:
        info["reason"] = "无可请求的代码清单"
        return info, pd.DataFrame()

    t0 = time.time()
    try:
        q = fetch_all(codes)
    except Exception as exc:                                  # noqa: BLE001
        info["reason"] = f"腾讯取数异常 {type(exc).__name__}: {exc}"[:160]
        return info, pd.DataFrame()
    if q.empty:
        info["reason"] = "腾讯返回空"
        return info, pd.DataFrame()

    ok_date, why = verify_date(q, day)
    info["date_check"] = why
    info["requested"] = len(codes)
    info["rows"] = int(len(q))
    info["elapsed_s"] = round(time.time() - t0, 1)
    if not ok_date:
        info["reason"] = why
        return info, pd.DataFrame()

    df = to_schema(q, day, shares_hint)
    if len(df) < MIN_NAMES:
        info["reason"] = f"行数不足 {len(df)} < {MIN_NAMES}"
        return info, df
    if df["outstanding_share"].isna().mean() > 0.05:
        info["reason"] = "流通股本缺失超过 5%"
        return info, df
    info["ok"] = True
    info["reason"] = "完整（腾讯）"
    return info, df


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=pd.Timestamp.now().strftime("%Y%m%d"))
    a = ap.parse_args()
    info, _ = snapshot(a.date)
    print(info)
    raise SystemExit(0 if info["ok"] else 2)


if __name__ == "__main__":
    main()
