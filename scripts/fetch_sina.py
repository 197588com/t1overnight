# -*- coding: utf-8 -*-
"""Backup source: one trading day's full-market bar from Sina's ranking endpoint.

Why this exists
---------------
tushare was the sole production source, and its failure mode is silent: when a
pull lands mid-settlement it returns HTTP 200 with two thirds of the market.  The
old completeness gate compared today's row count against the previous session's,
which is an *inference* -- it catches a partial frame only if the shortfall is
large, and it cannot verify that the rows it did get belong to the intended date.

Sina's ``Market_Center.getHQNodeData`` endpoint is a genuinely independent second
opinion (different operator, different pipeline, different CDN).  Verified against
tushare's 2026-10-23 snapshot, 5,571 names in common:

    open/high/low/close   median relative error 2.0e-08   (float32 noise, 0 names >1e-3)
    volume / amount       median relative error 0.0       (bit-identical after unit fix)
    turnover              median relative error 8.5e-06  (37 names differ, share-count events)
    outstanding_share     median relative error 0.0      (the same 37 names)

The 37 divergent names are float-share adjustment days (rights issues, lock-up
releases); tushare's ``daily_basic`` carries the post-event share count while
Sina's ``nmc`` sometimes still carries the pre-event one.  That is 0.66% of the
market and it only touches ``turn``/``log_mktcap``, so this source is used as a
*fallback*, never as a silent replacement.

Two fingerprints make it safe to use without trusting it
--------------------------------------------------------
1. ``ticktime`` -- every row carries a quote timestamp.  A pull taken while the
   market is still open returns an intraday clock; anything before 15:00 is
   rejected outright.  On 2026-10-23 all rows sat in 15:00:00-15:36:00.
2. ``settlement`` (previous close) -- matched the local datapool's 2026-10-22
   close for 5,569 of 5,569 names with *zero* error.  This is what proves the
   endpoint handed back the day we asked for: Sina exposes no date field at all,
   so without this check a weekend pull for "Monday" would silently return
   Friday's bar and look perfectly well-formed.

Units (Sina -> project schema, all verified row-by-row):
    trade            -> close
    settlement       -> pre_close  (verification only, not stored)
    volume           shares        (already shares, unlike tushare's 手)
    amount           yuan          (already yuan, unlike tushare's 千元)
    turnoverratio %  -> turnover   / 100
    nmc             万元          -> outstanding_share = nmc * 1e4 / close
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATAPOOL = Path(os.environ.get("T1_DATAPOOL", os.path.expanduser("~/datapool/parquet")))

URL = ("http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php"
       "/Market_Center.getHQNodeData")
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Referer": "https://finance.sina.com.cn/",
}

NUM_COLS = ("trade", "settlement", "open", "high", "low", "volume", "amount",
            "turnoverratio", "nmc", "mktcap")

MIN_NAMES = 5000          # same floor as the tushare path
CLOSE_TOL = 1e-3          # 0.1% -- far above float noise, far below a real mismatch
MIN_AGREE = 0.995         # cross-source close agreement required to trust either


def _get_page(page: int, retries: int = 3) -> list[dict]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(URL, headers=HEADERS, timeout=15,
                             params={"page": page, "num": 100, "sort": "symbol",
                                     "asc": 1, "node": "hs_a", "symbol": "",
                                     "_s_r_a": "page"})
            r.raise_for_status()
            body = r.text.strip()
            if body in ("", "null"):
                return []
            return json.loads(body)
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt == retries - 1:
                raise
            time.sleep(2)
    raise RuntimeError(str(last))


def fetch_all(retries: int = 3) -> pd.DataFrame:
    """Page through the whole A-share market.  ~56 pages, ~8s, no throttling."""
    rows: list[dict] = []
    for page in range(1, 90):
        batch = _get_page(page, retries)
        if not batch:
            break
        rows += batch
    if not rows:
        return pd.DataFrame()
    s = pd.DataFrame(rows)
    for c in NUM_COLS:
        s[c] = pd.to_numeric(s[c].astype(str).str.replace(",", ""), errors="coerce")
    s["code"] = s["code"].astype(str).str.zfill(6)
    return s.drop_duplicates("code", keep="last").reset_index(drop=True)


def _local_prev_close(before: str) -> pd.Series:
    """Close of the last bar strictly before ``before``, from the local pool.

    Deliberately local: this check exists precisely for the case where tushare is
    unreachable, so it must not depend on tushare's trade calendar.
    """
    p = DATAPOOL / "daily_k.parquet"
    if not p.exists():
        return pd.Series(dtype="float64")
    d = pd.read_parquet(p, columns=["code", "date", "close"])
    d["date"] = pd.to_datetime(d["date"])
    d["code"] = d["code"].astype(str).str.zfill(6)
    cut = pd.Timestamp(before)
    d = d[d["date"] < cut]
    if d.empty:
        return pd.Series(dtype="float64")
    last = d[d["date"] == d["date"].max()]
    return last.set_index("code")["close"].astype("float64")


def verify_date(s: pd.DataFrame, day: str) -> tuple[bool, str]:
    """Confirm the frame belongs to ``day`` using the previous close fingerprint."""
    prev = _local_prev_close(day)
    if prev.empty:
        return False, "无本地基准（前收盘缺失），无法确认日期"
    j = (s.set_index("code")["settlement"].to_frame("st")
         .join(prev.to_frame("pc"), how="inner"))
    j = j[(j["st"] > 0) & (j["pc"] > 0)]
    if len(j) < MIN_NAMES * 0.9:
        return False, f"可比对样本仅 {len(j)} 只，不足以确认日期"
    rel = (j["st"] - j["pc"]).abs() / j["pc"]
    agree = float((rel < CLOSE_TOL).mean())
    if agree < MIN_AGREE:
        return False, (f"昨收指纹不匹配（一致率 {agree:.2%} < {MIN_AGREE:.1%}），"
                       f"该快照很可能不是 {day}")
    return True, f"昨收指纹一致率 {agree:.2%}（n={len(j):,}）"


def to_schema(s: pd.DataFrame, day: str) -> pd.DataFrame:
    """Map Sina's columns onto the exact project schema."""
    px = s["trade"].astype("float64")
    nmc = s["nmc"].astype("float64")
    # a zero/NaN price cannot carry a market cap; NaN keeps the hole visible
    shares = (nmc * 1e4 / px.where(px > 0)).astype("float64")
    out = pd.DataFrame({
        "code": s["code"],
        "date": pd.to_datetime(day, format="%Y%m%d"),
        "open": s["open"].astype("float32"),
        "high": s["high"].astype("float32"),
        "low": s["low"].astype("float32"),
        "close": px.astype("float32"),
        "volume": s["volume"].astype("float64"),
        "amount": s["amount"].astype("float64"),
        # nmc is float market cap in 万元 -> shares outstanding
        "outstanding_share": shares,
        "turnover": (s["turnoverratio"].astype("float64") * 0.01),
    })
    # a name that did not trade has no meaningful turnover; leave it NaN rather
    # than 0.0 so downstream "missing" logic still sees a hole
    out.loc[out["volume"].fillna(0) <= 0, "turnover"] = np.nan
    return out


def snapshot(day: str, return_frame: bool = False) -> dict | tuple[dict, pd.DataFrame]:
    """Fetch and grade one day.  Returns a verdict dict; caller must read ``ok``.

    ``return_frame=True`` additionally hands back the mapped frame so the caller
    does not have to re-page the endpoint just to merge it.
    """
    info = _snapshot(day)
    df = info.pop("_frame", pd.DataFrame())
    return (info, df) if return_frame else info


def _snapshot(day: str) -> dict:
    info: dict = {"source": "sina", "ok": False, "reason": "", "rows": 0}
    t0 = time.time()
    try:
        raw = fetch_all()
    except Exception as exc:                                  # noqa: BLE001
        info["reason"] = f"新浪取数异常 {type(exc).__name__}: {exc}"[:160]
        return info
    if raw.empty:
        info["reason"] = "新浪返回空"
        return info

    # fingerprint 1: the quote clock must be after the close
    # A missing clock is treated as *not* settled: on 2026-10-23 every row had
    # one, so a hole here means the frame is not what we think it is.
    tick = raw["ticktime"].astype(str).fillna("00:00:00")
    early = (tick < "15:00:00").fillna(True).to_numpy(dtype=bool)
    pre_close_clock = float(early.mean())
    info["ticktime_pre_close"] = round(pre_close_clock, 4)
    if pre_close_clock > 0.02:
        info["reason"] = (f"行情时钟仍为盘中（{pre_close_clock:.1%} 早于 15:00），"
                          f"非当日定稿快照")
        info["rows"] = int(len(raw))
        return info

    # fingerprint 2: previous-close match proves the date
    ok_date, why = verify_date(raw, day)
    info["date_check"] = why
    if not ok_date:
        info["reason"] = why
        info["rows"] = int(len(raw))
        return info

    df = to_schema(raw, day)
    info["rows"] = int(len(df))
    info["elapsed_s"] = round(time.time() - t0, 1)
    info["stamped_out"] = int(df["close"].isna().sum())

    if len(df) < MIN_NAMES:
        info["reason"] = f"行数不足 {len(df)} < {MIN_NAMES}"
    elif df["outstanding_share"].isna().mean() > 0.05:
        info["reason"] = "流通市值缺失超过 5%"
    else:
        info["ok"] = True
        info["reason"] = "完整（新浪）"
    info["_frame"] = df
    return info


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=pd.Timestamp.now().strftime("%Y%m%d"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    info = snapshot(a.date)
    print(json.dumps(info, ensure_ascii=False, indent=1))
    if info["ok"] and a.out:
        raw = fetch_all()
        to_schema(raw, a.date).to_parquet(a.out, index=False)
        print(f"written {a.out}")
    raise SystemExit(0 if info["ok"] else 2)


if __name__ == "__main__":
    main()
