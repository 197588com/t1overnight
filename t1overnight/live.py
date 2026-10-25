# -*- coding: utf-8 -*-
"""Live plumbing: bulk market snapshot + point-in-time feature rebuild + scoring.

Why not the minute API: a full-market 1-min pull costs ~11 minutes, which cannot
be used at 14:36. Both the Tencent and Sina batch quote endpoints return a
full quote row for 80-120 names per request, so the whole market lands in ~10 s.
That is enough to rebuild a truncated daily bar for every stock, which is all the
model needs. (The eastmoney clist endpoint does the same job in 6 requests but
rate-limits aggressively - it is kept as a last resort only.)
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .features import add_cross_sectional, compute_base
from .features_v3 import add_shape, add_market, attach_industry
from .replay import truncate_bar, load_history

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def _prefix(code: str) -> str:
    return "sh" if code[0] == "6" else ("bj" if code[:2] in ("43", "83", "87", "92") else "sz")


def _fetch(url: str, referer: str, tries: int = 3) -> str:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": referer})
            return urllib.request.urlopen(req, timeout=12).read().decode("gbk", "ignore")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.3 * (i + 1))
    raise RuntimeError(f"quote request failed: {last}")


def _parse_tencent(txt: str) -> list[dict]:
    """Field count before the timestamp varies (empty order-book slots), so anchor there.

    After the 14-digit timestamp the layout is fixed:
        +1 change, +2 pct, +3 high, +4 low, +5 "price/volume/amount", +6 volume(lot)
    """
    out = []
    for line in txt.strip().split("\n"):
        if '="' not in line:
            continue
        f = line.split('="', 1)[1].rstrip('";').split("~")
        if len(f) < 20:
            continue
        ts = next((i for i, x in enumerate(f) if len(x) == 14 and x.isdigit()
                   and x.startswith("20")), None)
        if ts is None or ts + 6 >= len(f):
            continue
        try:
            trip = f[ts + 5].split("/")
            raw_vol = float(trip[1]) if len(trip) > 1 else float(f[ts + 6])
            amt = float(trip[2]) if len(trip) > 2 else np.nan
            px = float(f[3])
            # the volume field is lots on SH main board but shares on STAR - decide
            # per line from the implied share count, which cannot be ambiguous
            if amt > 0 and px > 0:
                est = amt / px
                raw_vol = raw_vol * 100.0 if abs(np.log(raw_vol * 100 / est)) <= abs(np.log(raw_vol / est)) else raw_vol
                raw_vol /= 100.0                     # normalise back to lots
            out.append({"code": f[2], "name": f[1], "price": px,
                        "prev_close": float(f[4]), "open": float(f[5]),
                        "high": float(f[ts + 3]), "low": float(f[ts + 4]),
                        "vol_lot": raw_vol, "amount": amt, "turn_pct": np.nan})
        except (ValueError, IndexError):
            continue
    return out


def _parse_sina(txt: str) -> list[dict]:
    out = []
    for line in txt.strip().split("\n"):
        if '="' not in line:
            continue
        head = line.split('="', 1)[0]
        body = line.split('="', 1)[1].rstrip('";')
        f = body.split(",")
        if len(f) < 32:
            continue
        code = head.split("_")[-1][2:]
        try:
            out.append({"code": code, "name": f[0],
                        "price": float(f[3]), "prev_close": float(f[2]), "open": float(f[1]),
                        "high": float(f[4]), "low": float(f[5]),
                        "vol_lot": float(f[8]) / 100.0, "amount": float(f[9]),
                        "turn_pct": np.nan})
        except (ValueError, IndexError):
            continue
    return out


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df["code"] = df["code"].astype(str).str.zfill(6)
    if "name" in df:
        df["name"] = df["name"].astype(str)
    # a suspended name prints 0 for every price; keep it out of the snapshot
    df = df[df["price"] > 0].copy()
    df["volume"] = df["vol_lot"] * 100.0          # dataset volume is in shares
    df["high"] = df[["high", "price"]].max(axis=1)
    df["low"] = np.where(df["low"] > 0, np.minimum(df["low"], df["price"]), df["price"])
    return df.reset_index(drop=True)


def bulk_snapshot(codes: list[str], batch: int = 80, sleep: float = 0.02) -> pd.DataFrame:
    """Full-market quote snapshot, Tencent first, Sina for whatever comes back empty."""
    rows, failed = [], []
    for i in range(0, len(codes), batch):
        chunk = codes[i:i + batch]
        q = ",".join(_prefix(c) + c for c in chunk)
        got = []
        for parser, url, ref in (
            (_parse_tencent, f"https://qt.gtimg.cn/q={q}", "https://gu.qq.com/"),
            (_parse_sina, f"https://hq.sinajs.cn/list={q}", "https://finance.sina.com.cn/"),
        ):
            try:
                got = parser(_fetch(url, ref))
            except RuntimeError:
                got = []
            if len(got) >= len(chunk) * 0.8:
                break
        rows.extend(got)
        missing = set(chunk) - {g["code"] for g in got}
        failed.extend(missing)
        if sleep:
            time.sleep(sleep)

    df = pd.DataFrame(rows).drop_duplicates("code")
    df = _normalise(df)
    df.attrs["missing"] = sorted(failed)
    return df


def attach_shares(snap: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Float shares come from the local daily table, so turnover stays identical
    to the definition used in training (volume / outstanding_share)."""
    so = bars.sort_values("date").groupby("code", sort=False)["outstanding_share"].last()
    snap = snap.copy()
    snap["outstanding_share"] = snap["code"].map(so)
    snap["turnover"] = snap["volume"] / snap["outstanding_share"]
    return snap


def codes_from_bars(path: str | Path) -> list[str]:
    """Every code that printed a bar on the most recent day in the file."""
    import pyarrow.parquet as pq
    schema = pq.ParquetFile(path).schema_arrow
    last = pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"]).max()
    d = pd.read_parquet(path, columns=["code", "date"],
                        filters=[("date", "==", last.strftime("%Y-%m-%d"))])
    return sorted(d["code"].astype(str).str.zfill(6).unique().tolist())


def sync_bar(hist: pd.DataFrame, snap: pd.DataFrame, day: str) -> pd.DataFrame:
    """Splice today's snapshot into the adjusted history as the bar for `day`.

    Two regimes, one rule:
      * replay  - the file already holds a completed bar for `day` -> replace it
      * live    - the file stops at day-1                          -> append it

    The adjustment factor must come from the last COMPLETED bar (day-1), because
    the adjusted series is anchored at whatever its own last date is:
        fac = adj_close(day-1) / raw_close(day-1)
    today's raw prices are then multiplied by it. Using adj(day)/raw(day) - what
    the original replay helper did - silently shifts the whole bar by one day.
    """
    h = hist.copy()
    h["date"] = pd.to_datetime(h["date"])
    today = pd.Timestamp(day)
    snap = snap.set_index("code")

    # drop a pre-existing bar for `day` so it can be rewritten from the snapshot
    h = h[h["date"] < today]
    if h.empty:
        raise RuntimeError(f"no history before {day}")

    anchors = h.groupby("code", sort=False).tail(1)
    fac = (anchors.set_index("code")["close"] / snap["prev_close"]).rename("fac")

    live_codes = [c for c in anchors["code"] if c in snap.index and np.isfinite(fac.get(c, np.nan))]
    anchors = anchors[anchors["code"].isin(live_codes)].copy()
    f = fac.reindex(anchors["code"]).to_numpy(np.float64)
    sp = snap.reindex(anchors["code"])

    new = pd.DataFrame({
        "code": anchors["code"].to_numpy(),
        "date": today,
        "open": sp["open"].to_numpy(np.float64) * f,
        "high": sp["high"].to_numpy(np.float64) * f,
        "low": sp["low"].to_numpy(np.float64) * f,
        "close": sp["price"].to_numpy(np.float64) * f,
        "volume": sp["volume"].to_numpy(np.float64),
        "amount": sp["amount"].to_numpy(np.float64),
        "outstanding_share": sp["outstanding_share"].to_numpy(np.float64),
        "turnover": sp["turnover"].to_numpy(np.float64),
        "adj_gap": f,
    })
    out = pd.concat([h, new], ignore_index=True, sort=False)
    return out.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)


def to_intra(snap: pd.DataFrame, day: str) -> pd.DataFrame:
    """Map a snapshot to the intraday contract used by the replay machinery."""
    return pd.DataFrame({
        "code": snap["code"].to_numpy(),
        "p_cut": snap["price"].to_numpy(np.float64),
        "hi_cut": snap["high"].to_numpy(np.float64),
        "lo_cut": snap["low"].to_numpy(np.float64),
        "vol_cut": snap["volume"].to_numpy(np.float64),
        "amt_cut": snap["amount"].to_numpy(np.float64),
        "close_raw": snap["prev_close"].to_numpy(np.float64),
        "dayvol": snap["volume"].to_numpy(np.float64),
    })


CARRY = ["days_listed", "roe", "net_profit_yoy", "revenue_yoy"]


def build_live_features(daily: pd.DataFrame, snap: pd.DataFrame, day: str,
                        fin: pd.DataFrame, bars: int = 90,
                        carry: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full v3 feature frame for `day`, using only information at the snapshot time."""
    hist = load_history(daily, day, bars)
    hist = sync_bar(hist, snap, day)
    today = pd.Timestamp(day)

    # only codes that actually printed today's bar can be scored
    hist = hist[hist.groupby("code", sort=False)["date"].transform("max") == today]
    adj_gap = (hist[hist["date"] == today].set_index("code")["adj_gap"])

    frames = []
    for _, g in hist.groupby("code", sort=False):
        if len(g) < 40:
            continue
        frames.append(compute_base(g))
    feat = pd.concat(frames, ignore_index=True)
    feat["code"] = feat["code"].astype(str).str.zfill(6)
    feat["date"] = pd.to_datetime(feat["date"])

    feat = add_cross_sectional(feat)
    feat = add_shape(feat)
    feat, _ = attach_industry(feat, fin)
    feat = add_market(feat)

    last = feat[feat["date"] == today].copy()
    last["name"] = last["code"].map(snap.set_index("code")["name"])
    last["price_now"] = last["code"].map(snap.set_index("code")["price"])
    last["prev_close"] = last["code"].map(snap.set_index("code")["prev_close"])
    # today's realised-until-now move, in the raw (tradable) price space
    last["ret_today"] = last["price_now"] / last["prev_close"] - 1.0
    last["adj_gap"] = last["code"].map(adj_gap)

    # days_listed counts bars from the start of the local history, and the
    # fundamentals were never merged here - both are carried from the panel
    if carry is not None and not carry.empty:
        c = carry.drop_duplicates("code").set_index("code")
        for col in CARRY:
            if col in c.columns:
                v = last["code"].map(c[col])
                last[col] = v.where(v.notna(), last[col])
        last["days_listed"] = last["days_listed"] + 1.0
    return last.sort_values("code").reset_index(drop=True)


def score(booster, df: pd.DataFrame) -> np.ndarray:
    names = booster.feature_name()
    x = df.copy()
    for c in names:
        if c not in x.columns:
            x[c] = np.nan
    x = x[names]
    for c in ("board", "ind_code"):
        if c in x.columns:
            x[c] = x[c].fillna(-1).astype("int16")
    return booster.predict(x).astype(np.float32)


def load_fin_first_industry(path: str | Path) -> pd.DataFrame:
    f = pd.read_parquet(path, columns=["code", "period", "announce_date", "industry"])
    f["code"] = f["code"].astype(str).str.zfill(6)
    f["announce_date"] = pd.to_datetime(f["announce_date"], errors="coerce")
    f = f.dropna(subset=["announce_date"]).sort_values(["code", "announce_date"])
    return f.groupby("code", as_index=False).first()


def load_bars(path: str | Path, day: str, bars: int = 90) -> pd.DataFrame:
    """Adjusted daily bars up to `day` (inclusive), last `bars` per code."""
    cols = ["code", "date", "open", "high", "low", "close", "volume",
            "amount", "outstanding_share", "turnover"]
    lo = pd.Timestamp(day) - pd.Timedelta(days=int(bars * 2.2))
    d = pd.read_parquet(path, columns=cols,
                        filters=[("date", ">", lo.strftime("%Y-%m-%d")),
                                 ("date", "<=", pd.Timestamp(day).strftime("%Y-%m-%d"))])
    d["code"] = d["code"].astype(str).str.zfill(6)
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["code", "date"])
    return d.groupby("code", sort=False).tail(bars).reset_index(drop=True)
