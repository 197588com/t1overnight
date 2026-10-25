"""Probe: how much of the T+1 edge lives in the 14:36 -> 15:00 window?

For a sample of model picks (and a market control), pull 1-minute bars and measure
  - drift  = close / price_at_1436 - 1        (the part of the label you cannot capture)
  - true_ret = next_close / price_at_1436 - 1  (what you'd actually earn entering at 14:36)
  - truncation of the three dominant features (turn, ret_1, close_pos) at 14:36
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "d226538b0b84959baae2d503d3502450fb26199c289cdf8fd249facf"
API = "https://api.tushare.pro"


def ts_code(code: str) -> str:
    c = str(code).zfill(6)
    if c[0] in "56" or c.startswith("68"):
        return f"{c}.SH"
    if c.startswith(("4", "8", "920")):
        return f"{c}.BJ"
    return f"{c}.SZ"


def query(api_name: str, params: dict, fields: str = "") -> pd.DataFrame:
    body = json.dumps({"api_name": api_name, "token": TOKEN,
                       "params": params, "fields": fields}).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(API, data=body,
                                         headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=40).read())
            if r.get("code") != 0:
                raise RuntimeError(r.get("msg"))
            d = r.get("data") or {}
            return pd.DataFrame(d.get("items") or [], columns=d.get("fields") or [])
        except Exception as e:                      # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    return pd.DataFrame()


def day_metrics(code: str, date: str) -> dict | None:
    df = query("stk_mins",
               {"ts_code": ts_code(code), "freq": "1min",
                "start_date": f"{date} 09:00:00", "end_date": f"{date} 15:10:00"},
               "ts_code,trade_time,open,high,low,close,vol,amount")
    if df.empty or len(df) < 100:
        return None
    df["t"] = df["trade_time"].str[-8:]
    df = df.sort_values("t").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "vol", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def px(t):
        s = df.loc[df["t"] == t, "close"]
        return float(s.iloc[0]) if len(s) else np.nan

    pre = df[df["t"] <= "14:36:00"]
    if pre.empty:
        return None
    p36, p40 = px("14:36:00"), px("14:40:00")
    dclose = float(df["close"].iloc[-1])
    # exclude the 15:00 auction print when accumulating continuous-session volume
    cum_vol = float(pre["vol"].sum())
    day_vol = float(df["vol"].sum())
    hi36, lo36 = float(pre["high"].max()), float(pre["low"].min())
    hi, lo = float(df["high"].max()), float(df["low"].min())
    rng36, rng = hi36 - lo36, hi - lo
    return {
        "p1436": p36, "p1440": p40, "close_m": dclose,
        "cumvol": cum_vol, "cumamt": float(pre["amount"].sum()), "dayvol": day_vol,
        "cumvol_ratio": cum_vol / day_vol if day_vol else np.nan,
        "cumamt_ratio": float(pre["amount"].sum()) / float(df["amount"].sum()),
        "hi36": hi36, "lo36": lo36, "day_high_m": hi, "day_low_m": lo,
        "pos36": (p36 - lo36) / rng36 if rng36 > 0 else np.nan,
        "pos_day": (dclose - lo) / rng if rng > 0 else np.nan,
        "n_bars": len(df),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="outputs/walkforward_v2/predictions.parquet")
    ap.add_argument("--pred-strict", default="outputs/walkforward_v2_lag1/predictions.parquet")
    ap.add_argument("--dataset", default="outputs/t1_dataset.parquet")
    ap.add_argument("--n", type=int, default=300, help="picks sampled per book")
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--out", default="outputs/probe1436/metrics.csv")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT))
    from t1overnight.backtest import SelectConfig, tradable

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(ROOT / args.dataset,
                           columns=["code", "date", "close", "open", "high", "low",
                                    "amt_ratio", "ret_20", "ma_dev_20", "turn",
                                    "exec_ok", "board", "amount"])
    base["date"] = pd.to_datetime(base["date"])
    base["code"] = base["code"].astype(str).str.zfill(6)
    base = base.sort_values(["code", "date"])
    base["next_close"] = base.groupby("code")["close"].shift(-1)
    base["prev_close"] = base.groupby("code")["close"].shift(1)
    base["next_open"] = base.groupby("code")["open"].shift(-1)

    def picks(path: str, n: int) -> pd.DataFrame:
        p = pd.read_parquet(ROOT / path)
        p["date"] = pd.to_datetime(p["date"])
        p["code"] = p["code"].astype(str).str.zfill(6)
        p = p.merge(base[["code", "date", "amt_ratio", "ret_20", "ma_dev_20"]],
                    on=["code", "date"], how="left")
        cfg = SelectConfig(top_n=1)
        p = p[tradable(p, cfg)].copy()
        key = "score" if "score" in p.columns else "prob"
        top = (p.sort_values(["date", key], ascending=[True, False])
                 .groupby("date").head(1).sort_values("date").reset_index(drop=True))
        step = max(1, len(top) // n)
        return top.iloc[::step].head(n)

    groups = {"pick_close": picks(args.pred, args.n),
              "pick_strict": picks(args.pred_strict, args.n)}

    rng = np.random.default_rng(7)
    mkt = base[base["exec_ok"] == 1].dropna(subset=["amt_ratio"])
    mkt = mkt[mkt["date"] >= "2022-01-01"]
    groups["market"] = mkt.iloc[rng.choice(len(mkt), args.n_random, replace=False)]

    rows = []
    for name, g in groups.items():
        print(f"[{name}] {len(g)} stock-days", flush=True)
        for i, r in enumerate(g.itertuples(), 1):
            d = r.date.strftime("%Y-%m-%d")
            try:
                m = day_metrics(r.code, d)
            except Exception as e:                  # noqa: BLE001
                print(f"  ! {r.code} {d} {e}", flush=True)
                m = None
            if m is None:
                continue
            m.update({"group": name, "code": r.code, "date": d,
                      "ret_1": getattr(r, "ret_1", np.nan),
                      "turn_d": getattr(r, "turn", np.nan)})
            rows.append(m)
            if i % 50 == 0:
                print(f"  {i}/{len(g)}", flush=True)
            time.sleep(0.06)

    df = pd.DataFrame(rows)
    df = df.merge(base[["code", "date", "close", "prev_close", "next_close",
                        "next_open", "turn", "amount"]]
                  .assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d")),
                  on=["code", "date"], how="left")
    # parquet closes are forward-adjusted, minute bars are raw: only same-day ratios are safe
    df["drift36"] = df["close_m"] / df["p1436"] - 1
    df["drift40"] = df["close_m"] / df["p1440"] - 1
    df["cc_ret"] = df["next_close"] / df["close"] - 1
    df["ret_day"] = df["close"] / df["prev_close"] - 1
    df["ret36"] = (1 + df["ret_day"]) / (1 + df["drift36"]) - 1
    df["true_ret36"] = (1 + df["cc_ret"]) * (1 + df["drift36"]) - 1
    df["true_ret40"] = (1 + df["cc_ret"]) * (1 + df["drift40"]) - 1
    df["turn36_ratio"] = df["turn"] * df["cumvol_ratio"]
    df.to_csv(out, index=False)

    pd.set_option("display.width", 220)
    agg = df.groupby("group").agg(
        n=("drift36", "size"),
        drift_mean=("drift36", "mean"), drift_med=("drift36", "median"),
        drift_win=("drift36", lambda s: (s > 0).mean()),
        drift_std=("drift36", "std"),
        vol_share=("cumvol_ratio", "mean"),
        ret36=("ret36", "mean"), ret_day=("ret_day", "mean"),
        cc=("cc_ret", "mean"), true36=("true_ret36", "mean"),
        true40=("true_ret40", "mean"),
        hit36=("true_ret36", lambda s: (s > 0).mean()),
        pos36=("pos36", "mean"), pos_day=("pos_day", "mean"),
    )
    print("\n=== 14:36 vs 15:00 ===")
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n(turn 在 14:36 的累计占全天比例 / close_pos 在 14:36 与收盘的差异)")
    sub = df.groupby("group").agg(
        turn_ratio=("cumvol_ratio", "mean"),
        turn_ratio_med=("cumvol_ratio", "median"),
        pos36=("pos36", "mean"), pos_day=("pos_day", "mean"),
        abs_pos_gap=("pos36", lambda s: np.nan),
    )
    print(sub.drop(columns=["abs_pos_gap"]).to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nrows saved -> {out}")


if __name__ == "__main__":
    main()
