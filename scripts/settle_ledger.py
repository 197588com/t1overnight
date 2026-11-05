# -*- coding: utf-8 -*-
"""Grade the forward ledger against what the tape actually did.

Why this exists
---------------
``daily_signal.py`` appends one row per (signal bar, tier) with the entry and exit
columns deliberately left empty -- they cannot be known at signal time.  Nothing
filled them afterwards, so the ledger grew a row a day and stayed permanently
ungradeable.  The gate before real money is "60 trading days, gross >= +25bp";
without this step that gate could never close.

The clock being traded is S+1 open -> S+2 open, with S = signal bar:

    signal on bar S  ->  row.trade_date = S+1  (buy at S+1 open)
    sell at S+2 open

so both prices come from the same rebuilt dataset the rest of the pipeline uses,
and they arrive two sessions after the row is written.

Rules
-----
* A row is only graded when the price is actually in the dataset.  Missing bar
  means "not yet", never "assume flat".
* If the entry session opened above ``gap_cap`` the trade was never taken.  That
  is recorded as ``no_fill_gap`` rather than deleted -- a skipped trade is part
  of the record, and quietly dropping it would flatter the hit rate.
* ``traded`` markers starting with ``void`` are left alone: they belong to rows
  already declared invalid (wrong model / wrong basis) and must not be revived.
  ``no_pick`` rows (model returned nothing that day) are skipped for the same
  reason -- there is no trade to grade.
* A trade_date that is not a session was written with weekday arithmetic and is
  rolled forward to the next real session, with a log line.  Left alone it would
  never grade, and the 60-day gate would silently lose that trade.
* Nothing is overwritten once filled, so a re-run is idempotent and a manually
  corrected price survives.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# round-trip cost used everywhere else in this project (commission + stamp + slippage)
COST_ROUND_TRIP = 0.0013


def _bars(dataset: Path) -> tuple[dict[pd.Timestamp, pd.Series], list[pd.Timestamp]]:
    """Return ({(date, code): open}, sorted sessions)."""
    d = pd.read_parquet(dataset, columns=["date", "code", "open"])
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["code"] = d["code"].astype(str).str.zfill(6)
    d = d.dropna(subset=["open"])
    lookup = {(r.date, r.code): float(r.open) for r in d.itertuples(index=False)}
    sessions = sorted(d["date"].unique())
    return lookup, [pd.Timestamp(s) for s in sessions]


def settle(ledger: Path, dataset: Path, cost: float = COST_ROUND_TRIP,
           dry_run: bool = False) -> pd.DataFrame:
    led = pd.read_csv(ledger, dtype={"code": str})
    for c in ("entry_price", "exit_price", "gross_ret", "net_ret"):
        if c not in led:
            led[c] = np.nan
        led[c] = pd.to_numeric(led[c], errors="coerce")
    for c in ("exit_date", "traded"):
        if c not in led:
            led[c] = ""
        led[c] = led[c].fillna("").astype(str)
    led["code"] = led["code"].astype(str).str.zfill(6)

    lookup, sessions = _bars(dataset)
    session_set = set(sessions)

    filled_entry = filled_exit = skipped_gap = 0

    for i, row in led.iterrows():
        if row["traded"].lower().startswith("void"):
            continue                                    # invalidated row, hands off
        if row["traded"] == "no_fill_gap":
            continue                                    # already judged unfilled

        try:
            tdate = pd.Timestamp(str(row["trade_date"])).normalize()
        except (ValueError, TypeError):
            continue
        if row["code"] in ("", "nan"):
            continue                        # no_pick / broken row, nothing to grade
        if tdate not in session_set:
            # A trade_date that is not a session means the row was written with
            # weekday arithmetic (the old code did exactly that, so a signal on
            # the last session before a holiday named the holiday).  The order
            # would really have been filled at the next actual open, so roll it
            # forward -- otherwise the row stays ungradeable forever and the
            # 60-day gate silently loses the trade.
            later_ss = [s for s in sessions if s > tdate]
            if not later_ss:
                continue
            print(f"[settle] {row['code']} trade_date {tdate.date()} 不是交易日"
                  f" → 顺延到 {later_ss[0].date()}", flush=True)
            tdate = later_ss[0]
            led.at[i, "trade_date"] = tdate.date().isoformat()

        # --- entry: the open of the trade date -------------------------------
        if pd.isna(row["entry_price"]):
            op = lookup.get((tdate, row["code"]))
            if op is None:
                continue                                # bar not in dataset yet
            cap = row.get("gap_cap")
            if pd.notna(cap) and op > float(cap):
                led.at[i, "traded"] = "no_fill_gap"
                skipped_gap += 1
                continue
            led.at[i, "entry_price"] = op
            if str(row["traded"]).strip() == "":
                led.at[i, "traded"] = "shadow_filled"
            filled_entry += 1

        # --- exit: the open of the session after the trade date --------------
        if pd.isna(led.at[i, "exit_price"]):
            later = [s for s in sessions if s > tdate]
            if not later:
                continue                                # exit session hasn't happened
            xdate = later[0]
            xp = lookup.get((xdate, row["code"]))
            if xp is None:
                continue
            entry = float(led.at[i, "entry_price"])
            if entry <= 0:
                continue
            gross = xp / entry - 1.0
            led.at[i, "exit_date"] = xdate.date().isoformat()
            led.at[i, "exit_price"] = xp
            led.at[i, "gross_ret"] = round(gross, 6)
            led.at[i, "net_ret"] = round(gross - cost, 6)
            filled_exit += 1

    if not dry_run:
        led.to_csv(ledger, index=False, encoding="utf-8-sig")

    done = led[(led["gross_ret"].notna())
               & (~led["traded"].str.lower().str.startswith("void"))]
    print(f"[settle] entry +{filled_entry} / exit +{filled_exit} / "
          f"gap-skip {skipped_gap} | graded rows total {len(done)}", flush=True)
    if len(done):
        avg = done["gross_ret"].mean() * 1e4
        hit = (done["gross_ret"] > 0).mean()
        print(f"[settle] 已结算 {len(done)} 笔：毛均 {avg:+.2f}bp，胜率 {hit:.1%}，"
              f"净均 {done['net_ret'].mean()*1e4:+.2f}bp", flush=True)
        # the gate before real money
        need = 60
        print(f"[settle] 门槛进度 {len(done)}/{need} 交易日"
              f"（{'已满足样本量' if len(done) >= need else '继续积累'}）", flush=True)

    pending = led[(led["entry_price"].notna()) & (led["exit_price"].isna())
                  & (~led["traded"].str.lower().str.startswith("void"))]
    if len(pending):
        print(f"[settle] 待结算 {len(pending)} 笔（等下一交易日开盘）", flush=True)
    return led


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=str(ROOT / "outputs/live/forward_ledger.csv"))
    ap.add_argument("--dataset", default=str(ROOT / "outputs/t1_dataset.parquet"))
    ap.add_argument("--cost", type=float, default=COST_ROUND_TRIP)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ledger, dataset = Path(args.ledger), Path(args.dataset)
    if not ledger.exists():
        print(f"[settle] 无账本 {ledger}，跳过")
        return
    if not dataset.exists():
        print(f"[settle] 无数据集 {dataset}，跳过")
        return
    settle(ledger, dataset, cost=args.cost, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
