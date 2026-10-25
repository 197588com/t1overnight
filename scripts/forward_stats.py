# -*- coding: utf-8 -*-
"""Forward bookkeeping: what actually happened to the names we scored at 14:36.

Every recorded session contributes one row: entry at the 14:36 print of day D,
exit at the next session's print. Gross return is measured in raw traded prices,
which is exactly what a broker statement would show.

This is the only honest way to answer "how much of the 14bp survives execution":
the historical minute-level replay would need ~5.8M requests, the forward log
needs 20 seconds a day.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(folder: Path, slot: str) -> dict[pd.Timestamp, pd.DataFrame]:
    out = {}
    for f in sorted(folder.glob(f"feat_*_{slot}.parquet")):
        d = pd.read_parquet(f)
        if d.empty:
            continue
        d["code"] = d["code"].astype(str).str.zfill(6)
        out[pd.Timestamp(d["date"].iloc[0])] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "outputs/forward"))
    ap.add_argument("--entry-slot", default="1436")
    ap.add_argument("--exit-slot", default="1501")
    ap.add_argument("--top", type=int, default=1)
    ap.add_argument("--cut", type=float, default=0.20)
    args = ap.parse_args()

    folder = Path(args.dir)
    entry = load(folder, args.entry_slot)
    exit_ = load(folder, args.exit_slot) or entry
    sessions = sorted(entry)
    if len(sessions) < 2:
        print(f"only {len(sessions)} session(s) recorded under {folder}")
        print("need at least 2 to compute a forward return - keep the daily job running.")
        return

    rows = []
    for a, b in zip(sessions[:-1], sessions[1:]):
        e = entry[a].set_index("code")
        x = exit_.get(b)
        if x is None:
            continue
        x = x.set_index("code")
        price_out = x["price_now"].rename("exit_price")
        df = e.join(price_out, how="inner")
        df["gross"] = df["exit_price"] / df["price_now"] - 1.0
        df = df[np.isfinite(df["gross"])]
        if len(df) < 100:
            continue

        df["pct_bad"] = df["bad_prob"].rank(pct=True)
        df["pct_rank"] = df["rank_score"].rank(pct=True) if df["rank_score"].notna().any() else np.nan
        flag = df["pct_bad"] >= 1 - args.cut
        rec = {"date": a.date(), "n": len(df), "all_bp": df["gross"].mean() * 1e4,
               "bl_bp": df.loc[flag, "gross"].mean() * 1e4,
               "kept_bp": df.loc[~flag, "gross"].mean() * 1e4,
               "hit": (df["gross"] > 0).mean()}
        if df["pct_rank"].notna().any():
            top = df.nlargest(args.top, "rank_score")
            rec["top_bp"] = top["gross"].mean() * 1e4
            rec["ic"] = df["rank_score"].corr(df["gross"], method="spearman")
        rows.append(rec)

    if not rows:
        print("no session pair with a usable exit price yet")
        return

    r = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"=== forward record, {len(r)} sessions "
          f"({r.date.min()} -> {r.date.max()}) ===")
    print(r.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nmeans across sessions:")
    for c in ["all_bp", "bl_bp", "kept_bp", "top_bp", "hit", "ic"]:
        if c in r:
            v = r[c].mean()
            se = r[c].std() / np.sqrt(len(r)) if len(r) > 1 else np.nan
            print(f"  {c:9s} {v:+8.2f}   (se {se:.2f})" if c != "hit" and c != "ic"
                  else f"  {c:9s} {v:+8.3f}   (se {se:.3f})")

    out = folder / "forward_result.csv"
    r.to_csv(out, index=False)
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
