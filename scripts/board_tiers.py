# -*- coding: utf-8 -*-
"""How much of the v4 edge survives when part of the market is not buyable.

The STAR board (688) needs 500k AUM + 24 months of trading history; ChiNext (300)
needs 100k + 24 months; the two main boards need nothing.  So "the model picked
688090" is useless advice for an account that cannot trade it -- but simply
skipping down the ranked list is not the same strategy either, because the
replacement pick carries different beta.

This script rebuilds the strategy under each permission tier: same folds, same
models, same costs, only the tradable universe is cut.  It answers "what would I
actually have earned", not "what did the model like".

Tiers are cumulative on purpose: every extra board costs another account
qualification, so the incremental value of unlocking it should be visible.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v4_eval import SPLIT, load_panel, line_for, top1_of  # noqa: E402

ANN = 244
COSTS = (10, 13, 15, 18, 20, 25)
BOARD_NAME = {0: "沪主板", 1: "深主板", 2: "创业板", 3: "科创板", 4: "北交所"}

# name -> allowed board codes.  Cumulative: each tier adds one qualification.
TIERS: dict[str, tuple[int, ...]] = {
    "T1 沪深主板": (0, 1),
    "T2 +创业板": (0, 1, 2),
    "T3 +科创板": (0, 1, 2, 3),
    "T4 全含北交所": (0, 1, 2, 3, 4),
}


def tier_top(p: pd.DataFrame, panel: pd.DataFrame, boards: tuple[int, ...]) -> pd.DataFrame:
    q = p[p["board"].isin(boards)]
    return top1_of(q, panel, drop_st=True)


def board_mix(top: pd.DataFrame) -> str:
    if "board" not in top.columns:
        return ""
    vc = top["board"].map(BOARD_NAME).value_counts()
    tot = int(vc.sum())
    return " ".join(f"{k}{v/tot*100:.0f}%" for k, v in vc.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="outputs/walkforward_v4/pred_drop_rank.parquet")
    ap.add_argument("--snapshot", default="outputs/live_test_v4/scores_latest.csv",
                    help="latest scored universe (already tradable + ST-free)")
    ap.add_argument("--out", default="outputs/tiers")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    panel = load_panel()
    p = pd.read_parquet(ROOT / args.pred)
    p["date"] = pd.to_datetime(p["date"])
    p["code"] = p["code"].astype(str).str.zfill(6)
    print(f"[pred] {len(p):,} rows, {p.date.min().date()} .. {p.date.max().date()}")

    rows, tops, clocks = [], {}, []
    for name, boards in TIERS.items():
        top = tier_top(p, panel, boards)
        tops[name] = top
        ln = line_for(top, name)
        ln["boards"] = "+".join(BOARD_NAME[b] for b in boards)
        ln["mix"] = board_mix(top)
        rows.append(ln)
        for c in ("d_oo", "d_oc1", "fwd_co", "fwd_cc"):
            r = top[c].dropna()
            clocks.append({"tier": name, "clock": c, "days": len(r),
                           "mean_bp": r.mean() * 1e4,
                           "geo_bp": float(np.expm1(np.log1p(r.clip(-0.99, None)).mean())) * 1e4,
                           "t": r.mean() / (r.std() / np.sqrt(len(r)))})
        print(f"[tier] {name:<14} n={len(top):>5}  mix {ln['mix']}", flush=True)

    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    show = tab[["run", "days", "mean_bp", "geo_bp", "hit", "t",
                "select_bp", "select_t", "valid_bp", "valid_t"]]
    print("\n=== top1, D open -> D+1 open, gross, per permission tier ===")
    print(show.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

    print("\n=== 100k capital, annualised net, geometric ===")
    print(f"  {'tier':<14}" + "".join(f"{c}bp".rjust(9) for c in COSTS))
    geo = {}
    for name, top in tops.items():
        g = float(np.expm1(np.log1p(top["d_oo"].dropna().clip(-0.99, None)).mean()))
        geo[name] = g
        line = f"  {name:<14}"
        for c in COSTS:
            line += f"{((1 + g - c/1e4)**ANN - 1)*100:+8.1f}%"
        print(line)

    # incremental value of each qualification, paired on common days
    print("\n=== paired vs the tier below (same days) ===")
    names = list(TIERS)
    inc = []
    for lo, hi in zip(names, names[1:]):
        a, b = tops[hi], tops[lo]
        idx = a.index.intersection(b.index)
        x = a.loc[idx, "d_oo"].dropna()
        y = b.loc[idx, "d_oo"].dropna()
        idx = x.index.intersection(y.index)
        d = (x.loc[idx] - y.loc[idx])
        if len(d) == 0 or float(d.std()) == 0:
            # identical picks -- unlocking this board never changes the choice
            print(f"  {lo:<14} -> {hi:<14} 同一批票（该板块从未被选中），无增量")
            continue
        t = d.mean() / (d.std() / np.sqrt(len(d)))
        inc.append({"from": lo, "to": hi, "days": int(len(d)),
                    "to_bp": float(x.loc[idx].mean() * 1e4),
                    "from_bp": float(y.loc[idx].mean() * 1e4),
                    "lift_bp": float(d.mean() * 1e4), "t": float(t),
                    "win": float((d > 0).mean())})
        print(f"  {lo:<14} {y.loc[idx].mean()*1e4:+7.2f}bp -> {hi:<14} "
              f"{x.loc[idx].mean()*1e4:+7.2f}bp  lift {d.mean()*1e4:+6.2f}bp "
              f"t={t:+5.2f} win={(d>0).mean():.2f}")

    # ---- today's actionable pick per tier -------------------------------
    today = pd.DataFrame()
    snap_p = ROOT / args.snapshot
    if snap_p.exists():
        s = pd.read_csv(snap_p, dtype={"code": str})
        s["code"] = s["code"].astype(str).str.zfill(6)
        if "board" not in s.columns:
            s["board"] = s["板块"].map({v: k for k, v in BOARD_NAME.items()})
        s["board"] = s["board"].astype("Int16")
        out_rows = []
        for name, boards in TIERS.items():
            q = s[s["board"].isin(boards)]
            q = q.nlargest(args.top, "score")
            for i, (_, r) in enumerate(q.iterrows()):
                out_rows.append({
                    "tier": name, "rank": i + 1, "code": r["code"], "name": r["name"],
                    "board": BOARD_NAME.get(int(r["board"]), "?"),
                    "close": round(float(r["close"]), 2),
                    "prev_close": round(float(r["prev_close"]), 2),
                    "gap_cap": round(float(r["prev_close"]) * 1.015, 2),
                    "score": round(float(r["score"]), 4),
                    "pct_in_pool": round(float(r["pct"]) * 100, 1),
                })
        today = pd.DataFrame(out_rows)
        print(f"\n=== actionable list per tier ({args.snapshot}) ===")
        for name in TIERS:
            q = today[today.tier == name].head(3)
            cells = " | ".join(f"{r.code} {r['name']}({r.board}) 上限{r.gap_cap:.2f}"
                               for _, r in q.iterrows())
            print(f"  {name:<14} {cells}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "eval_tiers.csv", index=False)
    pd.DataFrame(clocks).to_csv(out / "clocks_by_tier.csv", index=False)
    pd.DataFrame(inc).to_csv(out / "incremental.csv", index=False)
    if len(today):
        today.to_csv(out / "today_by_tier.csv", index=False, encoding="utf-8-sig")
    for name, top in tops.items():
        keep = [c for c in ("code", "board", "d_oo", "fwd_cc") if c in top.columns]
        top.reset_index()[["date"] + keep].to_csv(
            out / f"top1_{name.split()[0]}.csv", index=False)
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
