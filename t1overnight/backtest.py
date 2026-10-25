# -*- coding: utf-8 -*-
"""Selection, cost model and equity accounting for the T+1 overnight strategy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostConfig:
    commission: float = 0.00025   # per side (万2.5)
    stamp_duty: float = 0.0005    # sell side
    slippage: float = 0.0005      # per side (close-price execution)
    min_commission: float = 5.0

    @property
    def round_trip(self) -> float:
        return self.commission * 2 + self.stamp_duty + self.slippage * 2


@dataclass
class SelectConfig:
    top_n: int = 1
    min_price: float = 2.0
    max_price: float = 300.0
    min_amount: float = 50_000_000.0   # 5000 万元成交额
    exclude_st: bool = True
    exclude_limit_up_entry: bool = True
    limit_tolerance: float = 0.005
    require_finite_features: bool = True


def tradable(df: pd.DataFrame, cfg: SelectConfig) -> pd.Series:
    """Entry-side tradability mask evaluated on day D (before the close)."""
    m = df["close"].between(cfg.min_price, cfg.max_price)
    amt = df["amount"].where(df["amount"].notna(), df["close"] * df["volume"])
    m &= amt >= cfg.min_amount
    m &= df["volume"] > 0
    if cfg.exclude_st:
        m &= df["st_proxy"].fillna(0) < 0.5
    if cfg.exclude_limit_up_entry:
        m &= df["ret_1"] < (df["band"] - cfg.limit_tolerance)
    if cfg.require_finite_features:
        for c in ["ret_20", "ma_dev_20", "amt_ratio"]:
            if c in df.columns:
                m &= df[c].notna()
    return m.fillna(False)


def pick(df: pd.DataFrame, prob_col: str, cfg: SelectConfig) -> pd.DataFrame:
    """Top-N tradable names for each date, ranked by model probability."""
    mask = tradable(df, cfg)
    d = df.loc[mask]
    d = d.sort_values(["date", prob_col], ascending=[True, False])
    return d.groupby("date", sort=True).head(cfg.top_n)


def equity_from_daily(daily_ret: pd.Series, capital: float = 100_000.0,
                      cost: CostConfig | None = None) -> pd.DataFrame:
    """Compile an equity curve from a series of net daily returns."""
    eq = capital * (1.0 + daily_ret.fillna(0.0)).cumprod()
    peak = eq.cummax()
    out = pd.DataFrame({"date": eq.index, "ret": daily_ret.values, "equity": eq.values})
    out["drawdown"] = out["equity"] / peak.values - 1
    return out


def performance(daily_ret: pd.Series, label: str = "strategy") -> dict:
    r = daily_ret.dropna()
    if r.empty:
        return {"variant": label}
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    n = len(r)
    ann = eq.iloc[-1] ** (244 / n) - 1 if n > 1 else np.nan
    vol = r.std() * np.sqrt(244)
    sharpe = (r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else np.nan
    return {
        "variant": label,
        "days": n,
        "hit_rate": float((r > 0).mean()),
        "mean_daily": float(r.mean()),
        "median_daily": float(r.median()),
        "std_daily": float(r.std()),
        "total_return": float(eq.iloc[-1] - 1),
        "annual_return": float(ann),
        "annual_vol": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd),
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
        "t_stat": float(r.mean() / (r.std() / np.sqrt(n))) if r.std() > 0 else np.nan,
    }
