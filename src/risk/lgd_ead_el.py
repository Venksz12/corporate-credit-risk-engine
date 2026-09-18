from __future__ import annotations
import numpy as np
import pandas as pd

def calculate_ead(facilities: pd.DataFrame) -> pd.Series:
    undrawn = (facilities["sanctioned_limit"] - facilities["drawn_amount"]).clip(lower=0)
    return facilities["drawn_amount"] + facilities["ccf"] * undrawn

def calculate_lgd(facilities: pd.DataFrame, ead: pd.Series, discount_rate=.08) -> pd.Series:
    effective_collateral = facilities["collateral_value"] * (1-facilities["haircut"])
    recovery_cost = .05 * effective_collateral + .02 * ead
    net_recovery = (effective_collateral - recovery_cost).clip(lower=0)
    maturity = pd.to_datetime(facilities["maturity_date"])
    months = ((maturity - pd.Timestamp("2025-12-31")).dt.days / 30.4375).clip(lower=3, upper=60)
    pv_recovery = net_recovery / ((1+discount_rate) ** (months/12))
    recovery_rate = (pv_recovery / ead.replace(0,np.nan)).clip(0, .95).fillna(0)
    return (1-recovery_rate).clip(.05, 1.0)

def expected_loss(pd_values, lgd, ead):
    return np.asarray(pd_values) * np.asarray(lgd) * np.asarray(ead)
