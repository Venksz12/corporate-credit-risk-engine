from __future__ import annotations
import numpy as np
import pandas as pd

def _scale_bad(x, good, bad, reverse=False):
    x = pd.Series(x, dtype=float)
    if reverse:
        score = (x - good) / (bad - good)
    else:
        score = (good - x) / (good - bad)
    return np.clip(score, 0, 1) * 100

def deterioration_score(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    leverage = _scale_bad(x["debt_to_ebitda"], 2.0, 6.0, reverse=True)
    liquidity = _scale_bad(x["cash_to_debt"], 1.5, .25, reverse=False)
    coverage = _scale_bad(x["interest_coverage"], 5.0, 1.0, reverse=False)
    revenue = _scale_bad(x["revenue_growth"].fillna(.03), .08, -.12, reverse=False)
    cashflow = _scale_bad(x["operating_cash_flow_to_debt"], .25, -.02, reverse=False)
    behaviour = np.clip(
        .45 * _scale_bad(x["utilization"], .45, .95, reverse=True) +
        .35 * _scale_bad(x["days_past_due"], 0, 30, reverse=True) +
        .20 * np.clip(x["missed_payment_count"] * 12.5, 0, 100), 0, 100
    )
    x["deterioration_score"] = (
        .25*leverage + .20*liquidity + .20*coverage + .15*revenue +
        .10*cashflow + .10*behaviour
    ).clip(0, 100)
    x["risk_status"] = pd.cut(
        x["deterioration_score"], [-.01,25,50,75,100.01],
        labels=["Stable","Watch","Elevated","Critical"]
    )
    return x

def covenant_monitor(df: pd.DataFrame, max_leverage=4.0, min_coverage=2.0) -> pd.DataFrame:
    x = df.copy()
    x["leverage_breach"] = x["debt_to_ebitda"] > max_leverage
    x["coverage_breach"] = x["interest_coverage"] < min_coverage
    x["covenant_breach_count"] = x[["leverage_breach","coverage_breach"]].sum(axis=1)
    return x

def warning_signals(row: pd.Series) -> list[str]:
    signals = []
    if row.get("debt_to_ebitda", 0) > 4: signals.append("Debt/EBITDA covenant breached")
    if row.get("interest_coverage", 99) < 2: signals.append("Interest coverage below covenant")
    if row.get("utilization", 0) > .90: signals.append("Facility utilization above 90%")
    if row.get("revenue_growth", 0) < 0: signals.append("Revenue declining")
    if row.get("days_past_due", 0) >= 10: signals.append("Payment delinquency detected")
    return signals
