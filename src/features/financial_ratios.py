from __future__ import annotations
import numpy as np
import pandas as pd

def calculate_financial_ratios(financials: pd.DataFrame) -> pd.DataFrame:
    df = financials.copy().sort_values(["borrower_id", "fiscal_year"])
    safe = lambda s: s.replace(0, np.nan)
    df["revenue_growth"] = df.groupby("borrower_id")["revenue"].pct_change()
    df["ebitda_margin"] = df["ebitda"] / safe(df["revenue"])
    df["debt_to_ebitda"] = df["total_debt"] / safe(df["ebitda"])
    df["interest_coverage"] = df["ebitda"] / safe(df["interest_expense"])
    df["cash_to_debt"] = df["cash"] / safe(df["total_debt"])
    df["operating_cash_flow_to_debt"] = df["operating_cash_flow"] / safe(df["total_debt"])
    df["debt_ratio"] = df["total_debt"] / safe(df["total_assets"])
    for col in ["revenue_growth", "ebitda_margin", "debt_to_ebitda",
                "interest_coverage", "cash_to_debt", "operating_cash_flow_to_debt"]:
        df[f"{col}_change"] = df.groupby("borrower_id")[col].diff()
    return df.replace([np.inf, -np.inf], np.nan)

def borrower_feature_frame(financials: pd.DataFrame, behaviour: pd.DataFrame,
                           macro: pd.DataFrame) -> pd.DataFrame:
    ratios = calculate_financial_ratios(financials)
    latest = ratios.sort_values("fiscal_year").groupby("borrower_id").tail(1)
    beh = behaviour.copy()
    beh["observation_month"] = pd.to_datetime(beh["observation_month"])
    beh = beh.sort_values(["borrower_id", "observation_month"])
    rows = []
    for bid, g in beh.groupby("borrower_id"):
        g = g.tail(12)
        current = g.iloc[-1]
        rows.append({
            "borrower_id": bid,
            "utilization": current["utilization"],
            "days_past_due": current["days_past_due"],
            "missed_payment_count": g["missed_payment"].sum(),
            "payment_ratio": np.clip(g["payment_amount"].mean() / .05, 0, 2),
            "utilization_change": current["utilization"] - g.iloc[0]["utilization"],
        })
    behf = pd.DataFrame(rows)
    mac = macro.copy()
    mac["observation_month"] = pd.to_datetime(mac["observation_month"])
    mac = mac.sort_values("observation_month").iloc[-1]
    out = latest.merge(behf, on="borrower_id", how="left")
    for c in ["gdp_growth","inflation","policy_rate","unemployment","fx_index","industrial_growth"]:
        out[c] = mac[c]
    return out
