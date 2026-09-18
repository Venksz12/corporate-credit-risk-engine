from __future__ import annotations
import numpy as np
import pandas as pd
from .early_warning import deterioration_score, covenant_monitor
from .pd_model import predict_pd, risk_grade
from .lgd_ead_el import calculate_ead, calculate_lgd, expected_loss

SCENARIOS = {
    "Baseline": {"gdp_shock": 0.0, "revenue_shock": 0.0, "rate_bps": 0, "collateral_shock": 0.0},
    "Moderate Stress": {"gdp_shock": -3.0, "revenue_shock": -0.05, "rate_bps": 150, "collateral_shock": -0.10},
    "Severe Stress": {"gdp_shock": -6.0, "revenue_shock": -0.15, "rate_bps": 300, "collateral_shock": -0.20},
}

def run_stress(base: pd.DataFrame, facilities: pd.DataFrame, model, scenario: str) -> tuple[pd.DataFrame,pd.DataFrame]:
    s = SCENARIOS[scenario]
    x = base.copy()
    x["revenue"] *= (1+s["revenue_shock"])
    x["ebitda"] *= (1+s["revenue_shock"] * 1.15)
    x["interest_expense"] *= (1+s["rate_bps"]/10000)
    x["debt_to_ebitda"] = x["total_debt"] / x["ebitda"].clip(lower=1)
    x["interest_coverage"] = x["ebitda"] / x["interest_expense"].clip(lower=1)
    x["revenue_growth"] = x["revenue_growth"].fillna(.03) + s["revenue_shock"]
    x["policy_rate"] += s["rate_bps"]/100
    x["gdp_growth"] += s["gdp_shock"]
    x["deterioration_score"] = deterioration_score(x)["deterioration_score"]
    x = covenant_monitor(x)
    x["pd"] = predict_pd(model, x)
    x["risk_grade"] = risk_grade(x["pd"])
    fac = facilities.copy()
    fac["collateral_value"] *= (1+s["collateral_shock"])
    ead = calculate_ead(fac)
    lgd = calculate_lgd(fac, ead)
    fac["pd"] = fac["borrower_id"].map(x.set_index("borrower_id")["pd"])
    fac["lgd"] = lgd
    fac["ead"] = ead
    fac["expected_loss"] = expected_loss(fac["pd"], fac["lgd"], fac["ead"])
    return x, fac
