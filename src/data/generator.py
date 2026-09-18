from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

INDUSTRIES = [
    "Manufacturing", "Energy", "Retail", "Technology", "Healthcare",
    "Pharmaceuticals", "Construction", "Real Estate", "Transportation",
    "Telecommunications",
]
COUNTRIES = ["India", "Singapore", "UAE", "United Kingdom", "Australia"]
FACILITY_TYPES = ["Term Loan", "Revolving Credit", "Working Capital"]
COLLATERAL = ["Property", "Receivables", "Inventory", "Equipment", "Cash", "None"]

def generate_database(db_path: str | Path, schema_path: str | Path,
                      seed: int = 42, n_borrowers: int = 1000,
                      n_facilities: int = 3500, months: int = 36) -> None:
    rng = np.random.default_rng(seed)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(db_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        con.executescript(f.read())

    for table in ["facility_risk", "borrower_risk", "monthly_behaviour",
                  "macro_data", "facilities", "financials", "borrowers"]:
        con.execute(f"DELETE FROM {table}")

    industries = rng.choice(INDUSTRIES, n_borrowers)
    countries = rng.choice(COUNTRIES, n_borrowers, p=[.72, .07, .06, .08, .07])
    profiles = rng.choice(["Healthy", "Moderate", "Distressed"], n_borrowers,
                          p=[.62, .28, .10])

    borrowers = []
    for i in range(1, n_borrowers + 1):
        base_rev = float(np.exp(rng.normal(np.log(650e6), .55)))
        employees = int(max(50, base_rev / rng.uniform(180_000, 450_000)))
        borrowers.append((i, f"Borrower {i:04d}", industries[i-1], countries[i-1],
                          base_rev, employees, int(rng.integers(1980, 2018))))
    con.executemany("INSERT INTO borrowers VALUES (?,?,?,?,?,?,?)", borrowers)

    years = [2023, 2024, 2025]
    financial_rows = []
    borrower_state = {}
    industry_margin = {x: m for x, m in zip(INDUSTRIES, rng.uniform(.10, .22, len(INDUSTRIES)))}

    for b in borrowers:
        bid, _, industry, _, base_rev, _, _ = b
        profile = profiles[bid - 1]
        if profile == "Healthy":
            growth = rng.normal(.07, .025); margin = industry_margin[industry] + .02
            leverage = rng.normal(2.1, .35)
        elif profile == "Moderate":
            growth = rng.normal(.02, .04); margin = industry_margin[industry] - .015
            leverage = rng.normal(3.5, .55)
        else:
            growth = rng.normal(-.09, .045); margin = max(.035, industry_margin[industry] - .08)
            leverage = rng.normal(5.8, .75)

        rev = base_rev / ((1 + growth) ** 2)
        for year in years:
            year_growth = growth + rng.normal(0, .015)
            rev *= (1 + year_growth)
            ebitda = max(rev * margin * rng.normal(1, .06), rev * .015)
            debt = max(10e6, ebitda * max(.7, leverage + rng.normal(0, .15)))
            short_debt = debt * rng.uniform(.15, .38)
            cash = max(2e6, rev * rng.uniform(.025, .12) * (1 if profile != "Distressed" else .65))
            interest = max(1e6, debt * rng.uniform(.055, .10))
            ocf = ebitda * rng.uniform(.55, .95) - rng.uniform(0, interest * .25)
            assets = max(rev * rng.uniform(.75, 1.7), debt + cash)
            net_income = ebitda - interest - rng.uniform(.01, .05) * rev
            financial_rows.append((bid, year, rev, ebitda, net_income, assets, cash,
                                   debt, short_debt, interest, ocf))
        borrower_state[bid] = (profile, growth, margin, leverage)

    con.executemany("INSERT INTO financials VALUES (?,?,?,?,?,?,?,?,?,?,?)", financial_rows)

    facility_rows = []
    for fid in range(1, n_facilities + 1):
        bid = int(rng.integers(1, n_borrowers + 1))
        profile, _, _, leverage = borrower_state[bid]
        ftype = rng.choice(FACILITY_TYPES, p=[.42, .33, .25])
        seniority = "Senior Secured" if rng.random() < .72 else "Senior Unsecured"
        limit = float(np.exp(rng.normal(np.log(80e6), .65)))
        utilization_base = {"Healthy": .52, "Moderate": .68, "Distressed": .86}[profile]
        utilization = np.clip(rng.normal(utilization_base + .04*(leverage-3), .10), .15, .99)
        drawn = limit * utilization
        rate = float(np.clip(rng.normal(.075 + .006*(leverage-3), .012), .045, .16))
        collateral_type = rng.choice(COLLATERAL, p=[.25,.22,.18,.18,.08,.09])
        collateral = 0.0 if collateral_type == "None" else drawn * rng.uniform(.65, 1.45)
        haircut_map = {"Property": .30, "Receivables": .25, "Inventory": .40,
                       "Equipment": .35, "Cash": .05, "None": 1.0}
        haircut = haircut_map[collateral_type]
        ccf = {"Term Loan": .05, "Revolving Credit": .70, "Working Capital": .55}[ftype]
        maturity = pd.Timestamp("2028-12-31") + pd.DateOffset(months=int(rng.integers(-12, 48)))
        facility_rows.append((fid, bid, ftype, seniority, limit, drawn, rate,
                              collateral_type, collateral, haircut, maturity.date().isoformat(), ccf))
    con.executemany("INSERT INTO facilities VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", facility_rows)

    month_dates = pd.date_range(end="2025-12-31", periods=months, freq="ME")
    macro_rows = []
    for j, dt in enumerate(month_dates):
        gdp = 6.0 + 0.55*np.sin(j/6) + rng.normal(0, .25)
        infl = 4.8 + 0.5*np.cos(j/5) + rng.normal(0, .18)
        rate = 6.5 + 0.4*np.sin(j/8) + rng.normal(0, .12)
        unemp = 6.1 + rng.normal(0, .25)
        fx = 100 + j*.25 + rng.normal(0, .8)
        ind = 5.0 + rng.normal(0, .5)
        macro_rows.append((dt.date().isoformat(), gdp, infl, rate, unemp, fx, ind))
    con.executemany("INSERT INTO macro_data VALUES (?,?,?,?,?,?,?)", macro_rows)

    behaviour_rows = []
    for bid in range(1, n_borrowers + 1):
        profile, growth, margin, leverage = borrower_state[bid]
        base_util = {"Healthy": .50, "Moderate": .66, "Distressed": .84}[profile]
        for j, dt in enumerate(month_dates):
            trend = j / max(1, months-1)
            util = np.clip(base_util + trend * (0.12 if profile=="Distressed" else
                                                0.06 if profile=="Moderate" else .015)
                           + rng.normal(0, .045), .05, .995)
            stress = max(0, leverage - 3) * .8 + max(0, .60-util) * 0
            dpd = int(np.clip(rng.poisson(max(.15, stress + (3 if profile=="Distressed" else .5))),
                              0, 90))
            missed = int(dpd >= 10 or (rng.random() < (.018 if profile=="Healthy" else
                                                        .05 if profile=="Moderate" else .13)))
            payment = max(0, (1-util) * 0.04 + rng.normal(.025, .01))
            drawn = util * 1.0
            behaviour_rows.append((bid, dt.date().isoformat(), dpd, drawn, util,
                                   payment, missed))
    con.executemany("INSERT INTO monthly_behaviour VALUES (?,?,?,?,?,?,?)", behaviour_rows)

    con.commit()
    con.close()
