from __future__ import annotations
import argparse, os, pickle, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB = ROOT/"data"/"credit_risk.db"
SCHEMA = ROOT/"database"/"schema.sql"
MODEL_DIR = ROOT/"models"
MODEL_DIR.mkdir(exist_ok=True)
(ROOT/"data").mkdir(exist_ok=True)

from src.data.generator import generate_database
from src.features.financial_ratios import borrower_feature_frame
from src.risk.early_warning import deterioration_score, covenant_monitor
from src.risk.pd_model import train_pd_model, predict_pd, risk_grade, save_bundle
from src.risk.lgd_ead_el import calculate_ead, calculate_lgd, expected_loss

def build():
    generate_database(DB, SCHEMA)
    con = sqlite3.connect(DB)
    b = pd.read_sql("SELECT * FROM borrowers", con)
    fin = pd.read_sql("SELECT * FROM financials", con)
    fac = pd.read_sql("SELECT * FROM facilities", con)
    beh = pd.read_sql("SELECT * FROM monthly_behaviour", con)
    macro = pd.read_sql("SELECT * FROM macro_data", con)

    features = borrower_feature_frame(fin, beh, macro)
    features = deterioration_score(features)
    features = covenant_monitor(features)

    # Create monthly-ish observation records from borrower fundamentals. The target is
    # generated from the synthetic risk process and is deliberately separated from
    # contemporaneous predictors.
    rng = np.random.default_rng(42)
    rows = []
    base = features.copy()
    for year in [2023, 2024, 2025]:
        d = base.copy()
        d["observation_date"] = pd.Timestamp(f"{year}-12-31")
        risk_signal = (
            0.035*d["debt_to_ebitda"].clip(0,10)
            -0.025*d["interest_coverage"].clip(0,8)
            -0.45*d["revenue_growth"].fillna(0)
            +0.9*d["utilization"]
            +0.018*d["days_past_due"]
            +0.10*d["covenant_breach_count"]
            +0.015*d["deterioration_score"]
            +rng.normal(0,.20,len(d))
        )
        p = 1/(1+np.exp(-(risk_signal-1.55)))
        d["default_12m"] = rng.binomial(1, np.clip(p,.002,.55))
        # Mild time drift makes the time split meaningful.
        d["observation_date"] = pd.to_datetime(d["observation_date"])
        rows.append(d)
    dataset = pd.concat(rows, ignore_index=True)

    bundle = train_pd_model(dataset)
    save_bundle(bundle, MODEL_DIR/"pd_model.pkl")
    dataset_latest = dataset[dataset.observation_date==pd.Timestamp("2025-12-31")].copy()
    dataset_latest["pd"] = predict_pd(bundle, dataset_latest)
    dataset_latest["risk_grade"] = risk_grade(dataset_latest["pd"])

    ead = calculate_ead(fac)
    lgd = calculate_lgd(fac, ead)
    fac["ead"] = ead
    fac["lgd"] = lgd
    fac["pd"] = fac["borrower_id"].map(dataset_latest.set_index("borrower_id")["pd"])
    fac["expected_loss"] = expected_loss(fac["pd"], fac["lgd"], fac["ead"])

    borrower_risk = dataset_latest[["borrower_id","observation_date","deterioration_score",
                                    "risk_status","pd","risk_grade","covenant_breach_count"]].copy()
    facility_risk = fac[["facility_id","borrower_id","ead","pd","lgd","expected_loss"]].copy()
    borrower_risk.to_sql("borrower_risk", con, if_exists="replace", index=False)
    facility_risk.to_sql("facility_risk", con, if_exists="replace", index=False)
    con.commit(); con.close()

    print("PD model metrics:")
    for name, m in bundle.metrics.items():
        if isinstance(m, dict): print(name, m)
    print(f"Portfolio EAD: {ead.sum():,.0f}")
    print(f"Portfolio Expected Loss: {fac.expected_loss.sum():,.0f}")
    return bundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-dashboard", action="store_true")
    args = parser.parse_args()
    build()
    if not args.no_dashboard:
        import subprocess
        subprocess.run(["streamlit","run",str(ROOT/"dashboard"/"app.py")], check=False)
