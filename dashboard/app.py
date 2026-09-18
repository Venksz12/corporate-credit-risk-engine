from pathlib import Path
import sqlite3
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.financial_ratios import borrower_feature_frame, calculate_financial_ratios
from src.risk.early_warning import deterioration_score, covenant_monitor, warning_signals
from src.risk.pd_model import PDModelBundle
from src.risk.lgd_ead_el import calculate_ead, calculate_lgd
from src.risk.stress_testing import run_stress, SCENARIOS

DB = ROOT / "data" / "credit_risk.db"
MODEL = ROOT / "models" / "pd_model.pkl"

st.set_page_config(page_title="Corporate Credit Risk", page_icon="📊", layout="wide")

@st.cache_data
def load_tables():
    con = sqlite3.connect(DB)
    tables = {t: pd.read_sql_query(f"SELECT * FROM {t}", con) for t in
              ["borrowers","financials","facilities","monthly_behaviour","macro_data","borrower_risk","facility_risk"]}
    con.close()
    return tables

@st.cache_resource
def load_model():
    import pickle
    with open(MODEL, "rb") as f: return pickle.load(f)

tables = load_tables()
model = load_model()
b, f, fac, beh, macro, br, fr = [tables[k] for k in
    ["borrowers","financials","facilities","monthly_behaviour","macro_data","borrower_risk","facility_risk"]]

st.sidebar.title("Corporate Credit Risk")
page = st.sidebar.radio("Navigation", [
    "Portfolio Overview","Borrower Early Warning","Facility Risk Analytics",
    "Rating Migration Analysis","Macro Stress Testing"
])

def money(x): return f"₹{x/1e7:,.1f} Cr"

if page == "Portfolio Overview":
    st.title("Portfolio Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total EAD", money(fr.ead.sum()))
    c2.metric("Expected Loss", money(fr.expected_loss.sum()))
    c3.metric("Weighted PD", f"{np.average(fr.pd, weights=fr.ead):.2%}")
    c4.metric("High-Risk Borrowers", int((br.risk_grade >= 5).sum()))
    x = fac.merge(b[["borrower_id","industry"]], on="borrower_id").merge(fr[["facility_id","expected_loss"]], on="facility_id")
    by_ind = x.groupby("industry", as_index=False)["expected_loss"].sum().sort_values("expected_loss", ascending=False)
    st.plotly_chart(px.bar(by_ind, x="industry", y="expected_loss", title="Expected Loss by Industry"), use_container_width=True)
    st.plotly_chart(px.histogram(br, x="risk_grade", nbins=7, title="Risk Grade Distribution"), use_container_width=True)

elif page == "Borrower Early Warning":
    st.title("Borrower Early Warning")
    names = b["borrower_id"].astype(str)+" — "+b["borrower_name"]
    selected = st.selectbox("Borrower", names.tolist())
    bid = int(selected.split(" — ")[0])
    row = br[br.borrower_id==bid].iloc[0]
    bb = b[b.borrower_id==bid].iloc[0]
    cols = st.columns(4)
    cols[0].metric("Deterioration Score", f"{row.deterioration_score:.1f}")
    cols[1].metric("Status", row.risk_status)
    cols[2].metric("PD", f"{row.pd:.2%}")
    cols[3].metric("Risk Grade", int(row.risk_grade))
    fin = calculate_financial_ratios(f[f.borrower_id==bid])
    st.plotly_chart(px.line(fin, x="fiscal_year", y=["debt_to_ebitda","interest_coverage"],
                            markers=True, title="Financial Ratio Trajectory"), use_container_width=True)
    recent = beh[beh.borrower_id==bid].copy()
    recent["observation_month"] = pd.to_datetime(recent.observation_month)
    st.plotly_chart(px.line(recent, x="observation_month", y="utilization", title="Facility Utilization"), use_container_width=True)
    st.subheader("Active warning signals")
    latest = borrower_feature_frame(f, beh, macro)
    r = latest[latest.borrower_id==bid].merge(br[br.borrower_id==bid][["borrower_id","deterioration_score"]], on="borrower_id")
    r = covenant_monitor(r)
    for signal in warning_signals(r.iloc[0]):
        st.warning(signal)
    if not warning_signals(r.iloc[0]): st.success("No active illustrative warning rules triggered.")

elif page == "Facility Risk Analytics":
    st.title("Facility Risk Analytics")
    x = fac.merge(b[["borrower_id","borrower_name","industry"]], on="borrower_id").merge(fr, on=["facility_id","borrower_id"])
    x["EAD"] = x["ead"].map(money); x["EL"] = x["expected_loss"].map(money)
    st.dataframe(x[["facility_id","borrower_name","facility_type","sanctioned_limit","drawn_amount",
                    "EAD","pd","lgd","EL","collateral_value"]].sort_values("expected_loss", ascending=False),
                 use_container_width=True, hide_index=True)

elif page == "Rating Migration Analysis":
    st.title("Rating Migration Analysis")
    rng = np.random.default_rng(42)
    current = br["risk_grade"].astype(int)
    # Illustrative one-period migration derived from the current risk distribution and a
    # deterioration-sensitive transition rule, not a historical bank transition matrix.
    future = np.clip(current + rng.choice([-1,0,0,0,1], len(current),
                                           p=[.10,.25,.30,.25,.10]), 1, 7)
    matrix = pd.crosstab(current, future).reindex(index=range(1,8), columns=range(1,8), fill_value=0)
    st.plotly_chart(px.imshow(matrix, text_auto=True, labels={"x":"Future Grade","y":"Current Grade"},
                              title="Illustrative 7×7 Rating Migration Matrix"), use_container_width=True)
    st.caption("Migration is illustrative because the project uses synthetic data and does not contain real historical ratings.")

else:
    st.title("Macroeconomic Stress Testing")
    scenario = st.radio("Scenario", list(SCENARIOS), horizontal=True)
    latest = borrower_feature_frame(f, beh, macro)
    latest = latest.merge(f[f.fiscal_year==f.fiscal_year.max()][
        ["borrower_id","revenue","ebitda","total_debt","interest_expense"]], on="borrower_id")
    latest["deterioration_score"] = br.set_index("borrower_id").reindex(latest.borrower_id)["deterioration_score"].values
    latest = covenant_monitor(latest)
    base_fac = fac.copy()
    stressed_b, stressed_f = run_stress(latest, base_fac, model, scenario)
    base_el = fr.expected_loss.sum()
    stress_el = stressed_f.expected_loss.sum()
    c1,c2,c3 = st.columns(3)
    c1.metric("Scenario", scenario)
    c2.metric("Portfolio EL", money(stress_el), delta=money(stress_el-base_el))
    c3.metric("High-Risk Borrowers", int((stressed_b.risk_grade>=5).sum()))
    st.dataframe(stressed_b.nlargest(15,"pd")[["borrower_id","pd","risk_grade","deterioration_score",
                                                "debt_to_ebitda","interest_coverage"]], use_container_width=True)
