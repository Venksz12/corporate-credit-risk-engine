PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS borrowers (
    borrower_id INTEGER PRIMARY KEY,
    borrower_name TEXT NOT NULL,
    industry TEXT NOT NULL,
    country TEXT NOT NULL,
    annual_revenue REAL NOT NULL,
    employees INTEGER NOT NULL,
    incorporation_year INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS financials (
    borrower_id INTEGER NOT NULL,
    fiscal_year INTEGER NOT NULL,
    revenue REAL NOT NULL,
    ebitda REAL NOT NULL,
    net_income REAL NOT NULL,
    total_assets REAL NOT NULL,
    cash REAL NOT NULL,
    total_debt REAL NOT NULL,
    short_term_debt REAL NOT NULL,
    interest_expense REAL NOT NULL,
    operating_cash_flow REAL NOT NULL,
    PRIMARY KEY (borrower_id, fiscal_year),
    FOREIGN KEY (borrower_id) REFERENCES borrowers(borrower_id)
);

CREATE TABLE IF NOT EXISTS facilities (
    facility_id INTEGER PRIMARY KEY,
    borrower_id INTEGER NOT NULL,
    facility_type TEXT NOT NULL,
    seniority TEXT NOT NULL,
    sanctioned_limit REAL NOT NULL,
    drawn_amount REAL NOT NULL,
    interest_rate REAL NOT NULL,
    collateral_type TEXT NOT NULL,
    collateral_value REAL NOT NULL,
    haircut REAL NOT NULL,
    maturity_date TEXT NOT NULL,
    ccf REAL NOT NULL,
    FOREIGN KEY (borrower_id) REFERENCES borrowers(borrower_id)
);

CREATE TABLE IF NOT EXISTS monthly_behaviour (
    borrower_id INTEGER NOT NULL,
    observation_month TEXT NOT NULL,
    days_past_due INTEGER NOT NULL,
    drawn_amount REAL NOT NULL,
    utilization REAL NOT NULL,
    payment_amount REAL NOT NULL,
    missed_payment INTEGER NOT NULL,
    PRIMARY KEY (borrower_id, observation_month),
    FOREIGN KEY (borrower_id) REFERENCES borrowers(borrower_id)
);

CREATE TABLE IF NOT EXISTS macro_data (
    observation_month TEXT PRIMARY KEY,
    gdp_growth REAL NOT NULL,
    inflation REAL NOT NULL,
    policy_rate REAL NOT NULL,
    unemployment REAL NOT NULL,
    fx_index REAL NOT NULL,
    industrial_growth REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS borrower_risk (
    borrower_id INTEGER PRIMARY KEY,
    observation_date TEXT NOT NULL,
    deterioration_score REAL NOT NULL,
    risk_status TEXT NOT NULL,
    pd REAL NOT NULL,
    risk_grade INTEGER NOT NULL,
    covenant_breach_count INTEGER NOT NULL,
    FOREIGN KEY (borrower_id) REFERENCES borrowers(borrower_id)
);

CREATE TABLE IF NOT EXISTS facility_risk (
    facility_id INTEGER PRIMARY KEY,
    borrower_id INTEGER NOT NULL,
    ead REAL NOT NULL,
    pd REAL NOT NULL,
    lgd REAL NOT NULL,
    expected_loss REAL NOT NULL,
    FOREIGN KEY (facility_id) REFERENCES facilities(facility_id)
);
