# Corporate Credit Early-Warning & Expected-Loss Analytics System

An academic end-to-end prototype for analysing corporate credit risk at borrower and
facility level. It combines synthetic financial statements, facility characteristics,
repayment behaviour and macroeconomic variables to produce early-warning indicators,
one-year PD, illustrative LGD/EAD, expected loss, rating migration and macro stress tests.

> **Academic scope:** this is an educational prototype inspired by wholesale credit-risk
> concepts. It is not a regulatory-certified Basel model and is not a complete IFRS 9
> implementation.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

`run.py` initializes `data/credit_risk.db`, generates 1,000 borrowers, 3,500 facilities,
36 months of behaviour and macro data, trains the PD models, writes analytical tables,
and launches Streamlit.

To rebuild without launching the dashboard:

```bash
python run.py --no-dashboard
```

To package the repository:

```bash
python create_zip.py
```

## Architecture

Business problem → data model → financial ratios → trajectory → early warning →
covenants → PD → risk grade → migration → LGD/EAD → expected loss → stress testing →
portfolio analytics → dashboard.

## Main analytical methodology

- Deterioration score: illustrative weighted score using leverage, liquidity, coverage,
  revenue, cash flow and behaviour.
- PD: interpretable Logistic Regression plus Gradient Boosting baseline comparison.
- Validation: time-based split, ROC-AUC, Gini, KS and Brier score.
- Risk grades: seven project-defined PD bands.
- EAD: drawn exposure plus CCF × undrawn exposure.
- LGD: discounted net recoveries relative to EAD.
- Expected loss: PD × LGD × EAD.
- Stress testing: macro shocks propagate through revenue, EBITDA, interest expense,
  ratios, early warning and the trained PD model.

The synthetic data generator deliberately creates relationships such as higher leverage
→ weaker interest coverage → greater deterioration → higher default propensity, rather
than drawing every variable independently.

## Dashboard pages

1. Portfolio Overview
2. Borrower Early Warning
3. Facility Risk Analytics
4. Rating Migration Analysis
5. Macro Stress Testing

## Important limitations

The data are synthetic. Risk-grade thresholds, deterioration weights, CCFs, collateral
haircuts, recovery assumptions and stress transmission coefficients are project-defined
illustrative assumptions. Results should not be interpreted as actual bank risk estimates.
