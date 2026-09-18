from __future__ import annotations
from dataclasses import dataclass
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix

FEATURES = [
    "revenue_growth","ebitda_margin","debt_to_ebitda","interest_coverage",
    "cash_to_debt","operating_cash_flow_to_debt","utilization",
    "utilization_change","days_past_due","missed_payment_count","payment_ratio",
    "policy_rate","inflation","unemployment","gdp_growth","deterioration_score",
    "covenant_breach_count"
]

@dataclass
class PDModelBundle:
    logistic: Pipeline
    gradient_boosting: Pipeline
    selected_model: str
    metrics: dict

def ks_statistic(y_true, p):
    order = np.argsort(p)
    y = np.asarray(y_true)[order]
    p_arr = np.asarray(p)[order]
    bad = y.sum()
    good = len(y) - bad
    if bad == 0 or good == 0:
        return 0.0
    cum_bad = np.cumsum(y) / bad
    cum_good = np.cumsum(1-y) / good
    return float(np.max(np.abs(cum_bad-cum_good)))

def metrics_for(y, p):
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else .5
    return {"roc_auc": float(auc), "gini": float(2*auc-1),
            "ks": ks_statistic(y,p), "brier": float(brier_score_loss(y,p))}

def train_pd_model(dataset: pd.DataFrame) -> PDModelBundle:
    data = dataset.sort_values("observation_date").copy()
    dates = pd.to_datetime(data["observation_date"])
    q70, q85 = dates.quantile(.70), dates.quantile(.85)
    train = data[dates <= q70]
    test = data[dates > q85]
    if len(test) < 50:
        cut = int(len(data)*.80); train, test = data.iloc[:cut], data.iloc[cut:]
    Xtr, ytr = train[FEATURES], train["default_12m"].astype(int)
    Xte, yte = test[FEATURES], test["default_12m"].astype(int)

    def pipe(model):
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("model", model)])
    logit = pipe(LogisticRegression(max_iter=2000, class_weight="balanced"))
    gb = pipe(GradientBoostingClassifier(random_state=42, n_estimators=180, max_depth=2,
                                         learning_rate=.05))
    logit.fit(Xtr, ytr); gb.fit(Xtr, ytr)
    pl, pg = logit.predict_proba(Xte)[:,1], gb.predict_proba(Xte)[:,1]
    ml, mg = metrics_for(yte, pl), metrics_for(yte, pg)
    selected = "gradient_boosting" if (mg["roc_auc"], -mg["brier"]) > (ml["roc_auc"], -ml["brier"]) else "logistic"
    return PDModelBundle(logit, gb, selected, {"logistic":ml, "gradient_boosting":mg,
                                               "test_rows":len(test)})

def predict_pd(bundle: PDModelBundle, df: pd.DataFrame) -> np.ndarray:
    model = bundle.gradient_boosting if bundle.selected_model == "gradient_boosting" else bundle.logistic
    return np.clip(model.predict_proba(df[FEATURES])[:,1], .0005, .9995)

def risk_grade(pd_values):
    p = np.asarray(pd_values)
    return np.select(
        [p < .01, p < .025, p < .05, p < .10, p < .20, p < .50],
        [1,2,3,4,5,6], default=7
    ).astype(int)

def rating_migration(current: pd.Series, future: pd.Series) -> pd.DataFrame:
    grades = list(range(1,8))
    m = pd.crosstab(current, future, dropna=False).reindex(index=grades, columns=grades, fill_value=0)
    return m

def save_bundle(bundle: PDModelBundle, path):
    with open(path, "wb") as f: pickle.dump(bundle, f)
