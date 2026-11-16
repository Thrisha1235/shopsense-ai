"""Synthetic sessions with the real schema and a learnable purchase signal (no CSV needed)."""

import numpy as np
import pandas as pd

from ml import data_preprocessing as dp


def make_sessions(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    product_pages = rng.poisson(25, n)
    product_dur = product_pages * rng.gamma(2.0, 20.0, n)
    exit_rate = rng.beta(2, 40, n)
    page_values = np.where(rng.random(n) < 0.25, rng.gamma(2.0, 10.0, n), 0.0)
    logit = -2.2 + 0.08 * page_values + 0.0004 * product_dur - 25 * exit_rate
    revenue = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({
        "Administrative": rng.poisson(2, n),
        "Administrative_Duration": rng.gamma(1.0, 60.0, n),
        "Informational": rng.poisson(0.5, n),
        "Informational_Duration": rng.gamma(1.0, 30.0, n),
        "ProductRelated": product_pages,
        "ProductRelated_Duration": product_dur,
        "BounceRates": np.clip(exit_rate * rng.uniform(0.2, 0.9, n), 0, 0.2),
        "ExitRates": np.clip(exit_rate, 0, 0.2),
        "PageValues": page_values,
        "SpecialDay": rng.choice([0.0, 0.0, 0.0, 0.2, 0.6, 1.0], n),
        "Month": rng.choice(["Feb", "Mar", "May", "June", "Nov", "Dec"], n),
        "OperatingSystems": rng.integers(1, 4, n),
        "Browser": rng.integers(1, 4, n),
        "Region": rng.integers(1, 6, n),
        "TrafficType": rng.integers(1, 6, n),
        "VisitorType": rng.choice(dp.VISITOR_TYPES, n, p=[0.8, 0.18, 0.02]),
        "Weekend": rng.random(n) < 0.25,
        "Revenue": revenue,
    })


def make_prepared(n: int = 600, seed: int = 0) -> dp.PreparedData:
    raw = make_sessions(n, seed)
    X, y = dp.split_features_target(dp.clean(raw))
    X_tr, X_te, y_tr, y_te = dp.make_train_test_split(X, y)
    return dp.PreparedData(X_tr, X_te, y_tr, y_te, dp.build_preprocessor(), dp.validate(raw))


def valid_record() -> dict:
    return {
        "Administrative": 2, "Administrative_Duration": 45.0,
        "Informational": 0, "Informational_Duration": 0.0,
        "ProductRelated": 30, "ProductRelated_Duration": 1200.0,
        "BounceRates": 0.01, "ExitRates": 0.03, "PageValues": 0.0, "SpecialDay": 0.0,
        "Month": "Nov", "OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 2,
        "VisitorType": "Returning_Visitor", "Weekend": False,
    }
