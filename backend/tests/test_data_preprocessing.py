import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from ml import data_preprocessing as dp


def make_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Small synthetic frame with the real schema (no dependency on the CSV)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Administrative": rng.integers(0, 10, n),
        "Administrative_Duration": rng.uniform(0, 500, n),
        "Informational": rng.integers(0, 5, n),
        "Informational_Duration": rng.uniform(0, 200, n),
        "ProductRelated": rng.integers(0, 80, n),
        "ProductRelated_Duration": rng.uniform(0, 5000, n),
        "BounceRates": rng.uniform(0, 0.2, n),
        "ExitRates": rng.uniform(0, 0.2, n),
        "PageValues": rng.uniform(0, 50, n),
        "SpecialDay": rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], n),
        "Month": rng.choice(["Feb", "Mar", "May", "June", "Nov"], n),
        "OperatingSystems": rng.integers(1, 5, n),
        "Browser": rng.integers(1, 5, n),
        "Region": rng.integers(1, 9, n),
        "TrafficType": rng.integers(1, 10, n),
        "VisitorType": rng.choice(dp.VISITOR_TYPES, n),
        "Weekend": rng.choice([True, False], n),
        "Revenue": np.r_[np.ones(n // 4, bool), np.zeros(n - n // 4, bool)],
    })


def test_valid_frame_passes():
    report = dp.validate(make_df())
    assert report.ok, report.errors
    assert report.n_columns == 18


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.drop(columns=["ExitRates"]), "missing columns"),
        (lambda d: d.assign(BounceRates=1.5), "above 1"),
        (lambda d: d.assign(ProductRelated=-1), "negative"),
        (lambda d: d.assign(Month="Smarch"), "Month has unknown"),
        (lambda d: d.assign(VisitorType="Alien"), "VisitorType has unknown"),
        (lambda d: d.assign(Revenue="maybe"), "not boolean"),
    ],
)
def test_validation_catches_bad_data(mutate, message):
    report = dp.validate(mutate(make_df()))
    assert not report.ok
    assert any(message in e for e in report.errors), report.errors
    with pytest.raises(dp.DataValidationError):
        report.raise_if_invalid()


def test_clean_does_not_mutate_input_and_normalises():
    raw = make_df()
    snapshot = raw.copy(deep=True)
    out = dp.clean(raw)
    pd.testing.assert_frame_equal(raw, snapshot)  # original untouched
    assert "June" not in set(out["Month"]) and set(out["Month"]) <= set(dp.MONTH_ORDER)
    assert set(out[dp.TARGET].unique()) <= {0, 1}


def test_clean_keeps_missing_features_for_pipeline_and_drops_missing_target():
    raw = make_df().astype({"PageValues": float})
    raw.loc[0, "PageValues"] = np.nan
    raw[dp.TARGET] = raw[dp.TARGET].astype(object)
    raw.loc[1, dp.TARGET] = np.nan
    out = dp.clean(raw)
    assert len(out) == len(raw) - 1
    assert out["PageValues"].isna().sum() == 1  # left for the imputer


def test_target_not_in_features():
    X, y = dp.split_features_target(dp.clean(make_df()))
    assert dp.TARGET not in X.columns
    assert y.name == dp.TARGET


def test_split_is_stratified():
    X, y = dp.split_features_target(dp.clean(make_df(200)))
    _, _, y_tr, y_te = dp.make_train_test_split(X, y, test_size=0.25)
    assert abs(y_tr.mean() - y_te.mean()) < 0.03


def test_preprocessor_is_unfitted_and_learns_from_train_only():
    X, y = dp.split_features_target(dp.clean(make_df(200)))
    X_tr, X_te, _, _ = dp.make_train_test_split(X, y)
    pre = dp.build_preprocessor()
    with pytest.raises(NotFittedError):
        check_is_fitted(pre)

    pre.fit(X_tr)
    # The median imputer saw only training rows.
    num_imputer = pre.named_transformers_["num"].named_steps["impute"]
    linear_cols = [c for c in dp.NUMERIC_FEATURES if c not in dp.LOG_FEATURES]
    np.testing.assert_allclose(num_imputer.statistics_, X_tr[linear_cols].median().to_numpy())

    Xt = pre.transform(X_te)
    assert Xt.shape[0] == len(X_te) and not np.isnan(Xt).any()


def test_unseen_category_and_missing_values_at_inference():
    X, y = dp.split_features_target(dp.clean(make_df(200)))
    pre = dp.build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row["Month"] = "Jan"          # never seen in training
    row["PageValues"] = np.nan    # missing at inference
    out = pre.transform(row)
    assert out.shape == (1, len(pre.get_feature_names_out()))
    assert not np.isnan(out).any()


def test_exclude_drops_feature():
    pre = dp.build_preprocessor(exclude=("PageValues",))
    X, _ = dp.split_features_target(dp.clean(make_df()))
    names = pre.fit(X).get_feature_names_out()
    assert not any("PageValues" in n for n in names)


@pytest.mark.skipif(not dp.RAW_CSV_PATH.exists(), reason="dataset not downloaded")
def test_real_dataset_validates_and_is_unchanged():
    assert dp.verify_checksum() in (True, None)
    report = dp.validate(dp.load_raw())
    assert report.ok, report.errors
    assert (report.n_rows, report.n_columns) == (12330, 18)
