"""Tests for model training, leakage safety, reproducibility, saving/loading and inputs."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ml import data_preprocessing as dp
from ml import inference, train
from tests.synthetic import make_prepared, valid_record

FAST = dict(fast=True, cv_folds=3, n_jobs=1, verbose=False)


@pytest.fixture(scope="module")
def prepared():
    return make_prepared()


@pytest.fixture(scope="module")
def trained(prepared):
    return train.run_training(prepared, **FAST)


# ------------------------------------------------------------------ training
def test_all_models_trained_with_all_metrics(trained, prepared):
    results, models = trained
    assert set(results["feature_sets"]) == set(train.FEATURE_SETS)
    for fs, rec in results["feature_sets"].items():
        assert set(rec["models"]) == set(train.MODEL_LABELS)
        for name, m in rec["models"].items():
            t = m["test"]["at_tuned_threshold"]
            for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
                assert 0.0 <= t[k] <= 1.0, (fs, name, k)
            cm = t["confusion_matrix"]
            assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(prepared.y_test)
            assert 0.0 < m["test"]["threshold"] < 1.0
            assert m["cv"]["pr_auc"]["mean"] > 0
    # the synthetic signal is learnable: models beat the no-skill PR-AUC
    base = results["baseline_prior"]["pr_auc"]
    sel = results["feature_sets"]["with_page_values"]["selected_model"]
    assert results["feature_sets"]["with_page_values"]["models"][sel]["test"]["at_tuned_threshold"]["pr_auc"] > base


def test_selection_uses_cv_scores_only(trained):
    results, _ = trained
    for rec in results["feature_sets"].values():
        cv = {n: m["cv"]["pr_auc"]["mean"] for n, m in rec["models"].items()}
        assert rec["selected_model"] == max(cv, key=cv.get)


def test_select_best_ignores_test_metrics():
    def sr(name, cv):
        return train.SearchResult(name, {}, {"pr_auc": {"mean": cv, "std": 0.0}}, [], 0.0)
    # select_best only ever sees CV results; there is no way to pass test scores in.
    searches = {"a": sr("a", 0.60), "b": sr("b", 0.70), "c": sr("c", 0.65)}
    assert train.select_best(searches) == "b"


def test_preprocessing_is_inside_pipeline_and_fitted_on_train_only(trained, prepared):
    _, models = trained
    X_tr = prepared.X_train
    for fs, by_model in models.items():
        for tm in by_model.values():
            pipe = tm.estimator.estimator_
            assert isinstance(pipe, Pipeline)
            assert list(pipe.named_steps) == ["preprocess", "classifier"]
            pre = pipe.named_steps["preprocess"]
            assert isinstance(pre, ColumnTransformer)
            # imputer medians equal TRAINING medians -> fitted on train only
            for block in ("num", "num_log"):
                tr = pre.named_transformers_[block]
                cols = next(c for n, _, c in pre.transformers_ if n == block)
                np.testing.assert_allclose(tr.named_steps["impute"].statistics_,
                                           X_tr[cols].astype(float).median().to_numpy())


def test_target_and_excluded_features_not_used(trained):
    _, models = trained
    for fs, by_model in models.items():
        excluded = train.FEATURE_SETS[fs]
        for tm in by_model.values():
            names = tm.estimator.estimator_.named_steps["preprocess"].get_feature_names_out()
            assert not any(dp.TARGET in n for n in names)
            for f in excluded:
                assert not any(f in n for n in names)


def test_without_page_values_model_ignores_page_values(trained, prepared):
    _, models = trained
    est = models["without_page_values"]["gradient_boosting"].estimator
    X = prepared.X_test.copy()
    p1 = est.predict_proba(X)[:, 1]
    X["PageValues"] = 9999.0
    np.testing.assert_array_equal(p1, est.predict_proba(X)[:, 1])


# ----------------------------------------------------------- reproducibility
def test_training_is_reproducible(trained, prepared):
    results_a, models_a = trained
    results_b, models_b = train.run_training(prepared, **FAST)
    for fs in models_a:
        for name in models_a[fs]:
            np.testing.assert_array_equal(models_a[fs][name].test_proba, models_b[fs][name].test_proba)
            assert results_a["feature_sets"][fs]["models"][name]["test"] == \
                   results_b["feature_sets"][fs]["models"][name]["test"]


# ------------------------------------------------------------ save and load
@pytest.fixture(scope="module")
def saved_dir(trained, tmp_path_factory):
    results, models = trained
    d = tmp_path_factory.mktemp("models")
    train.save_artifacts(models, results, models_dir=d)
    return d


def test_saved_models_load_and_predict_identically(trained, prepared, saved_dir):
    results, models = trained
    manifest = json.loads((saved_dir / train.MANIFEST_NAME).read_text())
    assert manifest["default_model"] == train.DEFAULT_FEATURE_SET
    for fs in train.FEATURE_SETS:
        loaded = inference.load_model(fs, models_dir=saved_dir)
        original = models[fs][results["feature_sets"][fs]["selected_model"]].estimator
        np.testing.assert_array_equal(loaded.estimator.predict_proba(prepared.X_test),
                                      original.predict_proba(prepared.X_test))
        assert loaded.threshold == pytest.approx(original.best_threshold_)
        assert (saved_dir / manifest["models"][fs]["preprocessor_file"]).exists()
        assert loaded.required_features == train.required_features(train.FEATURE_SETS[fs])


def test_load_model_errors(tmp_path, saved_dir):
    with pytest.raises(inference.ModelNotFoundError):
        inference.load_model(models_dir=tmp_path)
    with pytest.raises(inference.ModelNotFoundError):
        inference.load_model("no_such_variant", models_dir=saved_dir)


# --------------------------------------------------------- prediction inputs
@pytest.fixture(scope="module")
def loaded(saved_dir):
    return inference.load_model("with_page_values", models_dir=saved_dir)


def test_predict_valid_record(loaded):
    out = inference.predict(loaded, valid_record())
    assert len(out) == 1
    p = out["purchase_probability"].iloc[0]
    assert 0.0 <= p <= 1.0
    assert out["predicted_purchase"].iloc[0] == (p >= loaded.threshold)


def test_predict_accepts_list_dataframe_and_normalises_labels(loaded):
    recs = [valid_record(), {**valid_record(), "Month": "june", "Weekend": "TRUE"},
            {**valid_record(), "Month": "NOV", "Revenue": True, "extra": "ignored"}]
    out_list = inference.predict(loaded, recs)
    out_df = inference.predict(loaded, pd.DataFrame(recs))
    assert len(out_list) == 3
    np.testing.assert_allclose(out_list["purchase_probability"], out_df["purchase_probability"])
    # Revenue present in the input must not change the score
    np.testing.assert_allclose(out_list["purchase_probability"].iloc[2],
                               inference.predict(loaded, valid_record())["purchase_probability"].iloc[0])


def test_missing_values_and_unseen_month_are_handled(loaded):
    rec = {**valid_record(), "PageValues": None, "ExitRates": np.nan, "Month": "Jan"}
    p = inference.predict(loaded, rec)["purchase_probability"].iloc[0]
    assert np.isfinite(p)


@pytest.mark.parametrize("change, message", [
    ({"BounceRates": 1.5}, "BounceRates"),
    ({"ProductRelated": -3}, "negative"),
    ({"ExitRates": "high"}, "non-numeric"),
    ({"Month": "Smarch"}, "Month"),
    ({"VisitorType": "Robot"}, "VisitorType"),
    ({"Browser": 2.5}, "integer code"),
    ({"Weekend": "sometimes"}, "Weekend"),
])
def test_invalid_inputs_rejected(loaded, change, message):
    with pytest.raises(inference.InputValidationError, match=message):
        inference.predict(loaded, {**valid_record(), **change})


def test_missing_required_feature_rejected(loaded):
    rec = valid_record()
    rec.pop("ExitRates")
    with pytest.raises(inference.InputValidationError, match="missing required features"):
        inference.predict(loaded, rec)


def test_page_values_not_required_for_leakage_safe_model(saved_dir):
    model = inference.load_model("without_page_values", models_dir=saved_dir)
    rec = valid_record()
    rec.pop("PageValues")
    assert np.isfinite(inference.predict(model, rec)["purchase_probability"].iloc[0])


def test_empty_input_rejected(loaded):
    with pytest.raises(inference.InputValidationError):
        inference.predict(loaded, [])


# ------------------------------------------------ real saved artefacts (if any)
real_artifacts = pytest.mark.skipif(
    not ((train.MODELS_DIR / train.MANIFEST_NAME).exists() and train.RESULTS_PATH.exists()
         and dp.RAW_CSV_PATH.exists()),
    reason="real models not trained yet (run `python -m ml.train`)",
)


@real_artifacts
def test_real_models_reproduce_reported_test_metrics():
    """Reloading the saved models and re-scoring the same test split gives the reported numbers."""
    results = json.loads(train.RESULTS_PATH.read_text())
    if results["protocol"]["fast_mode"]:
        pytest.skip("saved results come from a --fast smoke run")
    data = dp.prepare_datasets()
    for fs, rec in results["feature_sets"].items():
        model = inference.load_model(fs)
        reported = rec["models"][rec["selected_model"]]["test"]["at_tuned_threshold"]
        proba = model.estimator.predict_proba(data.X_test)[:, 1]
        pred = model.estimator.predict(data.X_test)
        again = train.classification_metrics(data.y_test, proba, pred)
        for k in ("pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy"):
            assert again[k] == pytest.approx(reported[k], abs=1e-12), (fs, k)
        assert again["confusion_matrix"] == reported["confusion_matrix"]


@real_artifacts
def test_real_default_model_scores_a_session():
    model = inference.load_model()
    out = inference.predict(model, valid_record())
    assert model.variant == json.loads((train.MODELS_DIR / train.MANIFEST_NAME).read_text())["default_model"]
    assert 0.0 <= out["purchase_probability"].iloc[0] <= 1.0
