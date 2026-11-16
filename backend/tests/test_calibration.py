"""Tests for probability calibration (ml/calibrator.py, ml/calibration.py)."""

import copy

import numpy as np
import pandas as pd
import pytest

from ml import calibration as cal
from ml import train
from ml.calibrator import ScoreCalibrator
from tests.synthetic import make_prepared


def _inflated(n=6000, seed=0, shift=1.5):
    """True probabilities p, labels y ~ Bernoulli(p), and over-confident scores."""
    rng = np.random.default_rng(seed)
    p = rng.beta(1.2, 5.0, n)
    y = (rng.random(n) < p).astype(int)
    logit = np.log(p / (1 - p)) + shift
    return p, y, 1 / (1 + np.exp(-logit))


def test_ece_small_when_calibrated_and_large_when_inflated():
    p, y, s = _inflated()
    assert cal.expected_calibration_error(y, p) < 0.03
    assert cal.expected_calibration_error(y, s) > 0.10


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrator_is_monotone_bounded_and_fixes_inflation(method):
    _, y, s = _inflated()
    c = ScoreCalibrator(method).fit(s, y)
    grid = np.linspace(0, 1, 501)
    out = c.predict(grid)
    assert np.all(np.diff(out) >= -1e-12)            # monotone -> ranking preserved
    assert out.min() >= 0 and out.max() <= 1
    assert abs(c.predict(s).mean() - y.mean()) < 0.01
    if method == "sigmoid":
        assert np.all(np.diff(out) > 0)               # strictly increasing


def test_invalid_method_rejected():
    with pytest.raises(ValueError):
        ScoreCalibrator("magic")


def test_acceptance_criteria():
    good = {"ece": 0.01, "calibration_in_the_large": 0.001, "brier_skill_score": 0.2, "brier": 0.10}
    ok, failures = cal.passes_criteria(good, raw_brier=0.12)
    assert ok and failures == []
    bad = {"ece": 0.2, "calibration_in_the_large": 0.1, "brier_skill_score": -0.1, "brier": 0.15}
    ok, failures = cal.passes_criteria(bad, raw_brier=0.12)
    assert not ok and len(failures) == 4


def test_validation_rejects_raw_inflated_scores_and_accepts_calibrated():
    _, y, s = _inflated()
    v = cal.validate_calibrators(s, y)
    raw_ok, _ = cal.passes_criteria(v["raw"], raw_brier=v["raw"]["brier"] + 1)
    assert not raw_ok                                   # raw scores fail (ECE / mean too high)
    assert v["methods"]["sigmoid"]["passes"]
    assert v["methods"]["sigmoid"]["brier"] < v["raw"]["brier"]


# -------------------------------------------- end-to-end on synthetic saved models
@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    prepared = make_prepared(n=1200, seed=3)
    results, models = train.run_training(prepared, fast=True, cv_folds=3, n_jobs=1, verbose=False)
    d = tmp_path_factory.mktemp("cal_models")
    train.save_artifacts(models, results, models_dir=d)
    return prepared, d


def _decision_part(rec):
    return {k: v for k, v in rec.items() if k not in ("heldout_test_check", "_plot", "created_at")}


def test_calibration_never_depends_on_test_data(saved):
    prepared, d = saved
    rec_a = cal.calibrate_variant("without_page_values", prepared, models_dir=d, n_jobs=1)

    # Replace the test split with garbage: random labels, altered features.
    scrambled = copy.copy(prepared)
    rng = np.random.default_rng(0)
    scrambled.y_test = pd.Series(rng.integers(0, 2, len(prepared.y_test)),
                                 index=prepared.y_test.index, name=prepared.y_test.name)
    scrambled.X_test = prepared.X_test.assign(ProductRelated_Duration=0.0)
    rec_b = cal.calibrate_variant("without_page_values", scrambled, models_dir=d, n_jobs=1)

    assert _decision_part(rec_a) == _decision_part(rec_b)
    assert rec_a["status"] in ("validated", "not_validated")
    if rec_a["status"] == "validated":
        assert (d / rec_a["calibrator_file"]).exists()
        assert rec_a["decision_threshold_probability"] is not None
    else:
        assert rec_a["calibrator_file"] is None and rec_a["failures"]


def test_calibration_record_pins_model_file(saved):
    prepared, d = saved
    rec = cal.calibrate_variant("with_page_values", prepared, models_dir=d, n_jobs=1)
    assert rec["model_sha256"] == cal.sha256_file(d / rec["model_file"])
    assert rec["criteria"] == cal.CRITERIA
