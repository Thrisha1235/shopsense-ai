"""Tests for SHAP explanations: ml/explain.py (unit) and /api/explain (API, real saved model)."""

import dataclasses
import json
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from app.api.explain import explanation_service_or_503, get_explanation_service, global_explanation_path
from app.config import get_settings
from app.main import app
from app.schemas import EXAMPLE_SESSION
from app.services.explanation import ExplanationService, ExplanationUnavailableError, load_explanation_service
from app.services.prediction import PredictionService
from ml import data_preprocessing as dp
from ml import explain as ex
from ml import train
from tests.synthetic import make_prepared, valid_record


# ============================================================ unit: synthetic RF
@pytest.fixture(scope="module")
def synthetic():
    prepared = make_prepared(n=800, seed=5)
    exclude = ("PageValues",)
    model = train.fit_with_tuned_threshold(
        RandomForestClassifier(n_estimators=30, max_depth=6, class_weight="balanced_subsample", random_state=0),
        {}, prepared.X_train, prepared.y_train, exclude=exclude, cv_folds=3, n_jobs=1)
    feats = train.required_features(exclude)
    return prepared, model, ex.ShapExplainer(model.estimator_, feats), feats


def test_feature_map_matches_pipeline_column_order(synthetic):
    _, model, explainer, feats = synthetic
    pre = model.estimator_.named_steps["preprocess"]
    cols = ex.feature_map(pre)
    assert [c for c, _ in cols] == list(pre.get_feature_names_out())
    assert {f for _, f in cols} == set(feats)                  # every input feature is represented
    assert sum(1 for _, f in cols if f == "Month") > 1          # one-hot -> several columns
    assert sum(1 for _, f in cols if f == "ExitRates") == 1     # numeric -> one column
    assert "PageValues" not in {f for _, f in cols}


def test_explanation_is_additive_and_matches_pipeline_score(synthetic):
    prepared, model, explainer, _ = synthetic
    for i in range(5):
        X = prepared.X_test.iloc[[i]]
        e = explainer.explain_row(X)
        assert e["model_score"] == pytest.approx(model.predict_proba(X)[0, 1], abs=1e-12)
        assert e["additivity_error"] <= ex.ADDITIVITY_TOL
        assert e["baseline_score"] + sum(r["contribution"] for r in e["contributions"]) == \
               pytest.approx(e["model_score"], abs=ex.ADDITIVITY_TOL)


def test_aggregation_sums_components_exactly(synthetic):
    prepared, _, explainer, feats = synthetic
    e = explainer.explain_row(prepared.X_test.iloc[[0]])
    assert [r["feature"] for r in e["contributions"]] != []
    assert sorted(r["feature"] for r in e["contributions"]) == sorted(feats)
    for r in e["contributions"]:
        assert r["contribution"] == pytest.approx(sum(c["contribution"] for c in r["components"]), abs=1e-12)
        assert r["aggregated_from"] == len(r["components"])
    month = next(r for r in e["contributions"] if r["feature"] == "Month")
    assert sum(c["transformed_value"] for c in month["components"]) == 1.0   # exactly one month is "on"
    mags = [abs(r["contribution"]) for r in e["contributions"]]
    assert mags == sorted(mags, reverse=True)


def test_explanations_are_deterministic(synthetic):
    prepared, _, explainer, _ = synthetic
    X = prepared.X_test.iloc[[3]]
    assert explainer.explain_row(X) == explainer.explain_row(X)


def test_explain_row_rejects_multiple_rows(synthetic):
    prepared, _, explainer, _ = synthetic
    with pytest.raises(ValueError):
        explainer.explain_row(prepared.X_test.iloc[:2])


def test_global_importance_shares_and_order(synthetic):
    prepared, _, explainer, feats = synthetic
    g = ex.global_importance(explainer, prepared.X_test.iloc[:30])
    assert len(g["features"]) == len(feats)
    assert sum(f["share_of_total"] for f in g["features"]) == pytest.approx(1.0)
    vals = [f["mean_abs_contribution"] for f in g["features"]]
    assert vals == sorted(vals, reverse=True)
    assert g["max_additivity_error"] <= ex.ADDITIVITY_TOL


def test_stratified_sample_keeps_purchase_rate():
    prepared = make_prepared(n=1000, seed=1)
    s = ex.stratified_sample(prepared.X_train, prepared.y_train, 200)
    assert abs(prepared.y_train.loc[s.index].mean() - prepared.y_train.mean()) < 0.01
    assert s.index.isin(prepared.X_train.index).all()          # training rows only


# ============================================================ API: real saved model
MODELS_DIR = get_settings().models_dir
_manifest_path = MODELS_DIR / "model_manifest.json"
_has_expl = _manifest_path.exists() and "explanations" in json.loads(_manifest_path.read_text())["models"].get(
    "without_page_values", {})
real = pytest.mark.skipif(not _has_expl, reason="explainer not built (run `python -m ml.explain`)")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def service():
    return load_explanation_service(str(MODELS_DIR), None)


@pytest.fixture
def override():
    def _set(dep, fn):
        app.dependency_overrides[dep] = fn
    yield _set
    app.dependency_overrides.clear()


def session(**kw):
    return {**EXAMPLE_SESSION, **kw}


@real
def test_explain_endpoint_contract_and_additivity(client):
    r = client.post("/api/explain", json=EXAMPLE_SESSION)
    assert r.status_code == 200
    b = r.json()
    e = b["explanation"]
    assert e["explains"] == "raw_model_score"
    assert "uncalibrated" in e["output_units"]
    total = sum(c["contribution"] for c in b["contributions"])
    assert e["baseline_score"] + total == pytest.approx(e["model_score"], abs=ex.ADDITIVITY_TOL)
    assert e["additivity_error"] <= ex.ADDITIVITY_TOL
    feats = [c["feature"] for c in b["contributions"]]
    assert sorted(feats) == sorted(train.required_features(("PageValues",)))
    assert "PageValues" not in feats
    month = next(c for c in b["contributions"] if c["feature"] == "Month")
    info = client.get("/api/model-info").json()
    assert month["aggregated_from"] == len(info["known_categories"]["Month"]) == 10
    assert b["model"]["variant"] == "without_page_values"
    assert e["baseline_score"] == pytest.approx(0.5) and 0 < e["typical_session_score"] < e["baseline_score"]
    assert e["additivity_error"] <= 1e-6
    assert any("not causes" in n for n in b["interpretation_notes"])


@real
def test_explanation_prediction_identical_to_predict_endpoint(client):
    for s in (EXAMPLE_SESSION, session(ProductRelated=3, ProductRelated_Duration=40.0, ExitRates=0.15,
                                        BounceRates=0.1, Month="May")):
        p = client.post("/api/predict", json=s).json()
        b = client.post("/api/explain", json=s).json()
        assert b["prediction"] == p                                   # calibration + threshold preserved
        assert b["explanation"]["model_score"] == pytest.approx(p["model_score"], abs=1e-12)


@real
def test_calibrated_view_is_consistent_and_monotone(client, service):
    for s in (EXAMPLE_SESSION, session(ProductRelated=1, ProductRelated_Duration=0.0, ExitRates=0.2,
                                        BounceRates=0.2, Month="Feb")):
        b = client.post("/api/explain", json=s).json()
        cv, e = b["calibrated_view"], b["explanation"]
        assert cv["available"] is True
        assert cv["probability_at_baseline"] == pytest.approx(
            float(service.prediction.calibrator.predict([e["baseline_score"]])[0]), abs=1e-12)
        assert cv["purchase_probability"] == b["prediction"]["purchase_probability"]
        # monotone calibrator: score above baseline <=> probability above baseline probability
        assert (e["model_score"] > e["baseline_score"]) == (cv["purchase_probability"] > cv["probability_at_baseline"])


@real
def test_explain_normalises_month_and_ignores_page_values(client):
    base = client.post("/api/explain", json=EXAMPLE_SESSION).json()
    lower = client.post("/api/explain", json=session(Month="november")).json()
    pv = client.post("/api/explain", json=session(PageValues=250.0)).json()
    assert lower["contributions"] == base["contributions"]
    assert pv["contributions"] == base["contributions"]
    assert any("PageValues" in w for w in pv["warnings"])


@real
def test_unseen_category_explained_with_warning(client):
    b = client.post("/api/explain", json=session(Month="Jan")).json()
    month = next(c for c in b["contributions"] if c["feature"] == "Month")
    assert all(comp["transformed_value"] == 0.0 for comp in month["components"])   # no known month is "on"
    assert any("Month='Jan'" in w for w in b["warnings"])


@real
@pytest.mark.parametrize("payload", [
    session(BounceRates=1.5), session(ProductRelated=-1), session(Month="Smarch"), session(Browser=2.5),
    session(Revenue=True), session(Unknown=1), {k: v for k, v in EXAMPLE_SESSION.items() if k != "ExitRates"},
])
def test_explain_rejects_invalid_input(client, payload):
    assert client.post("/api/explain", json=payload).status_code == 422


@real
def test_explain_missing_model_returns_503(client, tmp_path, override):
    override(get_explanation_service, lambda: explanation_service_or_503(tmp_path, None))
    assert client.post("/api/explain", json=EXAMPLE_SESSION).status_code == 503
    assert client.get("/api/explain/global").status_code == 503
    assert client.get("/health").status_code == 200


@real
def test_consistency_guard_returns_500_not_a_wrong_explanation(client, service, override):
    class Tampered:
        shap_version = "test"
        def explain_row(self, X):
            e = service.explainer.explain_row(X)
            return {**e, "model_score": e["model_score"] + 0.1}
    override(get_explanation_service, lambda: dataclasses.replace(service, explainer=Tampered()))
    r = client.post("/api/explain", json=EXAMPLE_SESSION)
    assert r.status_code == 500 and "consistency" in r.json()["detail"]


@real
def test_uncalibrated_fallback(client, service, override):
    uncal_pred = dataclasses.replace(service.prediction, calibrator=None, calibration_status="not_validated")
    override(get_explanation_service, lambda: dataclasses.replace(service, prediction=uncal_pred))
    b = client.post("/api/explain", json=EXAMPLE_SESSION).json()
    assert b["calibrated_view"]["available"] is False
    assert b["calibrated_view"]["probability_at_baseline"] is None
    assert b["prediction"]["purchase_probability"] is None
    assert b["explanation"]["explains"] == "raw_model_score"


@real
def test_global_importance_endpoint(client, override, tmp_path):
    r = client.get("/api/explain/global")
    assert r.status_code == 200
    g = r.json()
    assert len(g["features"]) == 16 and g["explains"] == "raw_model_score"
    assert sum(f["share_of_total"] for f in g["features"]) == pytest.approx(1.0)
    assert g["max_additivity_error"] <= ex.ADDITIVITY_TOL
    assert "test split" in g["sample_source"]
    # missing file -> 503; file for a different model -> 503
    override(global_explanation_path, lambda: tmp_path / "missing.json")
    assert client.get("/api/explain/global").status_code == 503
    stale = json.loads((get_settings().reports_dir / "explanations" / "global_shap.json").read_text())
    stale["model_sha256"] = "0" * 64
    (tmp_path / "stale.json").write_text(json.dumps(stale))
    override(global_explanation_path, lambda: tmp_path / "stale.json")
    assert client.get("/api/explain/global").status_code == 503


# ------------------------------------------------ explainer integrity at load
@pytest.fixture(scope="module")
def models_copy(tmp_path_factory):
    d = tmp_path_factory.mktemp("expl_models")
    man = json.loads(_manifest_path.read_text())
    meta = man["models"][man["default_model"]]
    for f in (meta["model_file"], meta["calibration"]["calibrator_file"]):
        shutil.copy(MODELS_DIR / f, d / f)
    return d, man


def _write(d, man, mutate):
    m = json.loads(json.dumps(man))
    mutate(m["models"][m["default_model"]])
    (d / "model_manifest.json").write_text(json.dumps(m))
    return PredictionService.load(d)


@real
def test_explainer_refuses_stale_or_incomplete_setup(models_copy):
    d, man = models_copy
    assert ExplanationService.load(_write(d, man, lambda m: None), d).explainer.baseline > 0
    with pytest.raises(ExplanationUnavailableError, match="changed"):
        ExplanationService.load(_write(d, man, lambda m: m["explanations"].update(model_sha256="0" * 64)), d)
    with pytest.raises(ExplanationUnavailableError, match="No explanation setup"):
        ExplanationService.load(_write(d, man, lambda m: m.pop("explanations")), d)


def test_valid_record_helper_is_explainable_schema():
    # the shared synthetic record has every field the API requires
    assert set(valid_record()) == set(dp.FEATURES)
