"""API tests for POST /api/predict and GET /api/model-info (FastAPI TestClient).

Uses the real saved model in backend/models (skipped if it has not been trained).
"""

import dataclasses
import json
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_service, service_or_503
from app.config import get_settings
from app.main import app
from app.schemas import EXAMPLE_SESSION, SessionFeatures
from app.services.prediction import ModelUnavailableError, PredictionService, load_service
from ml import inference

MODELS_DIR = get_settings().models_dir
pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "model_manifest.json").exists(),
    reason="no trained model (run `python -m ml.train` and `python -m ml.calibration`)",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def service():
    return load_service(str(MODELS_DIR), None)


@pytest.fixture(scope="module")
def manifest():
    return json.loads((MODELS_DIR / "model_manifest.json").read_text())


@pytest.fixture
def override():
    """Temporarily replace the service dependency; always restored."""
    def _set(fn):
        app.dependency_overrides[get_service] = fn
    yield _set
    app.dependency_overrides.pop(get_service, None)


def session(**changes):
    return {**EXAMPLE_SESSION, **changes}


# -------------------------------------------------------------------- /health
def test_health_endpoint_unchanged(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"status", "service", "version", "environment", "timestamp"}
    assert body["status"] == "ok"


# ---------------------------------------------------------------- model-info
def test_model_info(client, manifest):
    r = client.get("/api/model-info")
    assert r.status_code == 200
    info = r.json()
    meta = manifest["models"][manifest["default_model"]]
    assert info["variant"] == "without_page_values" == manifest["default_model"]
    assert info["is_default"] is True
    assert info["algorithm"] == "random_forest"
    assert info["ignored_features"] == ["PageValues"]
    assert "PageValues" not in info["required_features"] and len(info["required_features"]) == 16
    assert info["decision_threshold"]["model_score"] == pytest.approx(meta["threshold"])
    assert info["test_metrics_at_threshold"] == pytest.approx(meta["test_metrics_at_threshold"])
    assert info["calibration"]["status"] == "validated"
    assert info["calibration"]["probability_available"] is True
    assert info["calibration"]["training_cv_metrics"]["ece"] <= info["calibration"]["criteria"]["max_ece"]
    assert set(SessionFeatures.model_fields) <= set(info["input_schema"]["properties"])
    assert "Jan" not in info["known_categories"]["Month"]
    assert info["limitations"] and info["valid_when"]
    # metric quoted in the limitations comes from the manifest, and every template is filled
    assert any(f"{meta['test_metrics_at_threshold']['pr_auc']:.2f}" in line for line in info["limitations"])
    assert not any("{" in line for line in info["limitations"])


# ------------------------------------------------------------------- predict
def test_predict_valid_session(client, service):
    r = client.post("/api/predict", json=EXAMPLE_SESSION)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["model_score"] <= 1.0
    assert 0.0 <= body["purchase_probability"] <= 1.0
    assert body["score_type"] == "calibrated_probability" and body["probability_calibrated"] is True
    assert body["predicted_purchase"] == (body["model_score"] >= body["decision_threshold"]["model_score"])
    assert body["model"]["variant"] == "without_page_values"
    assert body["warnings"] == []


def test_predict_matches_saved_pipeline_and_calibrator(client, service):
    body = client.post("/api/predict", json=EXAMPLE_SESSION).json()
    direct = inference.predict(service.model, EXAMPLE_SESSION)
    assert body["model_score"] == pytest.approx(direct["purchase_probability"].iloc[0], abs=1e-12)
    assert body["predicted_purchase"] == bool(direct["predicted_purchase"].iloc[0])
    assert body["purchase_probability"] == pytest.approx(
        float(service.calibrator.predict([body["model_score"]])[0]), abs=1e-12)


def test_decisions_agree_on_score_and_probability_thresholds(client):
    rng = np.random.default_rng(0)
    for _ in range(25):
        s = session(ProductRelated=int(rng.integers(1, 200)),
                    ProductRelated_Duration=float(rng.uniform(0, 8000)),
                    ExitRates=float(rng.uniform(0, 0.2)), BounceRates=float(rng.uniform(0, 0.1)),
                    VisitorType=str(rng.choice(["Returning_Visitor", "New_Visitor"])),
                    Month=str(rng.choice(["Mar", "May", "Nov", "Dec"])))
        b = client.post("/api/predict", json=s).json()
        thr = b["decision_threshold"]
        assert b["predicted_purchase"] == (b["model_score"] >= thr["model_score"])
        assert b["predicted_purchase"] == (b["purchase_probability"] >= thr["purchase_probability"])


@pytest.mark.parametrize("month", ["nov", "NOV", "November", " Nov "])
def test_month_is_normalised(client, month):
    base = client.post("/api/predict", json=EXAMPLE_SESSION).json()["model_score"]
    r = client.post("/api/predict", json=session(Month=month))
    assert r.status_code == 200 and r.json()["model_score"] == pytest.approx(base)


def test_page_values_is_ignored_with_warning(client):
    a = client.post("/api/predict", json=session(PageValues=0.0)).json()
    b = client.post("/api/predict", json=session(PageValues=300.0)).json()
    assert a["model_score"] == b["model_score"]
    assert any("PageValues" in w for w in b["warnings"])


def test_unseen_categories_produce_warnings(client):
    r = client.post("/api/predict", json=session(Month="Jan", Browser=99))
    assert r.status_code == 200
    w = " ".join(r.json()["warnings"])
    assert "Month='Jan'" in w and "Browser=99" in w


@pytest.mark.parametrize("payload, field", [
    ({k: v for k, v in EXAMPLE_SESSION.items() if k != "ExitRates"}, "ExitRates"),
    (session(ProductRelated=-1), "ProductRelated"),
    (session(BounceRates=1.5), "BounceRates"),
    (session(ExitRates="high"), "ExitRates"),
    (session(Month="Smarch"), "Month"),
    (session(VisitorType="Robot"), "VisitorType"),
    (session(Browser=2.5), "Browser"),
    (session(Browser=0), "Browser"),
    (session(Weekend="sometimes"), "Weekend"),
    (session(ProductRelated_Duration=10**6), "ProductRelated_Duration"),
    (session(PageValues=-5), "PageValues"),
    (session(Revenue=True), "Revenue"),            # target cannot be smuggled in
    (session(Typo_Field=1), "Typo_Field"),          # unknown fields are rejected
])
def test_invalid_input_returns_422(client, payload, field):
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 422
    locs = [e["loc"][-1] for e in r.json()["detail"]]
    assert field in locs


@pytest.mark.parametrize("body", [None, [], [EXAMPLE_SESSION], "text"])
def test_malformed_body_returns_422(client, body):
    r = client.post("/api/predict", json=body)
    assert r.status_code == 422


def test_cors_preflight_from_frontend_dev_port(client):
    r = client.options("/api/predict", headers={
        "Origin": "http://localhost:3002", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:3002"


def test_openapi_documents_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "post" in paths["/api/predict"] and "get" in paths["/api/model-info"] and "/health" in paths


# ------------------------------------------------- uncalibrated / unavailable
def test_uncalibrated_service_returns_score_not_probability(client, service, override):
    uncal = dataclasses.replace(service, calibrator=None, calibration_status="not_validated",
                                calibration_reason="test: criteria not met")
    override(lambda: uncal)
    body = client.post("/api/predict", json=EXAMPLE_SESSION).json()
    assert body["purchase_probability"] is None
    assert body["probability_calibrated"] is False
    assert body["score_type"] == "model_score"
    assert body["decision_threshold"]["purchase_probability"] is None
    assert body["predicted_purchase"] == (body["model_score"] >= body["decision_threshold"]["model_score"])
    assert any("not a probability" in w for w in body["warnings"])
    info = client.get("/api/model-info").json()
    assert info["calibration"]["probability_available"] is False
    assert info["calibration"]["status"] == "not_validated"


def test_missing_model_returns_503_but_health_still_works(client, tmp_path, override):
    override(lambda: service_or_503(tmp_path, None))
    assert client.post("/api/predict", json=EXAMPLE_SESSION).status_code == 503
    assert client.get("/api/model-info").status_code == 503
    assert client.get("/health").status_code == 200


def test_unknown_variant_is_unavailable():
    with pytest.raises(ModelUnavailableError):
        PredictionService.load(MODELS_DIR, "no_such_variant")


# --------------------------------------------- calibration integrity at load
@pytest.fixture(scope="module")
def models_copy(tmp_path_factory, manifest):
    d = tmp_path_factory.mktemp("models_copy")
    meta = manifest["models"][manifest["default_model"]]
    for f in (meta["model_file"], meta["calibration"]["calibrator_file"]):
        shutil.copy(MODELS_DIR / f, d / f)
    return d


def _write_manifest(d, manifest, mutate):
    m = json.loads(json.dumps(manifest))
    mutate(m["models"][m["default_model"]]["calibration"])
    (d / "model_manifest.json").write_text(json.dumps(m))


def test_load_uses_calibrator_only_when_valid(models_copy, manifest):
    _write_manifest(models_copy, manifest, lambda c: None)
    assert PredictionService.load(models_copy).probability_available

    _write_manifest(models_copy, manifest, lambda c: c.update(model_sha256="0" * 64))
    s = PredictionService.load(models_copy)
    assert s.calibration_status == "model_mismatch" and not s.probability_available

    _write_manifest(models_copy, manifest, lambda c: c.update(status="not_validated", failures=["x"]))
    s = PredictionService.load(models_copy)
    assert s.calibration_status == "not_validated" and s.calibrator is None
    assert s.predict(EXAMPLE_SESSION)["purchase_probability"] is None

    _write_manifest(models_copy, manifest, lambda c: c.update(calibrator_file="missing.joblib"))
    assert PredictionService.load(models_copy).calibration_status == "missing"
