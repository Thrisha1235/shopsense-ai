"""Security hardening tests (Phase 7): CORS, docs exposure, headers, error leakage,
explain rate limiting and model-artifact integrity."""

import json
import shutil
import threading

import pytest
from fastapi.testclient import TestClient

from app.api import explain as explain_api
from app.api.explain import explanation_service_or_503, get_explanation_service
from app.api.routes import MODEL_UNAVAILABLE, get_service, service_or_503
from app.config import Settings, get_settings
from app.main import app, create_app
from app.schemas import EXAMPLE_SESSION
from app.services.prediction import ModelUnavailableError, PredictionService
from ml import integrity
from ml.inference import ArtifactIntegrityError, safe_artifact_path, sha256_file

MODELS_DIR = get_settings().models_dir
HAS_MODELS = (MODELS_DIR / "model_manifest.json").exists()
needs_models = pytest.mark.skipif(not HAS_MODELS, reason="no trained models")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def override():
    yield lambda dep, fn: app.dependency_overrides.__setitem__(dep, fn)
    app.dependency_overrides.clear()


# ------------------------------------------------------------- production config
def test_production_hides_docs_and_rejects_localhost_origins():
    prod = create_app(Settings(environment="production", cors_origins=["https://shop.example.com"]))
    with TestClient(prod) as c:
        assert c.get("/health").status_code == 200
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert c.get(path).status_code == 404, path
        r = c.get("/health", headers={"Origin": "http://localhost:3002"})
        assert "access-control-allow-origin" not in r.headers          # dev regex not active
        r = c.get("/health", headers={"Origin": "https://shop.example.com"})
        assert r.headers["access-control-allow-origin"] == "https://shop.example.com"


def test_docs_can_be_enabled_explicitly_in_production():
    prod = create_app(Settings(environment="production", enable_docs=True))
    with TestClient(prod) as c:
        assert c.get("/openapi.json").status_code == 200


def test_development_keeps_docs(client):
    assert client.get("/docs").status_code == 200


# ------------------------------------------------------------------------ CORS
def test_cors_is_minimal(client):
    ok = client.options("/api/predict", headers={"Origin": "http://localhost:3002",
                        "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
    assert ok.status_code == 200
    assert "access-control-allow-credentials" not in ok.headers       # no cookies/credentials
    bad_method = client.options("/api/predict", headers={"Origin": "http://localhost:3002",
                                "Access-Control-Request-Method": "DELETE"})
    assert bad_method.status_code == 400
    bad_header = client.options("/api/predict", headers={"Origin": "http://localhost:3002",
                                "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "x-api-key"})
    assert bad_header.status_code == 400
    evil = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in evil.headers


def test_security_headers_present(client):
    for r in (client.get("/health"), client.post("/api/predict", json={"bad": 1})):
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["referrer-policy"] == "no-referrer"


# --------------------------------------------------------------- error leakage
def test_503_does_not_leak_server_paths(client, tmp_path, override):
    override(get_service, lambda: service_or_503(tmp_path, None))
    override(get_explanation_service, lambda: explanation_service_or_503(tmp_path, None))
    for r in (client.post("/api/predict", json=EXAMPLE_SESSION), client.get("/api/model-info"),
              client.post("/api/explain", json=EXAMPLE_SESSION)):
        assert r.status_code == 503
        detail = r.json()["detail"]
        assert str(tmp_path) not in detail and ":\\" not in detail and "/Users/" not in detail
    assert client.post("/api/predict", json=EXAMPLE_SESSION).json()["detail"] == MODEL_UNAVAILABLE


# --------------------------------------------------------- explain rate limit
@needs_models
def test_explain_concurrency_limit_returns_429(client, monkeypatch):
    sem = threading.BoundedSemaphore(1)
    monkeypatch.setattr(explain_api, "_explain_slots", sem)
    assert sem.acquire(blocking=False)          # simulate one explanation already running
    try:
        r = client.post("/api/explain", json=EXAMPLE_SESSION)
        assert r.status_code == 429 and r.headers["retry-after"] == "1"
    finally:
        sem.release()
    assert client.post("/api/explain", json=EXAMPLE_SESSION).status_code == 200   # slot released again


# ------------------------------------------------------------ artifact integrity
@pytest.fixture(scope="module")
def models_copy(tmp_path_factory):
    d = tmp_path_factory.mktemp("integrity")
    man = json.loads((MODELS_DIR / "model_manifest.json").read_text())
    meta = man["models"][man["default_model"]]
    for f in (meta["model_file"], meta["preprocessor_file"], meta["calibration"]["calibrator_file"]):
        shutil.copy(MODELS_DIR / f, d / f)
    return d, man


def _manifest(d, man, mutate):
    m = json.loads(json.dumps(man))
    m["models"] = {m["default_model"]: m["models"][m["default_model"]]}   # only the copied variant
    mutate(m["models"][m["default_model"]])
    (d / "model_manifest.json").write_text(json.dumps(m))


@needs_models
def test_manifest_records_checksums_of_the_served_artifacts():
    man = json.loads((MODELS_DIR / "model_manifest.json").read_text())
    meta = man["models"][man["default_model"]]
    sha = sha256_file(MODELS_DIR / meta["model_file"])
    assert meta["model_sha256"] == sha
    # the same model file was calibrated and explained (artifacts unchanged since Phases 4 and 6)
    assert meta["calibration"]["model_sha256"] == sha == meta["explanations"]["model_sha256"]
    assert integrity.verify(MODELS_DIR) == []


@needs_models
def test_tampered_model_is_refused_before_unpickling(models_copy):
    d, man = models_copy
    _manifest(d, man, lambda m: None)
    assert PredictionService.load(d).probability_available
    _manifest(d, man, lambda m: m.update(model_sha256="0" * 64))
    with pytest.raises(ModelUnavailableError, match="SHA-256"):
        PredictionService.load(d)


@needs_models
def test_tampered_calibrator_falls_back_to_score_only(models_copy):
    d, man = models_copy
    _manifest(d, man, lambda m: m["calibration"].update(calibrator_sha256="0" * 64))
    s = PredictionService.load(d)
    assert s.calibrator is None and not s.probability_available
    assert s.predict(EXAMPLE_SESSION)["purchase_probability"] is None


@needs_models
@pytest.mark.parametrize("name", ["../model.joblib", "..\\model.joblib", "sub/model.joblib", "", ".."])
def test_artifact_names_cannot_escape_models_dir(models_copy, name):
    d, man = models_copy
    with pytest.raises(ArtifactIntegrityError):
        safe_artifact_path(d, name)
    _manifest(d, man, lambda m: m.update(model_file=name))
    with pytest.raises(ModelUnavailableError):
        PredictionService.load(d)


@needs_models
def test_integrity_command_detects_modified_file(models_copy, tmp_path):
    d, man = models_copy
    _manifest(d, man, lambda m: None)
    assert integrity.verify(d) == []
    bad = tmp_path / "bad"
    shutil.copytree(d, bad)
    meta = man["models"][man["default_model"]]
    with (bad / meta["preprocessor_file"]).open("ab") as f:
        f.write(b"tampered")
    problems = integrity.verify(bad)
    assert len(problems) == 1 and "MISMATCH" in problems[0]
