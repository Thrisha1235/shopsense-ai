"""Deployment hardening: verified artifact download, general API rate limit, production headers."""

from __future__ import annotations

import hashlib
import io
import json

import pytest
from fastapi.testclient import TestClient

from app.api import limits
from app.config import Settings
from app.main import create_app
from ml import fetch_artifacts as fa

GOOD = b"model-bytes-v1"
GOOD_SHA = hashlib.sha256(GOOD).hexdigest()


def _models_dir(tmp_path, name="purchase_model_x.joblib", sha=GOOD_SHA):
    manifest = {"default_model": "x", "models": {"x": {"model_file": name, "model_sha256": sha}}}
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _opener(payload: bytes, calls: list):
    def open_(req, timeout):
        calls.append(req.full_url)
        return _Resp(payload)
    return open_


def test_fetch_writes_file_only_when_checksum_matches(tmp_path):
    d, calls = _models_dir(tmp_path), []
    assert fa.fetch("https://example.org/rel/", d, opener=_opener(GOOD, calls)) == []
    assert (d / "purchase_model_x.joblib").read_bytes() == GOOD
    assert calls == ["https://example.org/rel/purchase_model_x.joblib"]


def test_fetch_discards_tampered_download(tmp_path):
    d = _models_dir(tmp_path)
    problems = fa.fetch("https://example.org/rel", d, opener=_opener(b"evil pickle", []))
    assert len(problems) == 1 and "MISMATCH" in problems[0]
    assert not (d / "purchase_model_x.joblib").exists()
    assert not list(d.glob(".download-*"))              # temp file cleaned up


def test_fetch_skips_valid_existing_file(tmp_path):
    d, calls = _models_dir(tmp_path), []
    (d / "purchase_model_x.joblib").write_bytes(GOOD)
    assert fa.fetch("https://example.org/rel", d, opener=_opener(b"unused", calls)) == []
    assert calls == []


def test_fetch_replaces_corrupt_existing_file(tmp_path):
    d = _models_dir(tmp_path)
    (d / "purchase_model_x.joblib").write_bytes(b"corrupt")
    assert fa.fetch("https://example.org/rel", d, opener=_opener(GOOD, [])) == []
    assert (d / "purchase_model_x.joblib").read_bytes() == GOOD


def test_fetch_enforces_size_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(fa, "MAX_BYTES", 4)
    d = _models_dir(tmp_path)
    problems = fa.fetch("https://example.org/rel", d, opener=_opener(GOOD, []))
    assert problems and "larger than" in problems[0]
    assert not (d / "purchase_model_x.joblib").exists()


def test_fetch_rejects_traversal_names_in_manifest(tmp_path):
    d, calls = _models_dir(tmp_path, name="../evil.joblib"), []
    problems = fa.fetch("https://example.org/rel", d, opener=_opener(GOOD, calls))
    assert problems and calls == []


@pytest.mark.parametrize("base", ["http://example.org/rel", "ftp://example.org/rel", "file:///tmp/x"])
def test_fetch_requires_https(base, tmp_path):
    with pytest.raises(ValueError):
        fa.fetch(base, _models_dir(tmp_path), opener=_opener(GOOD, []))


def test_fetch_allows_http_only_for_localhost(tmp_path):
    assert fa.fetch("http://127.0.0.1:9000/rel", _models_dir(tmp_path), opener=_opener(GOOD, [])) == []


def test_fetch_cli_without_url_is_usage_error(monkeypatch):
    monkeypatch.delenv("MODEL_ARTIFACTS_URL", raising=False)
    assert fa.main([]) == 2


# ---- general API rate limit ----

def test_api_rate_limit_defaults():
    assert Settings(environment="development").api_rate_limit == 0
    assert Settings(environment="production").api_rate_limit == 120
    assert Settings(environment="production", api_rate_limit_per_minute=0).api_rate_limit == 0


def test_api_rate_limit_returns_429_but_health_is_exempt(monkeypatch):
    monkeypatch.setattr(limits, "api_limiter", limits.RateLimiter(2, 60))
    client = TestClient(create_app())
    codes = [client.get("/api/model-info").status_code for _ in range(3)]
    assert codes[:2] == [200, 200] and codes[2] == 429
    r = client.get("/api/analytics")
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    assert client.get("/health").status_code == 200


def test_masked_ip_is_coarse():
    assert limits.masked_ip("203.0.113.77") == "203.0.113.0/24"
    assert limits.masked_ip("2001:db8:abcd:12::1") == "2001:db8:abcd::/48"
    assert limits.masked_ip("testclient") == "unknown"


# ---- production headers / proxy ----

def test_hsts_only_outside_development():
    prod = TestClient(create_app(Settings(environment="production", cors_origins=["https://app.example"])))
    dev = TestClient(create_app(Settings(environment="development")))
    assert prod.get("/health").headers["Strict-Transport-Security"] == "max-age=31536000"
    assert "Strict-Transport-Security" not in dev.get("/health").headers


def test_production_cors_allows_only_configured_origin():
    client = TestClient(create_app(Settings(environment="production", cors_origins=["https://shopsense.vercel.app"])))
    pre = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
    ok = client.options("/api/predict", headers={"Origin": "https://shopsense.vercel.app", **pre})
    bad = client.options("/api/predict", headers={"Origin": "https://evil.example", **pre})
    local = client.options("/api/predict", headers={"Origin": "http://localhost:3000", **pre})
    assert ok.headers.get("access-control-allow-origin") == "https://shopsense.vercel.app"
    assert "access-control-allow-origin" not in bad.headers
    assert "access-control-allow-origin" not in local.headers
