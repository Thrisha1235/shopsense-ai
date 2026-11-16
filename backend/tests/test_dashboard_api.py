"""Tests for the read-only dashboard endpoints (/api/analytics, /api/model-performance)."""

import json

import pytest
from fastapi.testclient import TestClient

from app.api.dashboard import dashboard_dir
from app.config import get_settings
from app.main import app

REPORTS = get_settings().reports_dir
DASH = REPORTS / "dashboard"
pytestmark = pytest.mark.skipif(
    not ((DASH / "analytics.json").exists() and (DASH / "model_performance.json").exists()),
    reason="dashboard data not generated (run `python -m ml.dashboard_data`)",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_analytics_totals_are_consistent_with_eda(client):
    r = client.get("/api/analytics")
    assert r.status_code == 200
    a = r.json()
    eda = json.loads((REPORTS / "eda_stats.json").read_text())["dataset"]
    o = a["overview"]
    assert (o["sessions"], o["purchases"], o["no_purchases"]) == (eda["rows"], eda["purchases"], eda["no_purchases"])
    assert o["conversion_rate"] == pytest.approx(eda["conversion_rate"])
    for block in (a["monthly"], a["visitor_types"], a["weekend"], a["traffic_sources"]["rows"],
                  a["product_duration"]["quintiles"], a["bounce_rate"]["bands"], a["exit_rate"]["bands"]):
        assert sum(r["sessions"] for r in block) == o["sessions"]
        assert sum(r["purchases"] for r in block) == o["purchases"]
        for row in block:
            assert row["ci95_low"] <= row["conversion_rate"] <= row["ci95_high"]
    assert o["months_absent"] == ["Jan", "Apr"]
    assert a["traffic_sources"]["rows"][-1]["source"] == "Other"


def test_model_performance_matches_phase3_results(client):
    r = client.get("/api/model-performance")
    assert r.status_code == 200
    p = r.json()
    results = json.loads((REPORTS / "model_results.json").read_text())
    assert p["default_variant"] == "without_page_values"
    assert len(p["models"]) == 6
    for m in p["models"]:
        src = results["feature_sets"][m["feature_set"]]["models"][m["model"]]["test"]["at_tuned_threshold"]
        assert m["test"]["pr_auc"] == pytest.approx(src["pr_auc"])
        assert m["confusion_matrix"] == src["confusion_matrix"]
    for variant, c in p["curves"].items():
        fpr = [pt["fpr"] for pt in c["roc"]]
        assert fpr == sorted(fpr) and fpr[0] == 0.0 and fpr[-1] == 1.0
        sel = next(m for m in p["models"] if m["feature_set"] == variant and m["selected"])
        assert c["operating_point"]["recall"] == pytest.approx(sel["test"]["recall"])
        assert c["operating_point"]["precision"] == pytest.approx(sel["test"]["precision"])
    assert p["calibration"]["without_page_values"]["status"] == "validated"


def test_missing_dashboard_data_returns_503(client, tmp_path):
    app.dependency_overrides[dashboard_dir] = lambda: tmp_path
    try:
        assert client.get("/api/analytics").status_code == 503
        assert client.get("/api/model-performance").status_code == 503
    finally:
        app.dependency_overrides.pop(dashboard_dir, None)


def test_dashboard_endpoints_allow_frontend_origin(client):
    r = client.get("/api/analytics", headers={"Origin": "http://localhost:3002"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:3002"
