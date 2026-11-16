from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["service"] == "ShopSense AI API"


def test_cors_allows_local_frontend():
    res = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_allows_fallback_dev_port():
    res = client.get("/health", headers={"Origin": "http://localhost:3002"})
    assert res.headers["access-control-allow-origin"] == "http://localhost:3002"


def test_cors_rejects_unknown_origin():
    res = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in res.headers
