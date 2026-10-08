from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_ok():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_response_has_request_id_header():
    with TestClient(app) as client:
        response = client.get("/health")
    assert "X-Request-ID" in response.headers
