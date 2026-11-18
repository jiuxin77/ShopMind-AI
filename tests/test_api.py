from fastapi.testclient import TestClient

from shopmind.api import app

client = TestClient(app)


def test_liveness_and_request_id():
    response = client.get("/health/live", headers={"X-Request-ID": "test-request"})
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}
    assert response.headers["X-Request-ID"] == "test-request"


def test_readiness_checks_demo_data():
    response = client.get("/health/ready")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["data"] == {"users": 300, "products": 200, "interactions": 5000}


def test_llm_endpoint_requires_server_side_key():
    response = client.post("/llm-shopping-agent", json={"need": "500元以内的通勤女装"})
    assert response.status_code == 503
    assert response.json()["detail"] == "OPENAI_API_KEY is not configured"
