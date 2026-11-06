import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient
from genai_evals.app import app

client = TestClient(app)

def test_health():
    assert client.get("/health").json()["status"] == "ok"

def test_suite_endpoint():
    payload = {"cases": [{"id": "c1", "query": "Show invoice for customer", "expected_entities": ["invoice", "customer"], "expected_attributes": ["invoice"], "context": ""}], "provider": "mock-maia"}
    r = client.post("/evaluate/suite", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["pass_rate"] == 1.0

def test_provider_endpoint():
    r = client.get("/providers/maia", params={"query": "hello"})
    assert r.json()["provider"] == "Maya"
