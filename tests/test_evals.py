import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genai_evals.models import EvalCase
from genai_evals.evals import evaluate_case, evaluate_suite, provider_gateway

def _case():
    return EvalCase(id="c1", query="Show the invoice attributes for customer account", expected_entities=["invoice", "customer", "account"], expected_attributes=["invoice", "customer"], context="domain model")

def test_entity_extraction():
    r = evaluate_case(_case())
    assert r.entity_recall == 1.0
    assert r.attribute_coverage == 1.0
    assert r.passed

def test_missing_entity_fails_recall():
    c = EvalCase(id="c2", query="Show invoices", expected_entities=["invoice", "customer", "account"], expected_attributes=["invoice"], context="")
    r = evaluate_case(c)
    assert r.entity_recall < 0.8
    assert not r.passed

def test_suite_pass_rate():
    cases = [_case(), EvalCase(id="c3", query="Show order for account", expected_entities=["order", "account"], expected_attributes=["order"], context="")]
    s = evaluate_suite(cases)
    assert s.pass_rate == 1.0
    assert s.mean_entity_recall == 1.0

def test_provider_gateway_normalizes():
    assert provider_gateway("maia", "hi")["provider"] == "Maya"
    assert provider_gateway("unknown", "hi")["provider"] == "unknown"
