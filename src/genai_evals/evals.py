from .models import EvalCase, EvalResult, EvalSuite

def _extract_entities(query: str, known: list[str]) -> list[str]:
    q = query.lower()
    seen: list[str] = []
    for e in known:
        if e.lower() in q and e not in seen:
            seen.append(e)
    return seen

def _extract_attributes(query: str, known: list[str]) -> list[str]:
    q = query.lower()
    seen: list[str] = []
    for a in known:
        if a.lower() in q and a not in seen:
            seen.append(a)
    return seen

def evaluate_case(case: EvalCase, provider: str = "mock-maia", tokens: int = 120) -> EvalResult:
    found_e = _extract_entities(case.query, case.expected_entities + ["account", "order", "invoice", "customer", "product"])
    found_a = _extract_attributes(case.query, case.expected_attributes)
    # Deterministic mock retrieval: query tokens that appear verbatim are "found"
    recall = len(found_e) / max(1, len(case.expected_entities))
    precision = len(found_e) / max(1, len(found_e))
    coverage = len(found_a) / max(1, len(case.expected_attributes))
    passed = recall >= 0.8 and coverage >= 0.5
    return EvalResult(case_id=case.id, entity_precision=precision, entity_recall=recall,
                      attribute_coverage=coverage, passed=passed, provider=provider, tokens=tokens)

def evaluate_suite(cases: list[EvalCase], provider: str = "mock-maia") -> EvalSuite:
    results = [evaluate_case(c, provider) for c in cases]
    mean_recall = sum(r.entity_recall for r in results) / max(1, len(results))
    pass_rate = sum(1 for r in results if r.passed) / max(1, len(results))
    return EvalSuite(suite_id=f"suite-{provider}", results=results, mean_entity_recall=mean_recall, pass_rate=pass_rate)

def provider_gateway(provider: str, query: str) -> dict:
    """LLM provider abstraction the JD asks for (accessing LLM providers)."""
    providers = {"maia": "Maya", "openai": "OpenAI", "anthropic": "Anthropic", "grok": "Grok"}
    normalized = providers.get(provider.lower(), provider)
    return {"provider": normalized, "model": f"{normalized.lower()}-default", "streaming": True, "query_len": len(query)}
