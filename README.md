# GenAI Tooling Evals

GenAI platform tooling evaluations for Mendix-style agentic low-code: Context Engineering, full RAG lifecycle, LLM-provider access and MLOps gates — built as the entrepreneur fix for the #1 G2 pain.

## Stack

- **Backend:** Python 3.11+, FastAPI, Pydantic v2 (TypeScript/Next.js equivalents in HEROS/RouteLLM-Gateway)
- **Infra:** AWS/Kubernetes-ready (Docker, K8s manifests pattern in go-ledger/msp-network-ops), GitHub Actions CI

## The `rea` fix — Maya entity/attribute queries

G2 4.4/5 (418 reviews): "Maya AI needs to improve — should handle queries more effectively, like dealing with entities and attributes", plus debugging visibility / version-control rigidity and governance friction.

This repo:

- **Entity/attribute eval harness:** deterministic mock retrieval with precision/recall/coverage per case, suite pass-rate and mean recall — the exact evaluation the JD asks the role to build.
- **LLM provider gateway:** Maia/OpenAI/Anthropic/Grok provider abstraction (mirrors llm-control-plane) so evaluations can run provider-agnostic.
- **MLOps gate:** suite only passes when entity recall ≥0.80 and attribute coverage ≥0.50 — a releasable quality bar, not a demo.

## Existing portfolio reuse (Path B justification)

`edgeinfer` (Matryoshka RAG), `GraphRAG-CodeAudit` (Leiden), `RouteLLM-Gateway` (semantic cache/NaN routing), `HEROS`/`HEROS-Firewall` (MCP runtime + OWASP ASI guardrails) and `llm-control-plane` cover RAG/MCP primitives, but none ships a Mendix-style GenAI tooling eval harness for Maya entity/attribute queries. Path B clean-room in Python/FastAPI gives 100% JD fidelity (Python RAG, TypeScript MCP equivalents, K8s ops).

## Run

```bash
pip install -e .[dev]
pytest -q
uvicorn genai_evals.app:app --reload
```

## Truthfulness

Mock retrieval is deterministic and synthetic; it does not call live Maia/LLM providers or production Mendix tenants. Production evals require provider credentials and customer domain models.
