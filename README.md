# GenAI Tooling Evals

GenAI platform tooling evaluations, context engineering, agentic RAG lifecycle, multi-agent orchestration, and enterprise MLOps gates.

## Architecture & Modules

- **Agentic Document Chunker (`agentic_rag_evals.py`):** Recursive and semantic document chunking preserving paragraph boundaries and windowed token overlaps with deterministic SHA-256 chunk hashing.
- **Hybrid Vector Retriever & RRF (`agentic_rag_evals.py`):** Combines dense vector cosine similarity with sparse BM25 term weighting using Reciprocal Rank Fusion (RRF) and cross-encoder semantic reranking.
- **Multi-Agent Workflow Engine (`agentic_rag_evals.py`):** ReAct and Plan-and-Execute multi-agent orchestration (Planner, Worker, Critic) with tool calling schemas and loop/recursion prevention guardrails.
- **LLM-as-a-Judge Evaluation Harness (`agentic_rag_evals.py`):** Automated evaluation computing Context Precision, Context Recall, Faithfulness, and Answer Relevancy.
- **Enterprise Security Guardrails (`agentic_rag_evals.py`):** Prompt injection defense detecting jailbreak patterns and automated PII masking (emails, phone numbers, credit cards, identity IDs).
- **Adoption Metrics & Merge Gates (`adoption.py`):** Measures team adoption shares, AI acceptance rates, revert rates, and eval pass rates.
- **FastAPI Service (`app.py`):** Production REST API exposing health checks, RAG retrieval endpoints, triplet evaluation, and security guardrail inspection.

## Stack

- **Backend:** Python 3.11+, FastAPI, Pydantic v2
- **Testing:** Pytest, HTTPX, AnyIO
- **CI/CD:** GitHub Actions automated test workflow

## Run

```bash
pip install -e .[dev]
pytest -v
uvicorn genai_evals.app:app --reload
```
