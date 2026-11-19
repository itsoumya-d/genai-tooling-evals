from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .agentic_rag_evals import (
    AgenticDocumentChunker,
    HybridVectorRetriever,
    LLMJudgeEvaluator,
)
from .evals import evaluate_case, evaluate_suite, provider_gateway
from .models import (
    EvalCase,
    EvalSuite,
    RAGRetrievalRequest,
    RAGRetrievalResponse,
    RAGTripletEvalRequest,
    RAGTripletEvalResponse,
    SecurityGuardrailRequest,
    SecurityGuardrailResponse,
)


class SuiteRequest(BaseModel):
    cases: list[EvalCase]
    provider: str = "mock-maia"


app = FastAPI(title="GenAI Tooling Evals")
_evaluator = LLMJudgeEvaluator()
_chunker = AgenticDocumentChunker()


@app.get("/health")
def health():
    return {"status": "ok", "service": "genai-tooling-evals"}


@app.post("/evaluate/case")
def eval_case(case: EvalCase, provider: str = "mock-maia"):
    return evaluate_case(case, provider)


@app.post("/evaluate/suite", response_model=EvalSuite)
def eval_suite(req: SuiteRequest):
    return evaluate_suite(req.cases, req.provider)


@app.get("/providers/{provider}")
def gateway(provider: str, query: str = "hello"):
    return provider_gateway(provider, query)


@app.post("/agentic-rag/retrieve", response_model=RAGRetrievalResponse)
def rag_retrieve(req: RAGRetrievalRequest):
    chunks = []
    for doc in req.documents:
        doc_id = doc.get("id", "doc")
        text = doc.get("text", "")
        meta = doc.get("metadata", {})
        doc_chunks = _chunker.chunk_document(text, doc_id=doc_id, metadata=meta)
        chunks.extend(doc_chunks)

    retriever = HybridVectorRetriever()
    retriever.index_chunks(chunks)
    hits = retriever.hybrid_rrf_search(
        query=req.query,
        top_k=req.top_k,
        dense_weight=req.dense_weight,
    )

    retrieved = [
        {
            "id": c.id,
            "content": c.content,
            "score": score,
            "metadata": c.metadata,
        }
        for c, score in hits
    ]
    return RAGRetrievalResponse(
        query=req.query,
        retrieved_chunks=retrieved,
        total_chunks=len(chunks),
    )


@app.post("/agentic-rag/evaluate", response_model=RAGTripletEvalResponse)
def rag_evaluate(req: RAGTripletEvalRequest):
    score = _evaluator.evaluate_rag_triplet(
        query=req.query,
        retrieved_contexts=req.retrieved_contexts,
        generated_answer=req.generated_answer,
        ground_truth=req.ground_truth,
    )
    return RAGTripletEvalResponse(
        context_precision=score.context_precision,
        context_recall=score.context_recall,
        faithfulness=score.faithfulness,
        answer_relevance=score.answer_relevance,
        passed=score.passed,
    )


@app.post("/security/guardrails", response_model=SecurityGuardrailResponse)
def security_guardrails(req: SecurityGuardrailRequest):
    res = _evaluator.evaluate_security_guardrails(
        prompt=req.prompt,
        response=req.response,
    )
    return SecurityGuardrailResponse(
        is_safe=res.is_safe,
        injection_detected=res.injection_detected,
        pii_detected=res.pii_detected,
        detected_patterns=res.detected_patterns,
        redacted_prompt=res.redacted_prompt,
        redacted_response=res.redacted_response,
    )
