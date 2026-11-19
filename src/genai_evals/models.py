from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class EvalCase(BaseModel):
    id: str
    query: str
    expected_entities: list[str]
    expected_attributes: list[str]
    context: str


class EvalResult(BaseModel):
    case_id: str
    entity_precision: float
    entity_recall: float
    attribute_coverage: float
    passed: bool
    provider: str
    tokens: int


class EvalSuite(BaseModel):
    suite_id: str
    results: list[EvalResult]
    mean_entity_recall: float
    pass_rate: float


class RAGRetrievalRequest(BaseModel):
    query: str
    documents: list[dict[str, Any]]
    top_k: int = 3
    dense_weight: float = 0.6


class RAGRetrievalResponse(BaseModel):
    query: str
    retrieved_chunks: list[dict[str, Any]]
    total_chunks: int


class RAGTripletEvalRequest(BaseModel):
    query: str
    retrieved_contexts: list[str]
    generated_answer: str
    ground_truth: str | None = None


class RAGTripletEvalResponse(BaseModel):
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevance: float
    passed: bool


class SecurityGuardrailRequest(BaseModel):
    prompt: str
    response: str


class SecurityGuardrailResponse(BaseModel):
    is_safe: bool
    injection_detected: bool
    pii_detected: bool
    detected_patterns: list[str]
    redacted_prompt: str
    redacted_response: str
