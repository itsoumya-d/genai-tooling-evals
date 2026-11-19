"""Enterprise Agentic RAG, Multi-Agent Orchestration, and LLM Guardrail Evaluations.

Provides production-grade building blocks for:
- Semantic & recursive document chunking with metadata preservation.
- Hybrid vector retrieval combining dense cosine similarity and sparse BM25 with Reciprocal Rank Fusion (RRF).
- Cross-encoder reranking heuristic for optimal context synthesis.
- Multi-agent workflow execution (Planner, Worker, Critic) with ReAct loops and loop/recursion guardrails.
- LLM-as-a-Judge RAG evaluation (Context Precision, Faithfulness, Answer Relevancy).
- Enterprise AI security guardrails (Prompt injection detection & PII redaction).
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class DocumentChunk:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    chunk_hash: str = ""
    embedding: list[float] | None = None


class AgenticDocumentChunker:
    """Recursive and semantic document chunker with sliding window overlap."""

    def __init__(self, default_chunk_size: int = 150, default_overlap: int = 25):
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        c_size = chunk_size or self.default_chunk_size
        c_overlap = overlap if overlap is not None else self.default_overlap
        meta = metadata or {}

        # Normalize whitespace while preserving structural paragraph breaks
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: list[DocumentChunk] = []
        chunk_idx = 0

        for p_idx, para in enumerate(paragraphs):
            words = para.split()
            if not words:
                continue

            if len(words) <= c_size:
                c_text = " ".join(words)
                c_hash = hashlib.sha256(c_text.encode("utf-8")).hexdigest()[:16]
                chunks.append(
                    DocumentChunk(
                        id=f"{doc_id}-p{p_idx}-c{chunk_idx}",
                        content=c_text,
                        metadata={**meta, "doc_id": doc_id, "paragraph": p_idx, "chunk_index": chunk_idx},
                        token_count=len(words),
                        chunk_hash=c_hash,
                    )
                )
                chunk_idx += 1
            else:
                # Sliding window split across long paragraph
                step = max(1, c_size - c_overlap)
                for start in range(0, len(words), step):
                    window = words[start : start + c_size]
                    if not window:
                        break
                    c_text = " ".join(window)
                    c_hash = hashlib.sha256(c_text.encode("utf-8")).hexdigest()[:16]
                    chunks.append(
                        DocumentChunk(
                            id=f"{doc_id}-p{p_idx}-c{chunk_idx}",
                            content=c_text,
                            metadata={
                                **meta,
                                "doc_id": doc_id,
                                "paragraph": p_idx,
                                "window_start": start,
                                "chunk_index": chunk_idx,
                            },
                            token_count=len(window),
                            chunk_hash=c_hash,
                        )
                    )
                    chunk_idx += 1
                    if start + c_size >= len(words):
                        break

        return chunks


def compute_pseudo_embedding(text: str, dims: int = 32) -> list[float]:
    """Generates a deterministic unit-normalized dense embedding vector from token n-grams."""
    vec = [0.0] * dims
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return [1.0 / math.sqrt(dims)] * dims

    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        idx = h % dims
        sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 1e-9:
        return [round(v / norm, 6) for v in vec]
    return [1.0 / math.sqrt(dims)] * dims


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    return max(0.0, min(1.0, dot / (norm1 * norm2)))


class HybridVectorRetriever:
    """Hybrid dense vector and BM25 sparse lexical search engine with Reciprocal Rank Fusion."""

    def __init__(self, embedding_dims: int = 32):
        self.embedding_dims = embedding_dims
        self.chunks: list[DocumentChunk] = []
        self.doc_freq: dict[str, int] = {}
        self.doc_lens: list[int] = []
        self.avg_dl: float = 0.0

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        self.doc_freq.clear()
        self.doc_lens.clear()

        total_words = 0
        for chunk in self.chunks:
            if chunk.embedding is None:
                chunk.embedding = compute_pseudo_embedding(chunk.content, self.embedding_dims)

            tokens = set(re.findall(r"\w+", chunk.content.lower()))
            for t in tokens:
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

            w_len = len(re.findall(r"\w+", chunk.content.lower()))
            self.doc_lens.append(w_len)
            total_words += w_len

        self.avg_dl = (total_words / max(1, len(self.chunks))) if self.chunks else 1.0

    def dense_search(self, query_vec: list[float], top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
        if not self.chunks:
            return []
        scores = []
        for chunk in self.chunks:
            emb = chunk.embedding or compute_pseudo_embedding(chunk.content, self.embedding_dims)
            sim = cosine_similarity(query_vec, emb)
            scores.append((chunk, round(sim, 4)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def sparse_bm25_search(self, query: str, top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
        if not self.chunks:
            return []
        q_tokens = re.findall(r"\w+", query.lower())
        if not q_tokens:
            return [(c, 0.0) for c in self.chunks[:top_k]]

        n = len(self.chunks)
        k1 = 1.5
        b = 0.75
        scores = []

        for idx, chunk in enumerate(self.chunks):
            c_tokens = re.findall(r"\w+", chunk.content.lower())
            freq_map: dict[str, int] = {}
            for t in c_tokens:
                freq_map[t] = freq_map.get(t, 0) + 1

            score = 0.0
            dl = self.doc_lens[idx] if idx < len(self.doc_lens) else len(c_tokens)
            len_norm = 1.0 - b + b * (dl / max(1e-6, self.avg_dl))

            for t in q_tokens:
                if t in self.doc_freq:
                    df = self.doc_freq[t]
                    idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                    tf = freq_map.get(t, 0)
                    tf_comp = (tf * (k1 + 1.0)) / (tf + k1 * len_norm)
                    score += idf * tf_comp

            scores.append((chunk, round(max(0.0, score), 4)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def hybrid_rrf_search(
        self,
        query: str,
        query_vec: list[float] | None = None,
        top_k: int = 5,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
    ) -> list[tuple[DocumentChunk, float]]:
        """Reciprocal Rank Fusion of dense and BM25 search ranks."""
        q_vec = query_vec or compute_pseudo_embedding(query, self.embedding_dims)
        dense_hits = self.dense_search(q_vec, top_k=max(top_k * 2, len(self.chunks)))
        sparse_hits = self.sparse_bm25_search(query, top_k=max(top_k * 2, len(self.chunks)))

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, DocumentChunk] = {}

        for rank, (chunk, _) in enumerate(dense_hits, start=1):
            chunk_map[chunk.id] = chunk
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (dense_weight / (rrf_k + rank))

        sparse_weight = 1.0 - dense_weight
        for rank, (chunk, _) in enumerate(sparse_hits, start=1):
            chunk_map[chunk.id] = chunk
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (sparse_weight / (rrf_k + rank))

        combined = [(chunk_map[cid], round(score, 6)) for cid, score in rrf_scores.items()]
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

    def rerank(self, query: str, candidates: list[DocumentChunk], top_n: int = 3) -> list[tuple[DocumentChunk, float]]:
        """Cross-encoder heuristic reranker evaluating semantic overlap and exact keyword density."""
        q_tokens = set(re.findall(r"\w+", query.lower()))
        reranked: list[tuple[DocumentChunk, float]] = []

        for chunk in candidates:
            c_tokens = set(re.findall(r"\w+", chunk.content.lower()))
            overlap = len(q_tokens & c_tokens)
            recall = overlap / max(1, len(q_tokens))
            precision = overlap / max(1, len(c_tokens))
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            reranked.append((chunk, round(f1, 4)))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_n]


@dataclass
class ToolDefinition:
    name: str
    description: str
    handler: Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class AgentStep:
    agent_name: str
    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None


class MultiAgentWorkflowEngine:
    """Orchestrates ReAct and Plan-and-Execute multi-agent workflows with loop detection."""

    def __init__(self, max_iterations: int = 8):
        self.max_iterations = max_iterations
        self.tools: dict[str, ToolDefinition] = {}

    def register_tool(self, tool: ToolDefinition) -> None:
        self.tools[tool.name] = tool

    def execute_workflow(self, task: str, plan_steps: list[str]) -> dict[str, Any]:
        execution_trace: list[AgentStep] = []
        state_hashes: set[str] = set()
        status = "COMPLETED"

        for idx, step_desc in enumerate(plan_steps):
            if idx >= self.max_iterations:
                status = "HALTED_MAX_ITERATIONS"
                break

            # Action routing heuristic based on step description
            thought = f"Executing plan step {idx + 1}: {step_desc}"
            action_name = None
            action_input: dict[str, Any] = {"query": step_desc}

            if "knowledge" in step_desc.lower() or "search" in step_desc.lower() or "rag" in step_desc.lower():
                action_name = "knowledge_search"
            elif "telemetry" in step_desc.lower() or "metric" in step_desc.lower():
                action_name = "telemetry_inspect"
            elif "verify" in step_desc.lower() or "eval" in step_desc.lower():
                action_name = "compliance_eval"

            # Check for loop / recursion condition
            sig = f"{action_name}:{step_desc}"
            sig_hash = hashlib.md5(sig.encode("utf-8")).hexdigest()
            if sig_hash in state_hashes:
                status = "HALTED_LOOP_DETECTED"
                execution_trace.append(
                    AgentStep(
                        agent_name="CriticAgent",
                        thought="Loop detected: repeated identical tool call signature.",
                        action=None,
                        observation={"error": "Infinite loop aborted"},
                    )
                )
                break
            state_hashes.add(sig_hash)

            obs: dict[str, Any] = {}
            if action_name and action_name in self.tools:
                obs = self.tools[action_name].handler(action_input)
            else:
                obs = {"status": "ok", "note": "Step completed without external tool call"}

            execution_trace.append(
                AgentStep(
                    agent_name="WorkerAgent",
                    thought=thought,
                    action=action_name,
                    action_input=action_input,
                    observation=obs,
                )
            )

        return {
            "task": task,
            "status": status,
            "total_steps": len(execution_trace),
            "trace": execution_trace,
        }


@dataclass
class RAGEvalScore:
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevance: float
    passed: bool


@dataclass
class SecurityGuardrailResult:
    is_safe: bool
    injection_detected: bool
    pii_detected: bool
    detected_patterns: list[str]
    redacted_prompt: str
    redacted_response: str


class LLMJudgeEvaluator:
    """Evaluates RAG generation quality and enforces security guardrails."""

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"system\s+override",
        r"disregard\s+the\s+above",
        r"jailbreak",
        r"dan\s+mode",
        r"you\s+are\s+now\s+an\s+unrestricted",
        r"reveal\s+your\s+system\s+prompt",
        r"act\s+as\s+root",
    ]

    PII_REGEXES = {
        "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "singapore_nric": r"\b[STFGMstfgm]\d{7}[A-Za-z]\b",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "phone_number": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b",
    }

    def evaluate_rag_triplet(
        self,
        query: str,
        retrieved_contexts: list[str],
        generated_answer: str,
        ground_truth: str | None = None,
    ) -> RAGEvalScore:
        q_tokens = set(re.findall(r"\w+", query.lower()))
        ans_tokens = set(re.findall(r"\w+", generated_answer.lower()))

        # Context Precision: fraction of retrieved contexts containing key query tokens
        relevant_chunks = 0
        for ctx in retrieved_contexts:
            ctx_tokens = set(re.findall(r"\w+", ctx.lower()))
            if len(q_tokens & ctx_tokens) / max(1, len(q_tokens)) >= 0.2:
                relevant_chunks += 1
        c_precision = round(relevant_chunks / max(1, len(retrieved_contexts)), 4)

        # Context Recall: fraction of ground truth entities/tokens found in retrieved contexts
        combined_ctx = " ".join(retrieved_contexts).lower()
        ctx_all_tokens = set(re.findall(r"\w+", combined_ctx))
        target_tokens = set(re.findall(r"\w+", ground_truth.lower())) if ground_truth else q_tokens
        overlap_target = len(target_tokens & ctx_all_tokens)
        c_recall = round(overlap_target / max(1, len(target_tokens)), 4)

        # Faithfulness: fraction of generated answer claims grounded in retrieved context
        grounded_tokens = len(ans_tokens & ctx_all_tokens)
        faithfulness = round(grounded_tokens / max(1, len(ans_tokens)), 4)

        # Answer Relevance: token overlap between question and generated answer
        STOPWORDS = {"what", "which", "how", "why", "who", "when", "where", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "with", "the", "a", "an", "and", "or"}
        content_q_tokens = {t for t in q_tokens if t not in STOPWORDS} or q_tokens
        ans_relevance = round(len(content_q_tokens & ans_tokens) / max(1, len(content_q_tokens)), 4)

        passed = (
            c_precision >= 0.5
            and c_recall >= 0.5
            and faithfulness >= 0.5
            and ans_relevance >= 0.15
        )

        return RAGEvalScore(
            context_precision=c_precision,
            context_recall=c_recall,
            faithfulness=faithfulness,
            answer_relevance=ans_relevance,
            passed=passed,
        )

    def evaluate_security_guardrails(self, prompt: str, response: str) -> SecurityGuardrailResult:
        detected_patterns: list[str] = []

        # Check prompt injection
        injection_detected = False
        p_lower = prompt.lower()
        for pat in self.INJECTION_PATTERNS:
            if re.search(pat, p_lower):
                injection_detected = True
                detected_patterns.append(f"PromptInjection:{pat}")

        # Check PII
        pii_detected = False
        redacted_prompt = prompt
        redacted_response = response

        for pii_type, pat in self.PII_REGEXES.items():
            if re.search(pat, redacted_prompt):
                pii_detected = True
                detected_patterns.append(f"PII:{pii_type}")
                redacted_prompt = re.sub(pat, f"[REDACTED_{pii_type.upper()}]", redacted_prompt)

            if re.search(pat, redacted_response):
                pii_detected = True
                detected_patterns.append(f"PII:{pii_type}")
                redacted_response = re.sub(pat, f"[REDACTED_{pii_type.upper()}]", redacted_response)

        is_safe = not injection_detected and not pii_detected

        return SecurityGuardrailResult(
            is_safe=is_safe,
            injection_detected=injection_detected,
            pii_detected=pii_detected,
            detected_patterns=detected_patterns,
            redacted_prompt=redacted_prompt,
            redacted_response=redacted_response,
        )
