from fastapi import FastAPI
from .models import EvalCase, EvalSuite
from .evals import evaluate_case, evaluate_suite, provider_gateway
from pydantic import BaseModel

class SuiteRequest(BaseModel):
    cases: list[EvalCase]
    provider: str = "mock-maia"

app = FastAPI(title="GenAI Tooling Evals")

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
