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
