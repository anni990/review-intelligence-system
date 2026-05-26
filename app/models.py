from pydantic import BaseModel, Field
from typing import List


class Issue(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class Intent(BaseModel):
    service_event: str
    customer_experience: str
    emotion: str


class ReviewAnalysis(BaseModel):
    review_text: str
    rating: int

    issues: List[Issue]

    intent: Intent

    severity: str

    root_cause: str

    recommended_actions: List[str]

    overall_confidence: float = Field(
        ge=0,
        le=1
    )


class BatchResponse(BaseModel):
    reviews: List[ReviewAnalysis]


class FinalResponse(BaseModel):
    total_reviews: int
    processed_batches: int
    results: List[ReviewAnalysis]