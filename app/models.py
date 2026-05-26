from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime


class JobStatus(str, Enum):
    """Job processing status enumeration."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(BaseModel):
    """Represents an asynchronous analysis job."""
    id: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    total_reviews: int
    processed_batches: Optional[int] = None
    results: Optional[List["ReviewAnalysis"]] = None
    error: Optional[str] = None
    reviews: Optional[List] = None  # Input reviews (not included in response)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobResponse(BaseModel):
    """Response model for job status queries."""
    id: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    total_reviews: int
    processed_batches: Optional[int] = None
    results: Optional[List["ReviewAnalysis"]] = None
    error: Optional[str] = None


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


# Update forward references for Pydantic models
Job.model_rebuild()
JobResponse.model_rebuild()