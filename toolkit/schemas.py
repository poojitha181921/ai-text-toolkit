from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ProcessingMetadata(BaseModel):
    model_name: str
    latency_ms: float
    input_token_count: Optional[int] = None
    output_token_count: Optional[int] = None

class SummaryData(BaseModel):
    summary_text: str
    original_char_count: int
    summary_char_count: int
    compression_ratio: float = Field(
        ..., description="Ratio of summary length to original text length"
    )

class ClassificationScore(BaseModel):
    label: str
    score: float = Field(..., description="Confidence probability between 0.0 and 1.0")

class ClassificationData(BaseModel):
    text: str
    top_label: str
    top_score: float
    all_scores: List[ClassificationScore]
    multi_label: bool

class EntityItem(BaseModel):
    text: str
    label: str
    score: float
    start: int
    end: int

class ExtractionData(BaseModel):
    text: str
    entities: List[EntityItem]
    grouped_entities: Dict[str, List[str]]

class ToolkitResponse(BaseModel):
    task: str
    status: str = "success"
    data: Dict[str, Any]
    metadata: ProcessingMetadata

class RetrievedChunk(BaseModel):
    chunk_id: int
    content: str
    similarity_score: float

class RAGData(BaseModel):
    query: str
    answer: str
    context_chunks: List[RetrievedChunk]
    grounded_prompt: str    