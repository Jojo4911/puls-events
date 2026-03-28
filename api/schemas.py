from datetime import datetime
from pydantic import BaseModel, Field


class SourceItem(BaseModel): # Détaille tous les sous-champs de "sources"
    title: str
    location_name: str
    location_city: str
    date_display: str
    url: str

class AskRequest(BaseModel):
    question: str = Field(min_length=1)

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    contexts: list[str]

class MetadataResponse(BaseModel):
    number_chunks: int
    vector_dimension: int
    embedding_provider: str
    llm_provider: str
    last_index_update: datetime | None = None
    api_version: str
