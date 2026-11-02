from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field

class VectorType(StrEnum):
    TABLE = "table"
    COLUMN = "column"
    GLOSSARY_KNOWLEDGE = "glossaryKnowledge"
    QUESTION_KNOWLEDGE = "questionKnowledge"

class MetadataKey(StrEnum):
    TABLE_ID = "tableId"
    COLUMN_ID = "columnId"
    KNOWLEDGE_ID = "knowledgeId"
    VECTOR_TYPE = "vectorType"
    DATABASE_ID = "databaseId"
    BUSINESS_TERM_ID = "businessTermId"

class RetrievalResult(BaseModel):
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
