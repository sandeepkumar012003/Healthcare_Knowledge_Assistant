from typing import List, Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):

    query: str

    mode: str = "general"


class QueryResponse(BaseModel):

    answer: str

    sources: List[str]

    diagnosis: Optional[list] = None

    eval_scores: Optional[dict] = None
