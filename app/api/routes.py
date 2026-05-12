from fastapi import APIRouter

from app.schemas.models import (
    QueryRequest,
    QueryResponse
)

from app.agents.crew import run_crew

router = APIRouter()


@router.get("/health")
def health():

    return {
        "status": "ok"
    }


@router.post(
    "/query",
    response_model=QueryResponse
)
def query_rag(request: QueryRequest):

    result = run_crew(
        request.query
    )

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        diagnosis=None,
        eval_scores={}
    )
