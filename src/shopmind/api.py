from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from shopmind.config import get_settings
from shopmind.data import load_demo_data
from shopmind.llm_shopping_agent import AgentResult, create_openai_agent
from shopmind.observability import configure_logging
from shopmind.retrieval import hybrid_recommend
from shopmind.schemas import (
    LLMAgentRequest,
    RecommendationItem,
    RecommendationRequest,
    ShoppingAssistantRequest,
    ShoppingAssistantResponse,
)
from shopmind.shopping_assistant import recommend_from_need

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("shopmind.api")
app = FastAPI(title=settings.app_name, version="0.2.0")


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "unhandled request failure",
            extra={"request_id": request_id, "method": request.method, "path": request.url.path},
        )
        response = JSONResponse(
            status_code=500,
            content={"detail": "internal service error", "request_id": request_id},
        )
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


@app.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
def readiness() -> dict[str, object]:
    try:
        users, products, interactions = load_demo_data()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ready",
        "data": {"users": len(users), "products": len(products), "interactions": len(interactions)},
        "llm_configured": settings.openai_api_key is not None,
    }


@app.post("/recommend", response_model=list[RecommendationItem])
def recommend(request: RecommendationRequest) -> list[RecommendationItem]:
    try:
        _, products, interactions = load_demo_data()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    result = hybrid_recommend(
        products=products,
        interactions=interactions,
        user_id=request.user_id,
        query=request.query,
        category=request.category,
        budget_max=request.budget_max,
        top_k=request.top_k,
    )
    return [
        RecommendationItem(**row)
        for row in result[
            ["product_id", "title", "category", "price", "score", "reason", "evidence"]
        ].to_dict("records")
    ]


@app.post("/shopping-assistant", response_model=ShoppingAssistantResponse)
def shopping_assistant(request: ShoppingAssistantRequest) -> ShoppingAssistantResponse:
    try:
        _, products, interactions = load_demo_data()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    intent, result = recommend_from_need(
        request.need,
        products,
        interactions,
        user_id=request.user_id,
        top_k=request.top_k,
    )
    items = [
        RecommendationItem(**row)
        for row in result[
            ["product_id", "title", "category", "price", "score", "reason", "evidence"]
        ].to_dict("records")
    ]
    return ShoppingAssistantResponse(intent=intent.to_dict(), recommendations=items)


@app.post("/llm-shopping-agent", response_model=AgentResult)
def llm_shopping_agent(request: LLMAgentRequest) -> AgentResult:
    if settings.openai_api_key is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured")
    try:
        _, products, interactions = load_demo_data()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    agent = create_openai_agent(
        products,
        interactions,
        settings.openai_api_key.get_secret_value(),
        request.model or settings.openai_model,
    )
    return agent.run(request.need, user_id=request.user_id)
