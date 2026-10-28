from fastapi import FastAPI, HTTPException

from shopmind.data import load_demo_data
from shopmind.retrieval import hybrid_recommend
from shopmind.schemas import RecommendationItem, RecommendationRequest

app = FastAPI(title="ShopMind AI", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


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
