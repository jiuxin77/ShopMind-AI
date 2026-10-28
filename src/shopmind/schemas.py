from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: str | None = None
    query: str = ""
    category: str | None = None
    budget_max: float | None = Field(default=None, gt=0)
    scene: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)


class RecommendationItem(BaseModel):
    product_id: str
    title: str
    category: str
    price: float
    score: float
    reason: str
    evidence: str


class ShoppingAssistantRequest(BaseModel):
    need: str = Field(min_length=1, max_length=500)
    user_id: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)


class ShoppingAssistantResponse(BaseModel):
    intent: dict[str, object]
    recommendations: list[RecommendationItem]
