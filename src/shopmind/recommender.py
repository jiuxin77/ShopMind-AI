from __future__ import annotations

import pandas as pd

EVENT_WEIGHT = {
    "view": 1.0,
    "click": 2.0,
    "favorite": 3.0,
    "cart": 4.0,
    "purchase": 6.0,
    "refund": -4.0,
}


def popular_recommend(
    products: pd.DataFrame,
    interactions: pd.DataFrame,
    category: str | None = None,
    budget_max: float | None = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """Deterministic popularity baseline with business-rule filtering."""
    candidates = products.loc[(products["in_stock"] == True) & (products["active"] == True)].copy()
    if category:
        candidates = candidates[candidates["category"] == category]
    if budget_max is not None:
        candidates = candidates[candidates["price"] <= budget_max]

    events = interactions.copy()
    events["event_weight"] = events["event_type"].map(EVENT_WEIGHT).fillna(0.0)
    scores = events.groupby("product_id", as_index=False)["event_weight"].sum()
    result = candidates.merge(scores, on="product_id", how="left").fillna({"event_weight": 0.0})
    result["score"] = result["event_weight"] + result["rating"] * 0.5
    result["reason"] = "符合库存与预算约束，并基于历史互动热度排序"
    return result.sort_values(["score", "rating"], ascending=False).head(top_k)
