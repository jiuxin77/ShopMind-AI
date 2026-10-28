import pandas as pd

from shopmind.recommender import popular_recommend


def test_recommendation_respects_budget_and_stock():
    products = pd.DataFrame([
        {"product_id": "P1", "title": "A", "category": "数码", "price": 100, "rating": 5, "in_stock": True, "active": True},
        {"product_id": "P2", "title": "B", "category": "数码", "price": 500, "rating": 5, "in_stock": True, "active": True},
        {"product_id": "P3", "title": "C", "category": "数码", "price": 80, "rating": 5, "in_stock": False, "active": True},
    ])
    interactions = pd.DataFrame({"product_id": ["P1", "P2", "P3"], "event_type": ["click", "purchase", "purchase"]})
    result = popular_recommend(products, interactions, category="数码", budget_max=200)
    assert result["product_id"].tolist() == ["P1"]

