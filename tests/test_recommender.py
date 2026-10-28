import pandas as pd

from shopmind.recommender import popular_recommend
from shopmind.retrieval import bm25_recall, hybrid_recommend, reciprocal_rank_fusion


def test_recommendation_respects_budget_and_stock():
    products = pd.DataFrame(
        [
            {
                "product_id": "P1",
                "title": "A",
                "category": "数码",
                "price": 100,
                "rating": 5,
                "in_stock": True,
                "active": True,
            },
            {
                "product_id": "P2",
                "title": "B",
                "category": "数码",
                "price": 500,
                "rating": 5,
                "in_stock": True,
                "active": True,
            },
            {
                "product_id": "P3",
                "title": "C",
                "category": "数码",
                "price": 80,
                "rating": 5,
                "in_stock": False,
                "active": True,
            },
        ]
    )
    interactions = pd.DataFrame(
        {"product_id": ["P1", "P2", "P3"], "event_type": ["click", "purchase", "purchase"]}
    )
    result = popular_recommend(products, interactions, category="数码", budget_max=200)
    assert result["product_id"].tolist() == ["P1"]


def test_bm25_finds_matching_chinese_title():
    products = pd.DataFrame(
        [
            {"product_id": "P1", "title": "通勤衬衫", "category": "女装", "brand": "青禾"},
            {"product_id": "P2", "title": "无线耳机", "category": "数码", "brand": "星野"},
        ]
    )
    result = bm25_recall("通勤女装", products)
    assert result.iloc[0]["product_id"] == "P1"


def test_rrf_combines_source_evidence():
    first = pd.DataFrame([{"product_id": "P1", "raw_score": 2.0, "source": "a", "rank": 1}])
    second = pd.DataFrame([{"product_id": "P1", "raw_score": 1.0, "source": "b", "rank": 1}])
    result = reciprocal_rank_fusion([first, second])
    assert result.iloc[0]["product_id"] == "P1"
    assert result.iloc[0]["evidence"] == "a#1, b#1"


def test_hybrid_recommend_respects_business_filters():
    products = pd.DataFrame(
        [
            {
                "product_id": "P1",
                "title": "通勤衬衫",
                "category": "女装",
                "brand": "青禾",
                "price": 199,
                "rating": 4.8,
                "in_stock": True,
                "active": True,
            },
            {
                "product_id": "P2",
                "title": "通勤外套",
                "category": "女装",
                "brand": "星野",
                "price": 999,
                "rating": 4.9,
                "in_stock": True,
                "active": True,
            },
        ]
    )
    interactions = pd.DataFrame(
        {
            "user_id": ["U1", "U2"],
            "product_id": ["P1", "P2"],
            "event_type": ["purchase", "purchase"],
        }
    )
    result = hybrid_recommend(products, interactions, query="通勤", budget_max=300)
    assert result["product_id"].tolist() == ["P1"]
