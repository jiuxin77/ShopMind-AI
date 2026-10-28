import pandas as pd

from shopmind.shopping_assistant import parse_shopping_intent, recommend_from_need


def test_parse_chinese_shopping_intent():
    intent = parse_shopping_intent("想买通勤用的简约女装，预算500元以内，不要星野品牌")
    assert intent.category == "女装"
    assert intent.budget_max == 500
    assert intent.scenes == ["通勤"]
    assert intent.styles == ["简约"]
    assert intent.exclusions == ["星野"]


def test_recommendation_applies_budget_and_exclusion():
    products = pd.DataFrame(
        [
            {
                "product_id": "P1",
                "title": "简约通勤衬衫",
                "category": "女装",
                "brand": "青禾",
                "price": 399.0,
                "rating": 4.8,
                "in_stock": True,
                "active": True,
            },
            {
                "product_id": "P2",
                "title": "简约通勤衬衫",
                "category": "女装",
                "brand": "星野",
                "price": 299.0,
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
    _, result = recommend_from_need(
        "简约通勤女装，500元以内，不要星野",
        products,
        interactions,
    )
    assert result["product_id"].tolist() == ["P1"]
    assert "未超过 ¥500 预算" in result.iloc[0]["reason"]
