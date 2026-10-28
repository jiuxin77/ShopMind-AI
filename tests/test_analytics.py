import pandas as pd

from shopmind.analytics import behavior_funnel, price_sensitivity


def test_behavior_funnel_has_stable_event_order():
    interactions = pd.DataFrame({"event_type": ["purchase", "view", "view", "click"]})
    result = behavior_funnel(interactions)
    assert result["event_type"].tolist() == ["view", "click", "favorite", "cart", "purchase", "refund"]
    assert result.loc[result.event_type == "view", "count"].item() == 2


def test_price_sensitivity_counts_purchase():
    products = pd.DataFrame({"product_id": ["P1"], "price": [99.0]})
    interactions = pd.DataFrame({"product_id": ["P1", "P1"], "event_type": ["view", "purchase"]})
    result = price_sensitivity(interactions, products)
    assert result.iloc[0]["purchases"] == 1
    assert result.iloc[0]["purchase_rate"] == 0.5

