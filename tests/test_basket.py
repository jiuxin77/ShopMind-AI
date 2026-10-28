import pandas as pd

from shopmind.basket import build_basket_index


def test_basket_index_finds_frequently_copurchased_item():
    interactions = pd.DataFrame(
        {
            "session_id": ["S1", "S1", "S2", "S2", "S3"],
            "product_id": ["P1", "P2", "P1", "P2", "P3"],
        }
    )
    index = build_basket_index(interactions)
    result = index.recommend({"P1"}, top_k=1)
    assert result.iloc[0]["product_id"] == "P2"
