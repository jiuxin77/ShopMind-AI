import pandas as pd

from shopmind.ranking import build_ranking_context, candidate_features


def test_candidate_features_extract_source_ranks_and_price_distance():
    products = pd.DataFrame(
        [
            {"product_id": "P1", "price": 10.0},
            {"product_id": "P2", "price": 20.0},
        ]
    )
    interactions = pd.DataFrame(
        [
            {"user_id": 1001, "product_id": "P1", "event_type": "purchase"},
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "product_id": "P2",
                "price": 20.0,
                "score": 0.2,
                "evidence": "item_cf#2, text_vector#4",
            }
        ]
    )
    context = build_ranking_context(products, interactions)
    features = candidate_features(candidates, 1001, context).iloc[0]
    assert features["item_cf_rr"] == 0.5
    assert features["text_vector_rr"] == 0.25
    assert features["price_distance"] == 10.0
    assert features["evidence_count"] == 2.0
