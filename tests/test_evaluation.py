import pandas as pd

from shopmind.evaluation import leave_one_out_split


def test_leave_one_out_uses_latest_positive_event():
    interactions = pd.DataFrame(
        [
            {
                "user_id": "U1",
                "product_id": "P1",
                "event_type": "click",
                "timestamp": "2026-01-01T00:00:00Z",
            },
            {
                "user_id": "U1",
                "product_id": "P2",
                "event_type": "purchase",
                "timestamp": "2026-01-02T00:00:00Z",
            },
            {
                "user_id": "U2",
                "product_id": "P3",
                "event_type": "view",
                "timestamp": "2026-01-03T00:00:00Z",
            },
        ]
    )
    train, test = leave_one_out_split(interactions)
    assert test["product_id"].tolist() == ["P2"]
    assert "P1" in train["product_id"].tolist()
