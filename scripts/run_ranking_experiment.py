from __future__ import annotations

import json
from pathlib import Path

from shopmind.data import load_demo_data
from shopmind.evaluation import evaluate_with_details, leave_one_out_split
from shopmind.item_cf import build_item_cf_index
from shopmind.ranking import evaluate_ranker, train_pointwise_ranker

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {
    "popular": 1.0,
    "item_cf": 1.0,
    "bm25": 1.0,
    "text_vector": 2.0,
    "recent_interest": 0.25,
}


def main() -> None:
    _, products, interactions = load_demo_data(ROOT / "data" / "processed" / "uci_online_retail")
    train, targets = leave_one_out_split(interactions)
    validation_targets = targets.iloc[:100]
    test_targets = targets.iloc[100:300]
    item_cf_index = build_item_cf_index(train)
    model, training = train_pointwise_ranker(
        products,
        train,
        validation_targets,
        item_cf_index,
        WEIGHTS,
    )
    baseline, _ = evaluate_with_details(
        products,
        interactions,
        k=10,
        max_users=200,
        user_offset=100,
        candidate_k=50,
        sources=set(WEIGHTS),
        source_weights=WEIGHTS,
    )
    reranked, details = evaluate_ranker(
        model,
        products,
        train,
        test_targets,
        item_cf_index,
        WEIGHTS,
        model_weight=0.1,
    )
    output = {
        "validation_slice": "users[0:100]",
        "test_slice": "users[100:300]",
        "training": training,
        "ranking_strategy": "90% reciprocal RRF rank + 10% reciprocal model rank",
        "features": list(model.feature_names_in_),
        "results": [
            {"model": "m6_weighted_rrf", **baseline},
            {"model": "m7_logistic_reranker", **reranked},
        ],
    }
    artifact_dir = ROOT / "artifacts"
    (artifact_dir / "m7_ranking_experiment.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    details.to_csv(artifact_dir / "m7_ranking_details.csv", index=False)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
