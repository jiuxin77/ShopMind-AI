from __future__ import annotations

import json
from pathlib import Path

from shopmind.data import load_demo_data
from shopmind.evaluation import evaluate_with_details

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data_dir = ROOT / "data" / "processed" / "uci_online_retail"
    _, products, interactions = load_demo_data(data_dir)
    base_weights = {"popular": 1.0, "item_cf": 1.0, "bm25": 1.0, "text_vector": 2.0}
    configurations = {
        "m5_weighted": base_weights,
        "m6_recent_0_25": {**base_weights, "recent_interest": 0.25},
        "m6_basket_0_25": {**base_weights, "basket": 0.25},
        "m6_both_0_10": {**base_weights, "recent_interest": 0.1, "basket": 0.1},
        "m6_both_0_25": {**base_weights, "recent_interest": 0.25, "basket": 0.25},
        "m6_both_0_50": {**base_weights, "recent_interest": 0.5, "basket": 0.5},
        "m6_both_1_00": {**base_weights, "recent_interest": 1.0, "basket": 1.0},
    }
    rows = []
    for name, weights in configurations.items():
        metrics, details = evaluate_with_details(
            products,
            interactions,
            k=10,
            max_users=200,
            user_offset=100,
            candidate_k=50,
            sources=set(weights),
            source_weights=weights,
        )
        rows.append({"model": name, "weights": weights, **metrics})
        details.to_csv(ROOT / "artifacts" / f"{name}_details.csv", index=False)
    best = max(rows, key=lambda row: (row["candidate_recall"], row["hit_rate_at_k"]))
    output = {
        "test_slice": "users[100:300]",
        "selection_metric": "candidate_recall, then hit_rate_at_k",
        "best_model": best["model"],
        "results": rows,
    }
    (ROOT / "artifacts" / "m6_recall_expansion.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
