from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopmind.data import load_demo_data
from shopmind.evaluation import evaluate_with_details, tune_fusion_weights

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune RRF weights on held-out users")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "uci_online_retail",
    )
    parser.add_argument("--validation-users", type=int, default=100)
    parser.add_argument("--test-users", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    _, products, interactions = load_demo_data(args.data_dir)
    best_weights, tuning = tune_fusion_weights(
        products, interactions, validation_users=args.validation_users, k=args.k
    )
    metrics, details = evaluate_with_details(
        products,
        interactions,
        k=args.k,
        max_users=args.test_users,
        user_offset=args.validation_users,
        sources=set(best_weights),
        source_weights=best_weights,
    )
    output = {
        "validation_users": args.validation_users,
        "test_users": int(metrics["users"]),
        "best_weights": best_weights,
        "test_metrics": metrics,
        "tuning_results": tuning.to_dict("records"),
    }
    artifact_dir = ROOT / "artifacts"
    (artifact_dir / "m5_weight_tuning.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    details.to_csv(artifact_dir / "m5_bad_cases.csv", index=False)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
