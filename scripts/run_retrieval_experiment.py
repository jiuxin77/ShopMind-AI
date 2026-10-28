from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopmind.data import load_demo_data
from shopmind.evaluation import compare_retrievers, evaluate_with_details

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ShopMind retrieval models")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "uci_online_retail",
    )
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "m3_model_comparison.json"
    )
    args = parser.parse_args()
    _, products, interactions = load_demo_data(args.data_dir)
    comparison = compare_retrievers(products, interactions, k=args.k, max_users=args.users)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = comparison.to_dict("records")
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    _, details = evaluate_with_details(
        products,
        interactions,
        k=args.k,
        max_users=args.users,
        sources={"popular", "item_cf", "bm25", "text_vector"},
    )
    bad_case_path = args.output.with_name("m4_bad_cases.csv")
    details.to_csv(bad_case_path, index=False)
    print(comparison.to_string(index=False))
    print(f"Saved evaluation details to {bad_case_path}")


if __name__ == "__main__":
    main()
