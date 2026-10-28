from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopmind.data import load_demo_data
from shopmind.evaluation import compare_retrievers

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
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
