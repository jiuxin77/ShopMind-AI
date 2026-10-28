from __future__ import annotations

import argparse
import time
from pathlib import Path

from shopmind.data import load_demo_data
from shopmind.item_cf import build_item_cf_index

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist the ItemCF similarity index")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "uci_online_retail",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "models" / "item_cf.joblib"
    )
    parser.add_argument("--neighbors", type=int, default=100)
    args = parser.parse_args()
    _, _, interactions = load_demo_data(args.data_dir)
    started = time.perf_counter()
    index = build_item_cf_index(interactions, neighbors_per_item=args.neighbors)
    index.save(args.output)
    elapsed = time.perf_counter() - started
    print(
        f"Saved {len(index.neighbors)} item neighbor lists to {args.output} "
        f"in {elapsed:.2f}s ({args.output.stat().st_size / 1024:.1f} KiB)"
    )


if __name__ == "__main__":
    main()
