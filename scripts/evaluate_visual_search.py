from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from shopmind.data import load_demo_data
from shopmind.visual_search import brightness_variant, build_visual_index, visual_search

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    _, products, _ = load_demo_data()
    products = products.head(100).copy()
    image_dir = ROOT / "data" / "generated_product_images"
    index = build_visual_index(products, image_dir)
    top1 = 0
    top5 = 0
    cases = []
    factors = (0.72, 0.86, 1.14, 1.28)
    for position, row in products.head(50).reset_index(drop=True).iterrows():
        query = brightness_variant(Image.open(index.image_paths[position]), factors[position % 4])
        result = visual_search(query, products, index, top_k=5)
        ranking = result["product_id"].astype(str).tolist()
        target = str(row["product_id"])
        rank = ranking.index(target) + 1 if target in ranking else None
        top1 += int(rank == 1)
        top5 += int(rank is not None and rank <= 5)
        cases.append(
            {"product_id": target, "rank": rank, "brightness_factor": factors[position % 4]}
        )
    output = {
        "image_source": "deterministic synthetic demo thumbnails",
        "evaluation": "brightness perturbation retrieval",
        "cases": len(cases),
        "top1_accuracy": top1 / len(cases),
        "top5_accuracy": top5 / len(cases),
        "details": cases,
    }
    (ROOT / "artifacts" / "m9_visual_search_eval.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
