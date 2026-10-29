from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

from shopmind.data import load_demo_data
from shopmind.shopping_assistant import recommend_from_need
from shopmind.visual_search import build_visual_index, visual_search

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    required = [
        ROOT / "requirements.txt",
        ROOT / "runtime.txt",
        ROOT / ".streamlit" / "config.toml",
        ROOT / "src" / "shopmind" / "dashboard.py",
        ROOT / "data" / "users.csv",
        ROOT / "data" / "products.csv",
        ROOT / "data" / "interactions.csv",
    ]
    files = {str(path.relative_to(ROOT)): path.exists() for path in required}
    users, products, interactions = load_demo_data()
    intent, recommendations = recommend_from_need(
        "通勤简约女装，预算500元以内，不要星野",
        products,
        interactions,
        top_k=3,
    )
    visual_products = products.head(10).copy()
    image_dir = ROOT / "data" / "generated_product_images"
    visual_index = build_visual_index(visual_products, image_dir)
    visual_result = visual_search(
        Image.open(visual_index.image_paths[0]), visual_products, visual_index, top_k=3
    )
    checks = {
        "required_files": all(files.values()),
        "data_loaded": len(users) > 0 and len(products) > 0 and len(interactions) > 0,
        "shopping_assistant": intent.category == "女装" and not recommendations.empty,
        "visual_search": not visual_result.empty
        and visual_result.iloc[0]["product_id"] == visual_products.iloc[0]["product_id"],
    }
    output = {
        "release": "M11",
        "python": sys.version.split()[0],
        "passed": all(checks.values()),
        "checks": checks,
        "files": files,
        "demo_rows": {
            "users": len(users),
            "products": len(products),
            "interactions": len(interactions),
        },
    }
    (ROOT / "artifacts" / "m11_deployment_check.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
