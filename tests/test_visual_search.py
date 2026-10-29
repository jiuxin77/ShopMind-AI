import pandas as pd

from shopmind.visual_search import build_visual_index, generate_demo_image, visual_search


def test_visual_search_returns_identical_demo_image_first(tmp_path):
    products = pd.DataFrame(
        [
            {"product_id": "P1", "title": "衬衫", "category": "女装", "brand": "A"},
            {"product_id": "P2", "title": "耳机", "category": "数码", "brand": "B"},
        ]
    )
    index = build_visual_index(products, tmp_path)
    query = generate_demo_image(products.iloc[0])
    result = visual_search(query, products, index, top_k=2)
    assert result.iloc[0]["product_id"] == "P1"
    assert result.iloc[0]["visual_score"] > 0.99
