from __future__ import annotations

import json
from pathlib import Path

from shopmind.data import load_demo_data
from shopmind.shopping_assistant import parse_shopping_intent, recommend_from_need

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    (
        "想买通勤用的简约女装，预算500元以内，不要星野品牌",
        {
            "category": "女装",
            "budget_min": None,
            "budget_max": 500.0,
            "scenes": ["通勤"],
            "styles": ["简约"],
            "exclusions": ["星野"],
        },
    ),
    (
        "预算800以内的数码产品，旅行便携，不要北岸",
        {
            "category": "数码",
            "budget_min": None,
            "budget_max": 800.0,
            "scenes": ["旅行"],
            "styles": ["便携"],
            "exclusions": ["北岸"],
        },
    ),
    (
        "至少300元的简约家居，适合居家",
        {
            "category": "家居",
            "budget_min": 300.0,
            "budget_max": None,
            "scenes": ["居家"],
            "styles": ["简约"],
            "exclusions": [],
        },
    ),
    (
        "清爽温和的护肤品，预算600",
        {
            "category": "护肤",
            "budget_min": None,
            "budget_max": 600.0,
            "scenes": [],
            "styles": ["清爽", "温和"],
            "exclusions": [],
        },
    ),
    (
        "送礼食品，最多400，不考虑云间",
        {
            "category": "食品",
            "budget_min": None,
            "budget_max": 400.0,
            "scenes": ["送礼"],
            "styles": [],
            "exclusions": ["云间"],
        },
    ),
    (
        "办公用商务女装，不超过700",
        {
            "category": "女装",
            "budget_min": None,
            "budget_max": 700.0,
            "scenes": ["办公"],
            "styles": ["商务"],
            "exclusions": [],
        },
    ),
]
FIELDS = ("category", "budget_min", "budget_max", "scenes", "styles", "exclusions")


def main() -> None:
    _, products, interactions = load_demo_data()
    slot_correct = 0
    constraint_passes = 0
    rows = []
    for query, expected in CASES:
        intent = parse_shopping_intent(query)
        actual = intent.to_dict()
        field_results = {field: actual[field] == expected[field] for field in FIELDS}
        slot_correct += sum(field_results.values())
        _, recommendations = recommend_from_need(query, products, interactions, top_k=10)
        constraint_ok = not recommendations.empty
        if expected["budget_min"] is not None:
            constraint_ok &= bool((recommendations["price"] >= expected["budget_min"]).all())
        if expected["budget_max"] is not None:
            constraint_ok &= bool((recommendations["price"] <= expected["budget_max"]).all())
        if expected["category"] is not None:
            constraint_ok &= bool((recommendations["category"] == expected["category"]).all())
        for excluded in expected["exclusions"]:
            constraint_ok &= not bool(
                recommendations[["title", "brand", "category"]]
                .astype(str)
                .apply(lambda column, term=excluded: column.str.contains(term).any())
                .any()
            )
        constraint_passes += int(constraint_ok)
        rows.append(
            {
                "query": query,
                "field_results": field_results,
                "recommendation_count": len(recommendations),
                "constraint_pass": bool(constraint_ok),
            }
        )
    output = {
        "cases": len(CASES),
        "slots": len(CASES) * len(FIELDS),
        "slot_accuracy": slot_correct / (len(CASES) * len(FIELDS)),
        "constraint_satisfaction_rate": constraint_passes / len(CASES),
        "details": rows,
    }
    (ROOT / "artifacts" / "m8_shopping_assistant_eval.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
