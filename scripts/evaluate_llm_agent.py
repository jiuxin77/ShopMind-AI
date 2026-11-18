from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from shopmind.data import load_demo_data
from shopmind.llm_shopping_agent import create_openai_agent

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the live ShopMind LLM agent.")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    args = parser.parse_args()
    if not args.api_key:
        parser.error("set OPENAI_API_KEY or pass --api-key")

    users, products, interactions = load_demo_data()
    del users
    cases = json.loads((ROOT / "data" / "llm_agent_eval_cases.json").read_text())
    agent = create_openai_agent(products, interactions, args.api_key, args.model)
    details = []
    for case in cases:
        started = time.perf_counter()
        result = agent.run(case["query"])
        latency_ms = (time.perf_counter() - started) * 1000
        first_tool = result.tool_trace[0]["tool"] if result.tool_trace else None
        tool_correct = first_tool == case["expected_tool"]
        allowed = set()
        for event in result.tool_trace:
            allowed.update(event.get("allowed_product_ids", []))
        returned = {item.product_id for item in result.recommendations}
        grounded = returned <= allowed if returned else True
        constraint_ok = True
        if result.recommendations and case.get("budget_max") is not None:
            selected = products[products.product_id.astype(str).isin(returned)]
            constraint_ok &= bool((selected.price <= case["budget_max"]).all())
        if result.recommendations and case.get("expected_category"):
            selected = products[products.product_id.astype(str).isin(returned)]
            constraint_ok &= bool((selected.category == case["expected_category"]).all())
        details.append(
            {
                "id": case["id"],
                "status": result.status,
                "tool_correct": tool_correct,
                "grounded": grounded,
                "constraint_ok": constraint_ok,
                "latency_ms": round(latency_ms, 1),
                "error": result.error,
            }
        )

    count = len(details)
    report = {
        "model": args.model,
        "cases": count,
        "tool_selection_accuracy": sum(row["tool_correct"] for row in details) / count,
        "grounded_product_rate": sum(row["grounded"] for row in details) / count,
        "constraint_satisfaction_rate": sum(row["constraint_ok"] for row in details) / count,
        "fallback_rate": sum(row["status"] == "fallback" for row in details) / count,
        "average_latency_ms": round(sum(row["latency_ms"] for row in details) / count, 1),
        "details": details,
    }
    output = ROOT / "artifacts" / "m12_llm_agent_eval.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
