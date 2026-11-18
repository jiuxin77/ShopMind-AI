import json
from types import SimpleNamespace

import pandas as pd

from shopmind.llm_shopping_agent import GroundedAnswer, LLMShoppingAgent


def _data():
    products = pd.DataFrame(
        [
            {
                "product_id": "P1",
                "title": "简约通勤衬衫",
                "category": "女装",
                "brand": "青禾",
                "price": 399.0,
                "rating": 4.8,
                "in_stock": True,
                "active": True,
            }
        ]
    )
    interactions = pd.DataFrame(
        {"user_id": ["U1"], "product_id": ["P1"], "event_type": ["purchase"]}
    )
    return products, interactions


class FakeResponses:
    def __init__(self, tool_name="search_catalog", fail=False):
        self.tool_name = tool_name
        self.fail = fail

    def create(self, **_kwargs):
        if self.fail:
            raise TimeoutError("model timeout")
        arguments = (
            {"question": "你的预算是多少？", "missing_fields": ["budget"]}
            if self.tool_name == "ask_clarification"
            else {
                "query": "简约通勤女装",
                "category": "女装",
                "budget_max": 500,
                "excluded_terms": [],
                "top_k": 5,
            }
        )
        call = SimpleNamespace(
            type="function_call",
            name=self.tool_name,
            arguments=json.dumps(arguments, ensure_ascii=False),
            call_id="call_1",
        )
        return SimpleNamespace(id="resp_1", output=[call])

    def parse(self, **_kwargs):
        parsed = GroundedAnswer.model_validate(
            {
                "answer": "推荐青禾衬衫，价格在预算内。",
                "recommendations": [
                    {"product_id": "P1", "reason": "适合通勤", "evidence": ["价格 ¥399"]},
                    {"product_id": "MADE_UP", "reason": "虚构", "evidence": []},
                ],
            }
        )
        return SimpleNamespace(output_parsed=parsed)


def test_agent_calls_search_and_drops_unsupported_products():
    products, interactions = _data()
    agent = LLMShoppingAgent(
        products, interactions, client=SimpleNamespace(responses=FakeResponses())
    )
    result = agent.run("500元以内的简约通勤女装")
    assert result.status == "completed"
    assert [item.product_id for item in result.recommendations] == ["P1"]
    assert result.tool_trace[0]["tool"] == "search_catalog"
    assert result.tool_trace[-1]["dropped_unsupported"] == 1


def test_agent_can_ask_a_clarifying_question():
    products, interactions = _data()
    agent = LLMShoppingAgent(
        products,
        interactions,
        client=SimpleNamespace(responses=FakeResponses("ask_clarification")),
    )
    result = agent.run("帮我买个东西")
    assert result.status == "needs_clarification"
    assert result.clarification_question == "你的预算是多少？"


def test_agent_falls_back_when_model_is_unavailable():
    products, interactions = _data()
    agent = LLMShoppingAgent(
        products,
        interactions,
        client=SimpleNamespace(responses=FakeResponses(fail=True)),
    )
    result = agent.run("500元以内的简约通勤女装")
    assert result.status == "fallback"
    assert result.recommendations[0].product_id == "P1"
    assert "TimeoutError" in result.error
