from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from shopmind.retrieval import hybrid_recommend
from shopmind.shopping_assistant import recommend_from_need


class ProductExplanation(BaseModel):
    product_id: str
    reason: str = Field(description="Why this retrieved product fits the user's need")
    evidence: list[str] = Field(description="Only facts present in the tool result")


class GroundedAnswer(BaseModel):
    answer: str
    recommendations: list[ProductExplanation]


class AgentResult(BaseModel):
    status: Literal["completed", "needs_clarification", "fallback", "error"]
    answer: str
    clarification_question: str | None = None
    recommendations: list[ProductExplanation] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    model: str | None = None
    error: str | None = None


SEARCH_TOOL = {
    "type": "function",
    "name": "search_catalog",
    "description": "Search the product catalog with explicit shopping constraints.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "category": {"type": ["string", "null"]},
            "budget_max": {"type": ["number", "null"]},
            "excluded_terms": {"type": "array", "items": {"type": "string"}},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query", "category", "budget_max", "excluded_terms", "top_k"],
        "additionalProperties": False,
    },
    "strict": True,
}

CLARIFY_TOOL = {
    "type": "function",
    "name": "ask_clarification",
    "description": "Ask one concise question when a key constraint is missing or ambiguous.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "missing_fields": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["question", "missing_fields"],
        "additionalProperties": False,
    },
    "strict": True,
}

SYSTEM_PROMPT = """You are ShopMind's shopping agent. Choose exactly one tool first.
Call search_catalog whenever the request contains a product category, product type, use case,
budget, style, or exclusion that can produce a useful catalog search. A category plus a budget
is always sufficient, even when the exact product subtype is unspecified. Use ask_clarification
only for genuinely underspecified requests where searching would be arbitrary, such as an item
with no category, recipient, use case, preference, or budget. Otherwise call search_catalog.
Never invent products, prices, brands, ratings, or evidence. After search results are returned,
recommend only returned product_ids and cite concrete facts from the tool output. Answer in the
user's language."""


@dataclass
class LLMShoppingAgent:
    products: pd.DataFrame
    interactions: pd.DataFrame
    client: Any
    model: str = "gpt-4.1-mini"

    def _search_catalog(self, arguments: dict[str, Any], user_id: object | None) -> pd.DataFrame:
        candidates = hybrid_recommend(
            self.products,
            self.interactions,
            user_id=user_id,
            query=str(arguments["query"]),
            category=arguments.get("category"),
            budget_max=arguments.get("budget_max"),
            top_k=max(int(arguments.get("top_k", 5)) * 5, 25),
        )
        excluded = [str(value).lower() for value in arguments.get("excluded_terms", [])]
        if excluded and not candidates.empty:
            searchable = (
                candidates["title"].fillna("").astype(str)
                + " "
                + candidates["brand"].fillna("").astype(str)
            ).str.lower()
            for term in excluded:
                candidates = candidates[~searchable.str.contains(term, regex=False)]
                searchable = searchable.loc[candidates.index]
        return candidates.head(int(arguments.get("top_k", 5))).reset_index(drop=True)

    @staticmethod
    def _tool_payload(candidates: pd.DataFrame) -> list[dict[str, Any]]:
        fields = ["product_id", "title", "category", "brand", "price", "rating", "evidence"]
        available = [field for field in fields if field in candidates.columns]
        return candidates[available].to_dict(orient="records")

    def run(self, need: str, user_id: object | None = None) -> AgentResult:
        trace: list[dict[str, Any]] = []
        try:
            first = self.client.responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=need,
                tools=[SEARCH_TOOL, CLARIFY_TOOL],
                tool_choice="required",
            )
            calls = [item for item in first.output if getattr(item, "type", "") == "function_call"]
            if len(calls) != 1:
                raise ValueError(f"expected exactly one tool call, received {len(calls)}")
            call = calls[0]
            arguments = json.loads(call.arguments)
            trace.append({"tool": call.name, "arguments": arguments, "status": "requested"})

            if call.name == "ask_clarification":
                question = str(arguments["question"])
                trace[-1]["status"] = "completed"
                return AgentResult(
                    status="needs_clarification",
                    answer=question,
                    clarification_question=question,
                    tool_trace=trace,
                    model=self.model,
                )
            if call.name != "search_catalog":
                raise ValueError(f"unsupported tool: {call.name}")

            candidates = self._search_catalog(arguments, user_id)
            payload = self._tool_payload(candidates)
            trace[-1].update({"status": "completed", "result_count": len(payload)})
            if not payload:
                return AgentResult(
                    status="completed",
                    answer="没有找到同时满足这些条件的商品，请放宽预算或筛选条件。",
                    tool_trace=trace,
                    model=self.model,
                )

            final = self.client.responses.parse(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                previous_response_id=first.id,
                input=[
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(payload, ensure_ascii=False),
                    }
                ],
                text_format=GroundedAnswer,
            )
            grounded = final.output_parsed
            allowed = {str(item["product_id"]): item for item in payload}
            recommendations = [
                item for item in grounded.recommendations if item.product_id in allowed
            ]
            dropped = len(grounded.recommendations) - len(recommendations)
            trace.append(
                {
                    "tool": "grounding_guard",
                    "status": "completed",
                    "allowed_product_ids": sorted(allowed),
                    "dropped_unsupported": dropped,
                }
            )
            return AgentResult(
                status="completed",
                answer=grounded.answer,
                recommendations=recommendations,
                tool_trace=trace,
                model=self.model,
            )
        except Exception as exc:  # noqa: BLE001 - external provider failures must degrade safely
            _intent, fallback = recommend_from_need(
                need, self.products, self.interactions, user_id=user_id, top_k=5
            )
            recommendations = [
                ProductExplanation(
                    product_id=str(row.product_id),
                    reason=str(row.reason),
                    evidence=[str(row.evidence)],
                )
                for row in fallback.itertuples(index=False)
            ]
            trace.append({"tool": "rule_fallback", "status": "completed"})
            return AgentResult(
                status="fallback",
                answer="大模型服务暂时不可用，已使用本地规则与检索结果降级返回。",
                recommendations=recommendations,
                tool_trace=trace,
                model=self.model,
                error=f"{type(exc).__name__}: {exc}",
            )


def create_openai_agent(
    products: pd.DataFrame,
    interactions: pd.DataFrame,
    api_key: str,
    model: str = "gpt-4.1-mini",
) -> LLMShoppingAgent:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2)
    return LLMShoppingAgent(products, interactions, client=client, model=model)
