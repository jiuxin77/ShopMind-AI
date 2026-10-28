from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import pandas as pd

from shopmind.retrieval import hybrid_recommend

CATEGORIES = ("女装", "数码", "家居", "护肤", "食品")
SCENES = ("通勤", "办公", "旅行", "居家", "送礼", "健身", "学习")
STYLES = ("简约", "休闲", "商务", "复古", "清爽", "温和", "低糖", "便携")


@dataclass(frozen=True)
class ShoppingIntent:
    original_query: str
    category: str | None
    budget_min: float | None
    budget_max: float | None
    scenes: list[str]
    styles: list[str]
    exclusions: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _first_amount(patterns: tuple[str, ...], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def parse_shopping_intent(text: str) -> ShoppingIntent:
    normalized = re.sub(r"\s+", " ", text.strip())
    category = next((value for value in CATEGORIES if value in normalized), None)
    budget_max = _first_amount(
        (
            r"(?:预算|价格)?\s*(\d+(?:\.\d+)?)\s*元?\s*(?:以内|以下)",
            r"(?:不超过|最多|低于)\s*(\d+(?:\.\d+)?)",
            r"预算\s*(\d+(?:\.\d+)?)",
        ),
        normalized,
    )
    budget_min = _first_amount(
        (r"(?:至少|高于|不低于)\s*(\d+(?:\.\d+)?)", r"(\d+(?:\.\d+)?)\s*元?\s*以上"),
        normalized,
    )
    scenes = [value for value in SCENES if value in normalized]
    styles = [value for value in STYLES if value in normalized]
    exclusions = []
    for match in re.finditer(r"(?:不要|排除|不考虑)\s*([^，。；,;]+)", normalized):
        phrase = re.split(r"(?:，|。|；|,|;|预算|价格)", match.group(1))[0].strip()
        phrase = re.sub(r"(?:这个)?品牌$", "", phrase).strip()
        if phrase:
            exclusions.append(phrase)
    return ShoppingIntent(
        original_query=normalized,
        category=category,
        budget_min=budget_min,
        budget_max=budget_max,
        scenes=scenes,
        styles=styles,
        exclusions=list(dict.fromkeys(exclusions)),
    )


def _exclusion_mask(products: pd.DataFrame, exclusions: list[str]) -> pd.Series:
    searchable = (
        products["title"].fillna("").astype(str)
        + " "
        + products["brand"].fillna("").astype(str)
        + " "
        + products["category"].fillna("").astype(str)
    ).str.lower()
    mask = pd.Series(True, index=products.index)
    for term in exclusions:
        mask &= ~searchable.str.contains(re.escape(term.lower()), regex=True)
    return mask


def recommend_from_need(
    text: str,
    products: pd.DataFrame,
    interactions: pd.DataFrame,
    user_id: object | None = None,
    top_k: int = 10,
) -> tuple[ShoppingIntent, pd.DataFrame]:
    intent = parse_shopping_intent(text)
    candidates = hybrid_recommend(
        products,
        interactions,
        user_id=user_id,
        query=intent.original_query,
        category=intent.category,
        budget_max=intent.budget_max,
        top_k=max(top_k * 5, 50),
    )
    if intent.budget_min is not None:
        candidates = candidates[candidates["price"] >= intent.budget_min]
    if intent.exclusions and not candidates.empty:
        candidates = candidates[_exclusion_mask(candidates, intent.exclusions)]
    candidates = candidates.head(top_k).copy()
    if candidates.empty:
        return intent, candidates

    def explain(row: pd.Series) -> str:
        facts = [f"{row['category']}商品", f"价格 ¥{row['price']:.2f}"]
        if intent.budget_max is not None:
            facts.append(f"未超过 ¥{intent.budget_max:.0f} 预算")
        if intent.scenes:
            facts.append(f"匹配{'/'.join(intent.scenes)}场景")
        if intent.styles:
            facts.append(f"包含{'/'.join(intent.styles)}偏好")
        facts.append(f"召回证据：{row['evidence']}")
        return "；".join(facts)

    candidates["reason"] = candidates.apply(explain, axis=1)
    return intent, candidates.reset_index(drop=True)
