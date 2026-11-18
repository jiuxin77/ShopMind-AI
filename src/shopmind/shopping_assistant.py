from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import pandas as pd

from shopmind.retrieval import hybrid_recommend

CATEGORIES = ("女装", "数码", "家居", "护肤", "食品")
SCENES = ("通勤", "办公", "旅行", "居家", "送礼", "健身", "学习")
STYLES = ("简约", "休闲", "商务", "复古", "清爽", "温和", "低糖", "便携")
CATEGORY_EN = {"女装": "Women's Fashion", "数码": "Electronics", "家居": "Home", "护肤": "Skincare", "食品": "Food"}
SCENE_EN = {"通勤": "commute", "办公": "office", "旅行": "travel", "居家": "home", "送礼": "gifting", "健身": "fitness", "学习": "study"}
STYLE_EN = {"简约": "minimalist", "休闲": "casual", "商务": "business", "复古": "vintage", "清爽": "refreshing", "温和": "gentle", "低糖": "low-sugar", "便携": "portable"}


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
        facts = [
            f"Category: {CATEGORY_EN.get(row['category'], row['category'])}",
            f"Price: ¥{row['price']:.2f}",
        ]
        if intent.budget_max is not None:
            facts.append(f"Within the ¥{intent.budget_max:.0f} budget")
        if intent.scenes:
            facts.append(f"Matches the {'/'.join(SCENE_EN.get(value, value) for value in intent.scenes)} context")
        if intent.styles:
            facts.append(
                f"Matches {'/'.join(STYLE_EN.get(value, value) for value in intent.styles)} preferences"
            )
        facts.append(f"Retrieval evidence: {row['evidence']}")
        return "；".join(facts)

    candidates["reason"] = candidates.apply(explain, axis=1)
    return intent, candidates.reset_index(drop=True)
