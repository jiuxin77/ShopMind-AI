from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from shopmind.recommender import EVENT_WEIGHT

POSITIVE_EVENTS = {"click", "favorite", "cart", "purchase"}


def _tokens(text: str) -> list[str]:
    """Tokenize mixed Chinese/English product text without an external segmenter."""
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", str(text).lower()).strip()
    words = normalized.split()
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    return words + [chinese[i : i + 2] for i in range(max(len(chinese) - 1, 0))]


def popular_recall(interactions: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    events = interactions.copy()
    events["raw_score"] = events["event_type"].map(EVENT_WEIGHT).fillna(0.0)
    result = events.groupby("product_id", as_index=False)["raw_score"].sum()
    result = result.sort_values("raw_score", ascending=False).head(top_k).reset_index(drop=True)
    result["source"] = "popular"
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def item_cf_recall(user_id: str, interactions: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    positive = interactions[interactions["event_type"].isin(POSITIVE_EVENTS)][
        ["user_id", "product_id"]
    ]
    positive = positive.drop_duplicates()
    history = set(positive.loc[positive["user_id"] == user_id, "product_id"])
    if not history:
        return pd.DataFrame(columns=["product_id", "raw_score", "source", "rank"])

    user_items = positive.groupby("user_id")["product_id"].apply(set)
    item_users: dict[str, set[str]] = defaultdict(set)
    for uid, items in user_items.items():
        for item in items:
            item_users[item].add(uid)

    scores: Counter[str] = Counter()
    for seed in history:
        seed_users = item_users[seed]
        for candidate, candidate_users in item_users.items():
            if candidate in history:
                continue
            overlap = len(seed_users & candidate_users)
            if overlap:
                scores[candidate] += overlap / math.sqrt(len(seed_users) * len(candidate_users))

    result = pd.DataFrame(scores.most_common(top_k), columns=["product_id", "raw_score"])
    result["source"] = "item_cf"
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def bm25_recall(query: str, products: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    query_tokens = _tokens(query)
    if not query_tokens:
        return pd.DataFrame(columns=["product_id", "raw_score", "source", "rank"])

    documents = (
        products["title"].fillna("")
        + " "
        + products["category"].fillna("")
        + " "
        + products["brand"].fillna("")
    ).map(_tokens)
    document_frequency = Counter(token for tokens in documents for token in set(tokens))
    avg_length = max(float(documents.map(len).mean()), 1.0)
    n_documents = len(documents)
    k1, b = 1.5, 0.75
    scores: list[float] = []
    for tokens in documents:
        frequencies = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            tf = frequencies[token]
            if not tf:
                continue
            df = document_frequency[token]
            idf = math.log(1 + (n_documents - df + 0.5) / (df + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(tokens) / avg_length))
        scores.append(score)

    result = pd.DataFrame({"product_id": products["product_id"], "raw_score": scores})
    result = result[result["raw_score"] > 0].sort_values("raw_score", ascending=False).head(top_k)
    result = result.reset_index(drop=True)
    result["source"] = "bm25"
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def reciprocal_rank_fusion(
    result_sets: list[pd.DataFrame], top_k: int = 20, rrf_k: int = 60
) -> pd.DataFrame:
    """Fuse ranked candidates while preserving per-source evidence."""
    scores: defaultdict[str, float] = defaultdict(float)
    evidence: defaultdict[str, list[str]] = defaultdict(list)
    for result in result_sets:
        for row in result.itertuples(index=False):
            scores[row.product_id] += 1.0 / (rrf_k + int(row.rank))
            evidence[row.product_id].append(f"{row.source}#{row.rank}")
    rows = [
        {"product_id": product_id, "score": score, "evidence": ", ".join(evidence[product_id])}
        for product_id, score in scores.items()
    ]
    return (
        pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    )


def hybrid_recommend(
    products: pd.DataFrame,
    interactions: pd.DataFrame,
    user_id: str | None = None,
    query: str = "",
    category: str | None = None,
    budget_max: float | None = None,
    top_k: int = 10,
) -> pd.DataFrame:
    recalls = [popular_recall(interactions)]
    if user_id:
        recalls.append(item_cf_recall(user_id, interactions))
    if query.strip():
        recalls.append(bm25_recall(query, products))
    fused = reciprocal_rank_fusion(recalls, top_k=max(top_k * 5, 50))
    candidates = products.merge(fused, on="product_id", how="inner")
    candidates = candidates[(candidates["in_stock"] == True) & (candidates["active"] == True)]
    if category:
        candidates = candidates[candidates["category"] == category]
    if budget_max is not None:
        candidates = candidates[candidates["price"] <= budget_max]
    candidates["reason"] = candidates["evidence"].map(lambda value: f"由多路召回命中：{value}")
    return candidates.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
