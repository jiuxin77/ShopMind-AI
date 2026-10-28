from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from shopmind.item_cf import build_item_cf_index
from shopmind.retrieval import POSITIVE_EVENTS, hybrid_recommend


def leave_one_out_split(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive = interactions[interactions["event_type"].isin(POSITIVE_EVENTS)].copy()
    positive["timestamp"] = pd.to_datetime(positive["timestamp"], utc=True)
    eligible = positive.groupby("user_id").filter(lambda group: group["product_id"].nunique() >= 2)
    test_indices = eligible.sort_values("timestamp").groupby("user_id").tail(1).index
    return interactions.drop(index=test_indices).copy(), interactions.loc[test_indices].copy()


def evaluate_with_details(
    products: pd.DataFrame,
    interactions: pd.DataFrame,
    k: int = 10,
    max_users: int = 100,
    sources: set[str] | None = None,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    train, test = leave_one_out_split(interactions)
    test = test.head(max_users)
    selected = sources or {"popular", "item_cf", "bm25", "text_vector"}
    item_cf_index = build_item_cf_index(train) if "item_cf" in selected else None
    hits = 0
    ndcg_sum = 0.0
    recommended_items: set[str] = set()
    latencies_ms: list[float] = []
    details: list[dict[str, object]] = []
    for row in test.itertuples(index=False):
        started = time.perf_counter()
        result = hybrid_recommend(
            products,
            train,
            user_id=row.user_id,
            top_k=k,
            sources=selected,
            item_cf_index=item_cf_index,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        latencies_ms.append(latency_ms)
        ranked = result["product_id"].tolist()
        recommended_items.update(ranked)
        hit = row.product_id in ranked
        rank = ranked.index(row.product_id) + 1 if hit else None
        if hit:
            hits += 1
            assert rank is not None
            ndcg_sum += 1.0 / math.log2(rank + 1)
        details.append(
            {
                "user_id": row.user_id,
                "target_product_id": row.product_id,
                "hit": hit,
                "rank": rank,
                "recommended_product_ids": "|".join(ranked),
                "bad_case_type": "" if hit else "target_not_recalled",
                "latency_ms": latency_ms,
            }
        )
    users = len(test)
    denominator = max(users, 1)
    metrics: dict[str, float | int] = {
        "users": users,
        "k": k,
        "hits": hits,
        "precision_at_k": hits / (denominator * k),
        "recall_at_k": hits / denominator,
        "hit_rate_at_k": hits / denominator,
        "ndcg_at_k": ndcg_sum / denominator,
        "catalog_coverage": len(recommended_items) / max(len(products), 1),
        "mean_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
    }
    return metrics, pd.DataFrame(details)


def evaluate_hybrid(
    products: pd.DataFrame,
    interactions: pd.DataFrame,
    k: int = 10,
    max_users: int = 100,
    sources: set[str] | None = None,
) -> dict[str, float | int]:
    metrics, _ = evaluate_with_details(
        products, interactions, k=k, max_users=max_users, sources=sources
    )
    return metrics


def compare_retrievers(
    products: pd.DataFrame,
    interactions: pd.DataFrame,
    k: int = 10,
    max_users: int = 50,
) -> pd.DataFrame:
    configurations = {
        "popular": {"popular"},
        "item_cf": {"item_cf"},
        "text_vector": {"text_vector"},
        "hybrid": {"popular", "item_cf", "bm25", "text_vector"},
    }
    rows = []
    for model, sources in configurations.items():
        metrics = evaluate_hybrid(products, interactions, k=k, max_users=max_users, sources=sources)
        rows.append({"model": model, **metrics})
    return pd.DataFrame(rows)
