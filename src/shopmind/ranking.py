from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shopmind.item_cf import ItemCFIndex
from shopmind.retrieval import POSITIVE_EVENTS, hybrid_recommend

SOURCES = ("popular", "item_cf", "bm25", "text_vector", "recent_interest")
FEATURE_COLUMNS = [
    "rrf_score",
    "product_popularity",
    "log_product_popularity",
    "price",
    "price_distance",
    "history_size",
    "evidence_count",
    *[f"{source}_rr" for source in SOURCES],
]


@dataclass
class RankingContext:
    popularity: dict[str, float]
    user_median_price: dict[str, float]
    user_history_size: dict[str, int]
    global_median_price: float


def build_ranking_context(products: pd.DataFrame, interactions: pd.DataFrame) -> RankingContext:
    positive = interactions[interactions["event_type"].isin(POSITIVE_EVENTS)].copy()
    popularity = positive["product_id"].astype(str).value_counts().astype(float).to_dict()
    history_size = positive.groupby("user_id").size().astype(int).to_dict()
    priced = positive.merge(products[["product_id", "price"]], on="product_id", how="left")
    user_median = priced.groupby("user_id")["price"].median().to_dict()
    return RankingContext(
        popularity=popularity,
        user_median_price=user_median,
        user_history_size=history_size,
        global_median_price=float(products["price"].median()),
    )


def candidate_features(
    candidates: pd.DataFrame,
    user_id: object,
    context: RankingContext,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    median_price = context.user_median_price.get(user_id, context.global_median_price)
    for row in candidates.itertuples(index=False):
        ranks = {source: 0.0 for source in SOURCES}
        evidence = str(row.evidence)
        for source, rank in re.findall(r"([a-z_]+)#(\d+)", evidence):
            if source in ranks:
                ranks[source] = 1.0 / float(rank)
        popularity = context.popularity.get(str(row.product_id), 0.0)
        values = {
            "rrf_score": float(row.score),
            "product_popularity": popularity,
            "log_product_popularity": float(np.log1p(popularity)),
            "price": float(row.price),
            "price_distance": abs(float(row.price) - median_price),
            "history_size": float(context.user_history_size.get(user_id, 0)),
            "evidence_count": float(len(re.findall(r"#\d+", evidence))),
            **{f"{source}_rr": ranks[source] for source in SOURCES},
        }
        rows.append(values)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def train_pointwise_ranker(
    products: pd.DataFrame,
    train_interactions: pd.DataFrame,
    validation_targets: pd.DataFrame,
    item_cf_index: ItemCFIndex,
    source_weights: dict[str, float],
    candidate_k: int = 50,
) -> tuple[Pipeline, dict[str, int]]:
    context = build_ranking_context(products, train_interactions)
    feature_frames: list[pd.DataFrame] = []
    labels: list[int] = []
    positive_groups = 0
    for target in validation_targets.itertuples(index=False):
        candidates = hybrid_recommend(
            products,
            train_interactions,
            user_id=target.user_id,
            top_k=candidate_k,
            sources=set(source_weights),
            source_weights=source_weights,
            item_cf_index=item_cf_index,
        )
        frame = candidate_features(candidates, target.user_id, context)
        target_labels = (candidates["product_id"].astype(str) == str(target.product_id)).astype(int)
        if target_labels.sum():
            positive_groups += 1
        feature_frames.append(frame)
        labels.extend(target_labels.tolist())
    x = pd.concat(feature_frames, ignore_index=True)
    y = np.asarray(labels)
    if y.sum() == 0:
        raise ValueError("No positive validation candidates available to train the ranker")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            ),
        ]
    )
    model.fit(x, y)
    return model, {
        "training_rows": len(y),
        "positive_rows": int(y.sum()),
        "positive_user_groups": positive_groups,
    }


def evaluate_ranker(
    model: Pipeline,
    products: pd.DataFrame,
    train_interactions: pd.DataFrame,
    test_targets: pd.DataFrame,
    item_cf_index: ItemCFIndex,
    source_weights: dict[str, float],
    k: int = 10,
    candidate_k: int = 50,
    model_weight: float = 0.1,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    context = build_ranking_context(products, train_interactions)
    hits = 0
    ndcg_sum = 0.0
    candidate_hits = 0
    latencies: list[float] = []
    recommended: set[str] = set()
    details: list[dict[str, object]] = []
    for target in test_targets.itertuples(index=False):
        started = time.perf_counter()
        candidates = hybrid_recommend(
            products,
            train_interactions,
            user_id=target.user_id,
            top_k=candidate_k,
            sources=set(source_weights),
            source_weights=source_weights,
            item_cf_index=item_cf_index,
        ).copy()
        features = candidate_features(candidates, target.user_id, context)
        candidates["model_score"] = model.predict_proba(features)[:, 1]
        candidates["base_rank"] = np.arange(1, len(candidates) + 1)
        candidates["model_rank"] = candidates["model_score"].rank(method="first", ascending=False)
        candidates["ranking_score"] = (1.0 - model_weight) / candidates["base_rank"] + (
            model_weight / candidates["model_rank"]
        )
        candidates = candidates.sort_values("ranking_score", ascending=False).reset_index(drop=True)
        latency = (time.perf_counter() - started) * 1000
        ranking = candidates["product_id"].astype(str).tolist()
        target_id = str(target.product_id)
        candidate_rank = ranking.index(target_id) + 1 if target_id in ranking else None
        if candidate_rank is not None:
            candidate_hits += 1
        hit = candidate_rank is not None and candidate_rank <= k
        if hit:
            hits += 1
            ndcg_sum += 1.0 / np.log2(candidate_rank + 1)
        recommended.update(ranking[:k])
        latencies.append(latency)
        details.append(
            {
                "user_id": target.user_id,
                "target_product_id": target_id,
                "hit": hit,
                "rank": candidate_rank if hit else None,
                "candidate_rank": candidate_rank,
                "bad_case_type": ""
                if hit
                else "ranking_miss"
                if candidate_rank
                else "candidate_not_recalled",
                "latency_ms": latency,
            }
        )
    users = max(len(test_targets), 1)
    metrics: dict[str, float | int] = {
        "users": len(test_targets),
        "k": k,
        "hits": hits,
        "precision_at_k": hits / (users * k),
        "recall_at_k": hits / users,
        "hit_rate_at_k": hits / users,
        "candidate_recall": candidate_hits / users,
        "ndcg_at_k": ndcg_sum / users,
        "catalog_coverage": len(recommended) / max(len(products), 1),
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
    }
    return metrics, pd.DataFrame(details)
