from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

POSITIVE_EVENTS = {"click", "favorite", "cart", "purchase"}


@dataclass
class ItemCFIndex:
    neighbors: dict[str, list[tuple[str, float]]]

    def recommend(self, history: set[str], top_k: int = 50) -> pd.DataFrame:
        scores: dict[str, float] = {}
        for seed in sorted(history):
            for candidate, similarity in self.neighbors.get(seed, []):
                if candidate not in history:
                    scores[candidate] = scores.get(candidate, 0.0) + similarity
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        result = pd.DataFrame(ranked, columns=["product_id", "raw_score"])
        result["source"] = "item_cf"
        result["rank"] = np.arange(1, len(result) + 1)
        return result

    def recommend_recent(self, history: list[str], top_k: int = 50) -> pd.DataFrame:
        """Favor the most recent items while retaining neighbor similarity."""
        scores: dict[str, float] = {}
        seen = set(history)
        for position, seed in enumerate(reversed(history)):
            recency_weight = 1.0 / (1.0 + position)
            for candidate, similarity in self.neighbors.get(seed, []):
                if candidate not in seen:
                    scores[candidate] = scores.get(candidate, 0.0) + similarity * recency_weight
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        result = pd.DataFrame(ranked, columns=["product_id", "raw_score"])
        result["source"] = "recent_interest"
        result["rank"] = np.arange(1, len(result) + 1)
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> ItemCFIndex:
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Unexpected index type: {type(value)!r}")
        return value


def build_item_cf_index(interactions: pd.DataFrame, neighbors_per_item: int = 100) -> ItemCFIndex:
    positive = interactions[interactions["event_type"].isin(POSITIVE_EVENTS)][
        ["user_id", "product_id"]
    ].drop_duplicates()
    if positive.empty:
        return ItemCFIndex(neighbors={})
    user_codes, users = pd.factorize(positive["user_id"], sort=True)
    item_codes, items = pd.factorize(positive["product_id"], sort=True)
    matrix = csr_matrix(
        (np.ones(len(positive), dtype=np.float32), (user_codes, item_codes)),
        shape=(len(users), len(items)),
    )
    similarity = cosine_similarity(matrix.T, dense_output=False).tocsr()
    neighbors: dict[str, list[tuple[str, float]]] = {}
    for item_index, item_id in enumerate(items.astype(str)):
        start, end = similarity.indptr[item_index], similarity.indptr[item_index + 1]
        indices = similarity.indices[start:end]
        values = similarity.data[start:end]
        candidates = [
            (str(items[index]), float(score))
            for index, score in zip(indices, values, strict=True)
            if index != item_index and score > 0
        ]
        neighbors[item_id] = sorted(candidates, key=lambda pair: pair[1], reverse=True)[
            :neighbors_per_item
        ]
    return ItemCFIndex(neighbors=neighbors)
