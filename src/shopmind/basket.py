from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class BasketIndex:
    neighbors: dict[str, list[tuple[str, float]]]

    def recommend(self, history: set[str], top_k: int = 50) -> pd.DataFrame:
        scores: dict[str, float] = {}
        for seed in sorted(history):
            for candidate, score in self.neighbors.get(seed, []):
                if candidate not in history:
                    scores[candidate] = scores.get(candidate, 0.0) + score
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
        result = pd.DataFrame(ranked, columns=["product_id", "raw_score"])
        result["source"] = "basket"
        result["rank"] = np.arange(1, len(result) + 1)
        return result


def build_basket_index(interactions: pd.DataFrame, neighbors_per_item: int = 100) -> BasketIndex:
    if "session_id" not in interactions.columns:
        return BasketIndex(neighbors={})
    pairs = interactions[["session_id", "product_id"]].dropna().drop_duplicates()
    if pairs.empty:
        return BasketIndex(neighbors={})
    session_codes, sessions = pd.factorize(pairs["session_id"], sort=True)
    item_codes, items = pd.factorize(pairs["product_id"], sort=True)
    matrix = csr_matrix(
        (np.ones(len(pairs), dtype=np.float32), (session_codes, item_codes)),
        shape=(len(sessions), len(items)),
    )
    similarity = cosine_similarity(matrix.T, dense_output=False).tocsr()
    neighbors: dict[str, list[tuple[str, float]]] = {}
    for item_index, item_id in enumerate(items.astype(str)):
        start, end = similarity.indptr[item_index], similarity.indptr[item_index + 1]
        candidates = [
            (str(items[index]), float(score))
            for index, score in zip(
                similarity.indices[start:end], similarity.data[start:end], strict=True
            )
            if index != item_index and score > 0
        ]
        neighbors[item_id] = sorted(candidates, key=lambda pair: pair[1], reverse=True)[
            :neighbors_per_item
        ]
    return BasketIndex(neighbors)
