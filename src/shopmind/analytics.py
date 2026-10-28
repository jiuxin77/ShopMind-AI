from __future__ import annotations

import pandas as pd

EVENT_ORDER = ["view", "click", "favorite", "cart", "purchase", "refund"]


def behavior_funnel(interactions: pd.DataFrame) -> pd.DataFrame:
    counts = interactions["event_type"].value_counts()
    result = pd.DataFrame({"event_type": EVENT_ORDER})
    result["count"] = result["event_type"].map(counts).fillna(0).astype(int)
    base = max(int(result.iloc[0]["count"]), 1)
    result["rate_from_view"] = result["count"] / base
    return result


def price_sensitivity(interactions: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    enriched = interactions.merge(products[["product_id", "price"]], on="product_id", how="left")
    enriched["price_band"] = pd.cut(
        enriched["price"],
        bins=[0, 100, 300, 600, 1000, float("inf")],
        labels=["0-100", "100-300", "300-600", "600-1000", "1000+"],
        right=False,
    )
    summary = (
        enriched.groupby("price_band", observed=True)
        .agg(
            interactions=("event_type", "size"),
            purchases=("event_type", lambda values: (values == "purchase").sum()),
        )
        .reset_index()
    )
    summary["purchase_rate"] = summary["purchases"] / summary["interactions"].clip(lower=1)
    return summary


def category_performance(interactions: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    enriched = interactions.merge(products[["product_id", "category"]], on="product_id", how="left")
    summary = (
        enriched.groupby("category")
        .agg(
            interactions=("event_type", "size"),
            purchases=("event_type", lambda values: (values == "purchase").sum()),
            refunds=("event_type", lambda values: (values == "refund").sum()),
        )
        .reset_index()
    )
    summary["conversion_rate"] = summary["purchases"] / summary["interactions"].clip(lower=1)
    summary["refund_rate"] = summary["refunds"] / summary["purchases"].clip(lower=1)
    return summary


def scene_performance(interactions: pd.DataFrame) -> pd.DataFrame:
    summary = (
        interactions.groupby("scene")
        .agg(
            interactions=("event_type", "size"),
            purchases=("event_type", lambda values: (values == "purchase").sum()),
        )
        .reset_index()
    )
    summary["conversion_rate"] = summary["purchases"] / summary["interactions"].clip(lower=1)
    return summary.sort_values("conversion_rate", ascending=False)
