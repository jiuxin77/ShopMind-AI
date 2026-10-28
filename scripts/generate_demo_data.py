from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def main() -> None:
    rng = random.Random(SEED)
    categories = {
        "护肤": ["舒缓面霜", "清爽精华", "温和洁面"],
        "女装": ["通勤衬衫", "休闲针织衫", "简约半身裙"],
        "数码": ["无线耳机", "机械键盘", "移动电源"],
        "家居": ["护眼台灯", "床品套装", "香薰机"],
        "食品": ["低糖麦片", "坚果组合", "即食燕麦"],
    }
    brands = ["青禾", "云间", "星野", "简物", "北岸"]
    scenes = ["通勤", "宿舍", "旅行", "送礼", "运动", "日常"]
    events = ["view", "click", "favorite", "cart", "purchase", "refund"]
    event_weights = [45, 25, 8, 10, 10, 2]

    users = pd.DataFrame([
        {"user_id": f"U{i:04d}", "age_group": rng.choice(["18-24", "25-34", "35-44"]),
         "city_tier": rng.choice([1, 2, 3]), "preferred_scene": rng.choice(scenes)}
        for i in range(1, 301)
    ])
    product_rows = []
    for i in range(1, 201):
        category = rng.choice(list(categories))
        product_rows.append({
            "product_id": f"P{i:04d}", "title": rng.choice(categories[category]),
            "category": category, "brand": rng.choice(brands),
            "price": round(rng.uniform(39, 1599), 2), "rating": round(rng.uniform(3.5, 5.0), 1),
            "in_stock": rng.random() > 0.05, "active": rng.random() > 0.03,
        })
    products = pd.DataFrame(product_rows)

    now = datetime.now(UTC)
    interaction_rows = []
    for i in range(5000):
        interaction_rows.append({
            "interaction_id": f"I{i:06d}", "user_id": rng.choice(users["user_id"].tolist()),
            "product_id": rng.choice(products["product_id"].tolist()),
            "event_type": rng.choices(events, weights=event_weights, k=1)[0],
            "scene": rng.choice(scenes),
            "timestamp": (now - timedelta(hours=rng.randint(0, 24 * 90))).isoformat(),
        })
    interactions = pd.DataFrame(interaction_rows)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    users.to_csv(DATA_DIR / "users.csv", index=False)
    products.to_csv(DATA_DIR / "products.csv", index=False)
    interactions.to_csv(DATA_DIR / "interactions.csv", index=False)
    print(f"Generated {len(users)} users, {len(products)} products, {len(interactions)} interactions")


if __name__ == "__main__":
    main()

