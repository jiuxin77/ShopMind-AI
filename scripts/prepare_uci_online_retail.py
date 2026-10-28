from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed" / "uci_online_retail"


def download_dataset() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / "online_retail.zip"
    if not archive.exists():
        urllib.request.urlretrieve(DATASET_URL, archive)
    with zipfile.ZipFile(archive) as bundle:
        xlsx_name = next(name for name in bundle.namelist() if name.lower().endswith(".xlsx"))
        bundle.extract(xlsx_name, RAW_DIR)
    return RAW_DIR / xlsx_name


def prepare(max_users: int, max_products: int) -> dict[str, int | str]:
    source = download_dataset()
    raw = pd.read_excel(source, engine="openpyxl")
    raw.columns = [str(column).strip() for column in raw.columns]
    clean = raw.dropna(subset=["CustomerID", "StockCode", "Description", "InvoiceDate"]).copy()
    clean = clean[(clean["Quantity"] > 0) & (clean["UnitPrice"] > 0)]
    clean = clean[~clean["InvoiceNo"].astype(str).str.upper().str.startswith("C")]
    clean["CustomerID"] = clean["CustomerID"].astype(int).astype(str)
    clean["StockCode"] = clean["StockCode"].astype(str)

    active_users = clean.groupby("CustomerID").size().nlargest(max_users).index
    clean = clean[clean["CustomerID"].isin(active_users)]
    active_products = clean.groupby("StockCode").size().nlargest(max_products).index
    clean = clean[clean["StockCode"].isin(active_products)].sort_values("InvoiceDate")

    product_stats = (
        clean.groupby("StockCode")
        .agg(
            title=("Description", lambda values: values.mode().iloc[0]),
            price=("UnitPrice", "median"),
        )
        .reset_index()
    )
    products = product_stats.rename(columns={"StockCode": "product_id"})
    products["category"] = "giftware"
    products["brand"] = "unknown"
    products["rating"] = 4.0
    products["in_stock"] = True
    products["active"] = True

    users = pd.DataFrame({"user_id": sorted(clean["CustomerID"].unique())})
    users["age_group"] = "unknown"
    users["city_tier"] = 0
    users["preferred_scene"] = "online_retail"

    interactions = clean[
        ["CustomerID", "StockCode", "InvoiceNo", "InvoiceDate", "Country", "Quantity"]
    ].copy()
    interactions = interactions.rename(
        columns={"CustomerID": "user_id", "StockCode": "product_id", "InvoiceDate": "timestamp"}
    )
    interactions["interaction_id"] = [f"UCI{i:08d}" for i in range(len(interactions))]
    interactions["session_id"] = interactions["InvoiceNo"].astype(str)
    interactions["event_type"] = "purchase"
    interactions["scene"] = interactions["Country"].fillna("unknown")
    interactions = interactions[
        [
            "interaction_id",
            "user_id",
            "product_id",
            "event_type",
            "scene",
            "session_id",
            "timestamp",
        ]
    ]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    users.to_csv(PROCESSED_DIR / "users.csv", index=False)
    products.to_csv(PROCESSED_DIR / "products.csv", index=False)
    interactions.to_csv(PROCESSED_DIR / "interactions.csv", index=False)
    stats: dict[str, int | str] = {
        "source": "UCI Online Retail, DOI 10.24432/C5BW33",
        "license": "CC BY 4.0",
        "raw_rows": len(raw),
        "clean_rows": len(clean),
        "users": len(users),
        "products": len(products),
        "interactions": len(interactions),
    }
    (PROCESSED_DIR / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare UCI Online Retail data")
    parser.add_argument("--max-users", type=int, default=500)
    parser.add_argument("--max-products", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(prepare(args.max_users, args.max_products), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
