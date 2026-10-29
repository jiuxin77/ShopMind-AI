from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

CATEGORY_COLORS = {
    "女装": (205, 117, 145),
    "数码": (67, 110, 168),
    "家居": (103, 146, 103),
    "护肤": (191, 151, 113),
    "食品": (205, 153, 65),
    "giftware": (125, 112, 177),
}


@dataclass
class VisualIndex:
    product_ids: list[str]
    embeddings: np.ndarray
    image_paths: list[Path]


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def product_image_path(image_dir: Path, product_id: str) -> Path:
    return image_dir / f"{hashlib.sha1(product_id.encode('utf-8')).hexdigest()}.png"


def generate_demo_image(row: pd.Series, size: int = 192) -> Image.Image:
    """Create a deterministic synthetic thumbnail; never presented as a real product photo."""
    product_id = str(row["product_id"])
    digest = _digest(product_id)
    base = CATEGORY_COLORS.get(str(row["category"]), (120, 120, 120))
    background = tuple(min(255, int(channel * 0.55 + 95)) for channel in base)
    accent = tuple(45 + digest[index] % 180 for index in range(3))
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)
    margin = 18 + digest[3] % 16
    shape = digest[4] % 3
    box = (margin, margin, size - margin, size - margin - 34)
    if shape == 0:
        draw.rounded_rectangle(box, radius=18, fill=accent, outline=(255, 255, 255), width=4)
    elif shape == 1:
        draw.ellipse(box, fill=accent, outline=(255, 255, 255), width=4)
    else:
        draw.polygon(
            [
                (size // 2, margin),
                (size - margin, size - margin - 34),
                (margin, size - margin - 34),
            ],
            fill=accent,
            outline=(255, 255, 255),
        )
    stripe = 4 + digest[5] % 10
    for x in range(margin, size - margin, stripe * 2):
        draw.line((x, margin, x, size - margin - 34), fill=(255, 255, 255), width=2)
    font = ImageFont.load_default()
    label = f"{product_id[:10]}  {str(row['category'])[:10]}"
    draw.rectangle((0, size - 32, size, size), fill=(25, 31, 45))
    draw.text((10, size - 23), label, fill=(255, 255, 255), font=font)
    return image


def ensure_demo_catalog_images(products: pd.DataFrame, image_dir: Path) -> list[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for _, row in products.iterrows():
        path = product_image_path(image_dir, str(row["product_id"]))
        if not path.exists():
            generate_demo_image(row).save(path)
        paths.append(path)
    return paths


def image_embedding(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB").resize((32, 32)), dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV").resize((32, 32)), dtype=np.float32) / 255.0
    histograms = [
        np.histogram(hsv[:, :, i], bins=12, range=(0, 1), density=True)[0] for i in range(2)
    ]
    chromaticity = rgb / np.maximum(rgb.sum(axis=2, keepdims=True), 1e-6)
    spatial = chromaticity.reshape(4, 8, 4, 8, 3).mean(axis=(1, 3)).ravel()
    gray = rgb.mean(axis=2)
    gray = gray / max(float(gray.mean()), 1e-6)
    edges = np.array([np.abs(np.diff(gray, axis=0)).mean(), np.abs(np.diff(gray, axis=1)).mean()])
    vector = np.concatenate([*histograms, spatial, edges]).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def build_visual_index(products: pd.DataFrame, image_dir: Path) -> VisualIndex:
    paths = ensure_demo_catalog_images(products, image_dir)
    embeddings = np.vstack([image_embedding(Image.open(path)) for path in paths])
    return VisualIndex(products["product_id"].astype(str).tolist(), embeddings, paths)


def visual_search(
    query: Image.Image,
    products: pd.DataFrame,
    index: VisualIndex,
    top_k: int = 5,
) -> pd.DataFrame:
    scores = index.embeddings @ image_embedding(query)
    order = np.argsort(-scores)[:top_k]
    matches = pd.DataFrame(
        {
            "product_id": [index.product_ids[i] for i in order],
            "visual_score": [float(scores[i]) for i in order],
            "image_path": [str(index.image_paths[i]) for i in order],
        }
    )
    return matches.merge(products, on="product_id", how="left")


def brightness_variant(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)
