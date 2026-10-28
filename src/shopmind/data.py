from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_demo_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load users, products and interactions generated for the demo."""
    paths = [data_dir / name for name in ("users.csv", "products.csv", "interactions.csv")]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Demo data missing: {', '.join(missing)}. Run generate_demo_data.py"
        )
    return tuple(pd.read_csv(path) for path in paths)  # type: ignore[return-value]
