from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ensure_demo_data() -> None:
    required = [ROOT / "data" / name for name in ("users.csv", "products.csv", "interactions.csv")]
    if all(path.exists() for path in required):
        return
    print("[setup] Demo data is missing; generating deterministic sample data...")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_demo_data.py")], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the ShopMind AI portfolio demo")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    ensure_demo_data()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "src" / "shopmind" / "dashboard.py"),
        "--server.port",
        str(args.port),
        "--server.headless",
        "true" if args.headless else "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    print(f"[start] ShopMind AI → http://localhost:{args.port}")
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
