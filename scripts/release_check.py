from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ARTIFACTS = [
    "m5_weight_tuning.json",
    "m6_recall_expansion.json",
    "m7_ranking_experiment.json",
    "m8_shopping_assistant_eval.json",
    "m9_visual_search_eval.json",
]


def run_check(name: str, command: list[str]) -> dict[str, object]:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    output_lines = (result.stdout or result.stderr).strip().splitlines()
    return {
        "name": name,
        "passed": result.returncode == 0,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "summary": output_lines[-1] if output_lines else "ok",
    }


def main() -> None:
    checks = [
        run_check("ruff", [sys.executable, "-m", "ruff", "check", "."]),
        run_check("pytest", [sys.executable, "-m", "pytest", "-q"]),
        run_check(
            "compile",
            [sys.executable, "-m", "compileall", "-q", "src/shopmind", "scripts"],
        ),
    ]
    artifact_status = {name: (ROOT / "artifacts" / name).exists() for name in REQUIRED_ARTIFACTS}
    output = {
        "release": "M10",
        "passed": all(check["passed"] for check in checks) and all(artifact_status.values()),
        "checks": checks,
        "required_artifacts": artifact_status,
    }
    destination = ROOT / "artifacts" / "m10_release_check.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
