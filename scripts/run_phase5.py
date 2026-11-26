#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biological_analysis.phase5_validation import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 adversarial validation")
    parser.add_argument("--permutations", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(run(ROOT, n_permutations=args.permutations), indent=2))


if __name__ == "__main__":
    main()

