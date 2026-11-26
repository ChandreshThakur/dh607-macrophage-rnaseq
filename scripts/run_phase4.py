#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biological_analysis.phase4_interaction import finalize, run_count_model, run_course  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the validated Phase 4 interaction analysis")
    parser.add_argument("--stage", choices=("course", "count", "finalize", "all"), default="all")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    output = {}
    if args.stage in ("course", "all"):
        output["course"] = run_course(ROOT)
    if args.stage in ("count", "all"):
        output["count"] = run_count_model(ROOT)
    if args.stage in ("finalize", "all"):
        output["summary"] = finalize(ROOT)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
