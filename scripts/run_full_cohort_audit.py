#!/usr/bin/env python
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exploratory_analysis.full_cohort_audit import run  # noqa: E402

print(json.dumps(run(ROOT), indent=2))

