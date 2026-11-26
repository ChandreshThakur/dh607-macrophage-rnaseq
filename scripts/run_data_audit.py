#!/usr/bin/env python
"""Run the reproducible local-input audit from the repository root."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_validation.audit import run  # noqa: E402

run(ROOT)

