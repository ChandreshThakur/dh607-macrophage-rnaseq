#!/usr/bin/env python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastq_pipeline.reference import prepare  # noqa: E402

prepare(ROOT)

