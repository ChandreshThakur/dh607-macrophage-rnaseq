#!/usr/bin/env python
"""Run FastQC and MultiQC only when the established tools are installed."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FASTQ = ROOT / "data/raw_fastq"
FASTQC_OUT = ROOT / "results/qc/fastqc"
MULTIQC_OUT = ROOT / "results/qc/multiqc"
STATUS = ROOT / "results/qc/tool_status.json"


def main() -> int:
    fastqc = shutil.which("fastqc")
    bundled = ROOT / ".tools/fastqc-0.12.1/FastQC"
    multiqc = shutil.which("multiqc") or str(ROOT / ".tools/multiqc-env/Scripts/multiqc.exe")
    java = shutil.which("java")
    bundled_available = bundled.is_dir() and java is not None
    status = {
        "fastqc": fastqc or (str(bundled) if bundled_available else "not found"),
        "multiqc": multiqc if Path(multiqc).is_file() or shutil.which(multiqc) else "not found",
        "fastq_count": len(list(FASTQ.glob("*.fastq.gz"))),
        "status": "blocked_missing_tools" if (not fastqc and not bundled_available) or not Path(multiqc).is_file() else "running",
    }
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    if (not fastqc and not bundled_available) or not Path(multiqc).is_file():
        STATUS.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print("FastQC and/or MultiQC not found; see results/qc/tool_status.json")
        return 2
    FASTQC_OUT.mkdir(parents=True, exist_ok=True)
    MULTIQC_OUT.mkdir(parents=True, exist_ok=True)
    inputs = sorted(str(path) for path in FASTQ.glob("*.fastq.gz"))
    if fastqc:
        subprocess.run([fastqc, "--threads", "4", "--extract", "--outdir", str(FASTQC_OUT), *inputs], check=True)
    else:
        classpath = os.pathsep.join([".", "htsjdk.jar", "jbzip2-0.9.jar", "cisd-jhdf5.jar"])
        command = [
            java, "-Xmx2048m", "-XX:+ExitOnOutOfMemoryError", "-XX:ParallelGCThreads=1",
            f"-Dfastqc.output_dir={FASTQC_OUT}", "-Dfastqc.threads=4", "-Dfastqc.unzip=true",
            "-Djava.awt.headless=true", "-classpath", classpath,
            "uk.ac.babraham.FastQC.FastQCApplication", *inputs,
        ]
        subprocess.run(command, cwd=bundled, check=True)
    subprocess.run([multiqc, str(FASTQC_OUT), "--outdir", str(MULTIQC_OUT), "--force"], check=True)
    status["status"] = "complete"
    STATUS.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
