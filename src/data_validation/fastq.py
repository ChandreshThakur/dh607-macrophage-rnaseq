"""Streaming FASTQ validation without replacing established QC software."""

from __future__ import annotations

import gzip
import itertools
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, TextIO


@dataclass
class FastqRecord:
    identifier: str
    sequence: str
    separator: str
    quality: str


@dataclass
class FastqStats:
    filename: str
    read_count: int = 0
    malformed_records: int = 0
    min_read_length: int | None = None
    max_read_length: int | None = None
    modal_read_length: int | None = None
    min_quality_ascii: int | None = None
    max_quality_ascii: int | None = None
    quality_encoding: str = "undetermined"
    first_read_id: str = ""
    last_read_id: str = ""
    mate_filename: str = ""
    mate_read_count_equal: bool = False
    pairing_id_mismatches: int = 0
    gzip_readable: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def canonical_read_id(identifier: str) -> str:
    """Remove FASTQ syntax and mate decoration from an identifier."""
    token = identifier.strip().split()[0]
    if token.startswith("@"):
        token = token[1:]
    if token.endswith("/1") or token.endswith("/2"):
        token = token[:-2]
    return token


def iter_fastq(handle: TextIO) -> Iterator[FastqRecord]:
    """Yield four-line records and fail clearly on a truncated record."""
    while True:
        lines = list(itertools.islice(handle, 4))
        if not lines:
            return
        if len(lines) != 4:
            raise ValueError(f"truncated FASTQ record with {len(lines)} line(s)")
        yield FastqRecord(*(line.rstrip("\r\n") for line in lines))


def _update(stats: FastqStats, lengths: Counter[int], record: FastqRecord) -> None:
    stats.read_count += 1
    if not stats.first_read_id:
        stats.first_read_id = record.identifier
    stats.last_read_id = record.identifier
    length = len(record.sequence)
    lengths[length] += 1
    stats.min_read_length = length if stats.min_read_length is None else min(stats.min_read_length, length)
    stats.max_read_length = length if stats.max_read_length is None else max(stats.max_read_length, length)
    if (
        not record.identifier.startswith("@")
        or not record.separator.startswith("+")
        or not record.sequence
        or len(record.sequence) != len(record.quality)
    ):
        stats.malformed_records += 1
    if record.quality:
        qmin = min(map(ord, record.quality))
        qmax = max(map(ord, record.quality))
        stats.min_quality_ascii = qmin if stats.min_quality_ascii is None else min(stats.min_quality_ascii, qmin)
        stats.max_quality_ascii = qmax if stats.max_quality_ascii is None else max(stats.max_quality_ascii, qmax)


def _finish(stats: FastqStats, lengths: Counter[int]) -> None:
    if lengths:
        stats.modal_read_length = lengths.most_common(1)[0][0]
    qmin, qmax = stats.min_quality_ascii, stats.max_quality_ascii
    if qmin is None:
        stats.quality_encoding = "undetermined"
    elif qmin < 59:
        stats.quality_encoding = "Phred+33"
    elif qmin >= 64:
        stats.quality_encoding = "ambiguous Phred+33/Phred+64 from observed range"
    else:
        stats.quality_encoding = "most consistent with Phred+33"
    if qmax is not None and qmax > 104:
        stats.quality_encoding = "invalid/unknown (quality ASCII > 104)"


def validate_pair(read1: Path, read2: Path) -> tuple[FastqStats, FastqStats]:
    """Validate both gzipped mates in one streaming pass and compare every ID."""
    stats1, stats2 = FastqStats(read1.name), FastqStats(read2.name)
    stats1.mate_filename, stats2.mate_filename = read2.name, read1.name
    lengths1: Counter[int] = Counter()
    lengths2: Counter[int] = Counter()
    try:
        with gzip.open(read1, "rt", encoding="ascii") as left, gzip.open(
            read2, "rt", encoding="ascii"
        ) as right:
            for rec1, rec2 in itertools.zip_longest(iter_fastq(left), iter_fastq(right)):
                if rec1 is not None:
                    _update(stats1, lengths1, rec1)
                if rec2 is not None:
                    _update(stats2, lengths2, rec2)
                if rec1 is None or rec2 is None:
                    continue
                if canonical_read_id(rec1.identifier) != canonical_read_id(rec2.identifier):
                    stats1.pairing_id_mismatches += 1
                    stats2.pairing_id_mismatches += 1
    except (OSError, EOFError, UnicodeDecodeError):
        stats1.gzip_readable = False
        stats2.gzip_readable = False
        raise
    _finish(stats1, lengths1)
    _finish(stats2, lengths2)
    equal = stats1.read_count == stats2.read_count
    stats1.mate_read_count_equal = equal
    stats2.mate_read_count_equal = equal
    return stats1, stats2

