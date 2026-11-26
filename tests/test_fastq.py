import gzip

from data_validation.fastq import canonical_read_id, validate_pair


def _write(path, records):
    with gzip.open(path, "wt", encoding="ascii") as handle:
        for identifier, sequence, quality in records:
            handle.write(f"{identifier}\n{sequence}\n+\n{quality}\n")


def test_canonical_read_id_handles_both_common_styles():
    assert canonical_read_id("@abc/1") == "abc"
    assert canonical_read_id("@abc 2:N:0:1") == "abc"


def test_pair_validation_counts_and_matches(tmp_path):
    left, right = tmp_path / "x.1.fastq.gz", tmp_path / "x.2.fastq.gz"
    _write(left, [("@a/1", "ACGT", "!!!!"), ("@b/1", "AAA", "###")])
    _write(right, [("@a/2", "TGCA", "!!!!"), ("@b/2", "TTT", "###")])
    s1, s2 = validate_pair(left, right)
    assert s1.read_count == s2.read_count == 2
    assert s1.pairing_id_mismatches == 0
    assert s1.malformed_records == s2.malformed_records == 0
    assert s1.quality_encoding == "Phred+33"


def test_pair_validation_detects_bad_record_and_mismatch(tmp_path):
    left, right = tmp_path / "x.1.fastq.gz", tmp_path / "x.2.fastq.gz"
    _write(left, [("@a/1", "ACGT", "!!!")])
    _write(right, [("@different/2", "TGCA", "!!!!")])
    s1, s2 = validate_pair(left, right)
    assert s1.malformed_records == 1
    assert s1.pairing_id_mismatches == s2.pairing_id_mismatches == 1

