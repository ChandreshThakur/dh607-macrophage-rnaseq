import hashlib

from data_validation.audit import md5sum


def test_md5sum_known_input(tmp_path):
    path = tmp_path / "known.bin"
    path.write_bytes(b"DH607\n")
    assert md5sum(path) == hashlib.md5(b"DH607\n").hexdigest()

