import gzip
from pathlib import Path
import pytest
from paper_atlas.public_release import audit, prepare


@pytest.mark.parametrize("text", ["sk-proj-" + "x" * 70, "ghp_" + "a" * 40,
                                 "-----BEGIN " + "PRIVATE KEY-----", "/home/" + "owner/private"])
def test_release_audit_redacts_findings(tmp_path, text):
    (tmp_path / "bad.txt").write_text(text)
    with pytest.raises(ValueError) as caught:
        audit(tmp_path)
    assert text not in str(caught.value)
    assert "bad.txt" in str(caught.value)


def test_release_audit_checks_compressed_data(tmp_path):
    (tmp_path / "data.json.gz").write_bytes(gzip.compress(b'{"secret":"ghp_' + b'a' * 40 + b'"}'))
    with pytest.raises(ValueError, match="api_token"):
        audit(tmp_path)


def test_release_no_overwrite(tmp_path):
    with pytest.raises(ValueError, match="already exists"):
        prepare(tmp_path, tmp_path, tmp_path)


def test_release_no_credentials_or_local_files(tmp_path):
    (tmp_path / "config.env").write_text("anything")
    with pytest.raises(ValueError, match="private_file_type"):
        audit(tmp_path)
