"""Resource and storage independence from the original training repository."""
import importlib
import json
from pathlib import Path

from paper_atlas.conference_acceptance import BASELINES
from paper_atlas.conference_survey import DEFAULT_TOPICS
from paper_atlas.conference_web import ASSETS


def test_packaged_resources():
    assert json.loads(BASELINES.read_text())["rows"]
    assert json.loads(DEFAULT_TOPICS.read_text())["topics"]
    for name in ("index.html", "app.js", "style.css", "compact.css", "algorithm.html", "algorithm.js"):
        assert (ASSETS/name).is_file()
    assert BASELINES.parent.parent == ASSETS.parent


def test_independent_storage_environment(monkeypatch):
    import paper_atlas.settings as settings
    with monkeypatch.context() as patch:
        patch.delenv("PAPER_ATLAS_DATA_DIR", raising=False)
        patch.setenv("TINY_GPT_DATAROOT", "/not-the-paper-atlas-data")
        assert importlib.reload(settings).DATA_DIR == "./data"
        patch.setenv("PAPER_ATLAS_DATA_DIR", "/paper-atlas-test-data")
        assert importlib.reload(settings).DATA_DIR == "/paper-atlas-test-data"
    importlib.reload(settings)


def test_no_original_package_imports():
    import paper_atlas
    for path in Path(paper_atlas.__file__).parent.glob("*.py"):
        text = path.read_text()
        assert "from tiny_gpt" not in text and "import tiny_gpt" not in text
        assert "from global_setting" not in text
