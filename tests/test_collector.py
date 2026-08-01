from pathlib import Path

import pytest

from src.collector import DiffFeatures, TestHistoryCollector


@pytest.fixture
def collector(tmp_path):
    db_path = tmp_path / "history.db"
    return TestHistoryCollector(repo_path=".", db_path=str(db_path))


def test_validate_git_repo_raises_outside_repo(tmp_path):
    non_repo = tmp_path / "not_a_repo"
    non_repo.mkdir()
    with pytest.raises(SystemExit):
        TestHistoryCollector(repo_path=str(non_repo), db_path=str(tmp_path / "history.db"))


def test_extract_diff_features_detects_config_and_api_changes(collector):
    diff_text = "+def login():\n+    return True\n+import jwt\n"
    changed_files = ["config/settings.yaml", "src/api/routes.py"]
    features = collector.extract_diff_features(diff_text, changed_files)

    assert isinstance(features, DiffFeatures)
    assert features.lines_added == 3
    assert "login" in features.functions_modified
    assert "jwt" in features.imports_changed
    assert features.has_config_change is True
    assert features.has_api_change is True
    assert features.has_db_migration is False


def test_generate_synthetic_data_populates_db(collector):
    n = collector.generate_synthetic_data(n_commits=5, n_tests=4, seed=1)
    assert n == 5

    export_path = collector.export_to_json(output_path=str(Path(collector.db_path).parent / "history.json"))
    import json

    records = json.loads(Path(export_path).read_text())
    assert len(records) == 5
    assert all(r["test_results"] for r in records)
