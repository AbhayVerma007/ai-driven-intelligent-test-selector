import json

import pytest

from src.collector import TestHistoryCollector
from src.selector import IntelligentTestSelector, SelectionConfig
from src.trainer import TestFailurePredictor


@pytest.fixture
def selector(tmp_path):
    db_path = tmp_path / "history.db"
    history_path = tmp_path / "history.json"
    model_path = tmp_path / "model.joblib"
    audit_path = tmp_path / "audit_log.jsonl"

    collector = TestHistoryCollector(repo_path=".", db_path=str(db_path))
    collector.generate_synthetic_data(n_commits=20, n_tests=6, seed=7)
    collector.export_to_json(output_path=str(history_path))

    predictor = TestFailurePredictor(model_path=str(model_path))
    data = predictor.load_data(json_path=str(history_path))
    matrix = predictor.build_feature_matrix(data)
    predictor.train(matrix)
    predictor.save_model()

    return IntelligentTestSelector(
        model_path=str(model_path),
        history_path=str(history_path),
        audit_log_path=str(audit_path),
        config=SelectionConfig(critical_test_names={"tests/test_module_0.py::test_case_0"}),
    )


def test_critical_test_always_selected_even_if_scored_low(selector, monkeypatch):
    test_names = ["tests/test_module_0.py::test_case_0", "tests/test_module_0.py::test_case_1"]
    monkeypatch.setattr(selector, "discover_tests", lambda test_path="tests": test_names)
    monkeypatch.setattr(selector, "score_tests", lambda *a, **k: {t: 0.01 for t in test_names})

    result = selector.select_tests(diff_features={"lines_added": 1, "lines_deleted": 0}, changed_files=[])

    assert "tests/test_module_0.py::test_case_0" in result["selected_tests"]
    assert result["decisions"]["tests/test_module_0.py::test_case_0"]["tier"] == "critical"


def test_max_skip_rate_cap_enforced(selector, monkeypatch):
    test_names = [f"tests/test_module_1.py::test_case_{i}" for i in range(20)]
    monkeypatch.setattr(selector, "discover_tests", lambda test_path="tests": test_names)
    monkeypatch.setattr(selector, "score_tests", lambda *a, **k: {t: 0.01 for t in test_names})

    result = selector.select_tests(diff_features={"lines_added": 1, "lines_deleted": 0}, changed_files=[], rng_seed=1)

    skip_rate = 1 - (result["selection_summary"]["selected"] / result["selection_summary"]["total_tests"])
    assert skip_rate <= selector.config.max_skip_rate


def test_low_risk_minimum_sample_rate_enforced(selector, monkeypatch):
    test_names = [f"tests/test_module_2.py::test_case_{i}" for i in range(30)]
    monkeypatch.setattr(selector, "discover_tests", lambda test_path="tests": test_names)
    monkeypatch.setattr(selector, "score_tests", lambda *a, **k: {t: 0.01 for t in test_names})

    result = selector.select_tests(diff_features={"lines_added": 1, "lines_deleted": 0}, changed_files=[], rng_seed=2)

    low_risk_selected = result["selection_summary"]["low_risk_sampled"]
    assert low_risk_selected >= round(len(test_names) * selector.config.low_risk_sample_rate)


def test_audit_log_written_for_every_selection(selector, monkeypatch, tmp_path):
    test_names = ["tests/test_module_3.py::test_case_0"]
    monkeypatch.setattr(selector, "discover_tests", lambda test_path="tests": test_names)
    monkeypatch.setattr(selector, "score_tests", lambda *a, **k: {t: 0.5 for t in test_names})

    selector.select_tests(diff_features={"lines_added": 1, "lines_deleted": 0}, changed_files=[])

    lines = selector.audit_log_path.read_text().strip().splitlines()
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert "decisions" in entry and test_names[0] in entry["decisions"]


def test_generate_report_contains_summary(selector, monkeypatch):
    test_names = ["tests/test_module_0.py::test_case_0"]
    monkeypatch.setattr(selector, "discover_tests", lambda test_path="tests": test_names)
    monkeypatch.setattr(selector, "score_tests", lambda *a, **k: {t: 0.9 for t in test_names})

    result = selector.select_tests(diff_features={"lines_added": 1, "lines_deleted": 0}, changed_files=[])
    report = selector.generate_report(result)

    assert "TestWise AI" in report
    assert "Selected Tests" in report
