"""End-to-end TestWise AI demo: bootstrap data -> train -> select.

Run from the repo root:
    python -m examples.basic_demo
"""

from __future__ import annotations

import json
from pathlib import Path

from src.collector import TestHistoryCollector
from src.selector import IntelligentTestSelector
from src.trainer import TestFailurePredictor

DEMO_DIR = Path("data/demo")


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DEMO_DIR / "history.db"
    history_path = DEMO_DIR / "history.json"
    model_path = DEMO_DIR / "model.joblib"
    audit_path = DEMO_DIR / "audit_log.jsonl"

    print("1. Bootstrapping 200 synthetic commits (cold-start data)...")
    collector = TestHistoryCollector(repo_path=".", db_path=str(db_path))
    collector.generate_synthetic_data(n_commits=200, n_tests=50)
    collector.export_to_json(output_path=str(history_path))

    print("2. Training the failure-prediction model...")
    predictor = TestFailurePredictor(model_path=str(model_path))
    data = predictor.load_data(json_path=str(history_path))
    matrix = predictor.build_feature_matrix(data)
    metrics = predictor.train(matrix)
    predictor.save_model()
    print(f"   Metrics: {json.dumps(metrics, indent=2)}")

    selector = IntelligentTestSelector(
        model_path=str(model_path),
        history_path=str(history_path),
        audit_log_path=str(audit_path),
    )
    # Reuse the synthetic test names as the "discoverable" suite for this demo,
    # since no real pytest suite of that size exists in this repo.
    demo_test_names = sorted(selector._tracker.known_tests())
    selector.discover_tests = lambda test_path="tests": demo_test_names  # type: ignore[method-assign]

    scenarios = {
        "high_risk (API + DB migration change)": (
            {"lines_added": 80, "lines_deleted": 20, "has_config_change": False,
             "has_db_migration": True, "has_api_change": True},
            ["src/api/routes.py", "src/db/migrations/0002_add_field.py"],
        ),
        "low_risk (small isolated util tweak)": (
            {"lines_added": 4, "lines_deleted": 1, "has_config_change": False,
             "has_db_migration": False, "has_api_change": False},
            ["src/utils.py"],
        ),
    }

    for label, (diff_features, changed_files) in scenarios.items():
        print(f"3. Selecting tests for scenario: {label}...")
        result = selector.select_tests(diff_features, changed_files=changed_files, author="alice@example.com")
        print(json.dumps(result["selection_summary"], indent=2))

        report_path = DEMO_DIR / f"selection_report_{label.split()[0]}.md"
        report_path.write_text(selector.generate_report(result))
        print(f"   Report written to {report_path}\n")


if __name__ == "__main__":
    main()
