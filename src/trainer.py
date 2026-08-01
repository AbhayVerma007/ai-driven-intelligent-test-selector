"""Model training for TestWise AI test-failure prediction."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "lines_added",
    "lines_deleted",
    "num_changed_files",
    "has_config_change",
    "has_db_migration",
    "has_api_change",
    "file_overlap_score",
    "historical_failure_rate",
    "time_since_last_failure",
    "developer_failure_rate",
]


def _test_module_stem(test_name: str) -> str:
    """Extract a comparable module name from a pytest node id, e.g.
    'tests/test_auth.py::test_login' -> 'auth'."""
    file_part = test_name.split("::")[0]
    stem = Path(file_part).stem
    return stem[len("test_"):] if stem.startswith("test_") else stem


def compute_file_overlap(test_name: str, changed_files: list[str]) -> float:
    """Similarity between the test's target module and the changed files, 0..1."""
    if not changed_files:
        return 0.0
    test_stem = _test_module_stem(test_name)
    best = 0.0
    for f in changed_files:
        file_stem = Path(f).stem
        ratio = SequenceMatcher(None, test_stem, file_stem).ratio()
        best = max(best, ratio)
    return round(best, 4)


def build_feature_row(
    diff_features: dict[str, Any],
    changed_files: list[str],
    test_name: str,
    historical_failure_rate: float,
    time_since_last_failure: int,
    developer_failure_rate: float,
) -> dict[str, float]:
    """Build one feature row shared between training (trainer.py) and
    inference (selector.py) so features never drift between the two."""
    return {
        "lines_added": diff_features.get("lines_added", 0),
        "lines_deleted": diff_features.get("lines_deleted", 0),
        "num_changed_files": len(changed_files),
        "has_config_change": int(diff_features.get("has_config_change", False)),
        "has_db_migration": int(diff_features.get("has_db_migration", False)),
        "has_api_change": int(diff_features.get("has_api_change", False)),
        "file_overlap_score": compute_file_overlap(test_name, changed_files),
        "historical_failure_rate": historical_failure_rate,
        "time_since_last_failure": time_since_last_failure,
        "developer_failure_rate": developer_failure_rate,
    }


class TestStatsTracker:
    """Maintains running (leakage-free) per-test and per-developer failure
    statistics as commits are replayed in chronological order. Used both to
    build training labels and, after replay, to score a brand-new diff."""

    def __init__(self) -> None:
        self._test_stats: dict[str, dict[str, int | None]] = defaultdict(
            lambda: {"runs": 0, "fails": 0, "last_fail_index": None}
        )
        self._dev_stats: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"runs": 0, "fails": 0}
        )
        self._index = 0

    def current_features(self, author: str, test_name: str) -> tuple[float, int, float]:
        s = self._test_stats[test_name]
        hist_fail_rate = s["fails"] / s["runs"] if s["runs"] else 0.0
        time_since_last_failure = (
            self._index - s["last_fail_index"] if s["last_fail_index"] is not None else self._index + 1
        )
        d = self._dev_stats[(author, test_name)]
        dev_fail_rate = d["fails"] / d["runs"] if d["runs"] else 0.0
        return hist_fail_rate, time_since_last_failure, dev_fail_rate

    def update(self, author: str, test_name: str, failed: bool) -> None:
        s = self._test_stats[test_name]
        s["runs"] += 1
        if failed:
            s["fails"] += 1
            s["last_fail_index"] = self._index
        d = self._dev_stats[(author, test_name)]
        d["runs"] += 1
        if failed:
            d["fails"] += 1

    def advance_commit(self) -> None:
        self._index += 1

    def known_tests(self) -> list[str]:
        return list(self._test_stats.keys())


class TestFailurePredictor:
    """Trains and serves a GradientBoostingClassifier that predicts, for a
    given (code change, test) pair, the probability the test fails."""

    def __init__(self, model_path: str = "models/test_failure_model.joblib"):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model: GradientBoostingClassifier | None = None

    def load_data(self, json_path: str = "data/test_history.json") -> list[dict[str, Any]]:
        records = json.loads(Path(json_path).read_text())
        records.sort(key=lambda r: r["timestamp"])
        return records

    def build_feature_matrix(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        tracker = TestStatsTracker()
        rows = []
        for record in records:
            changed_files = record["changed_files"]
            diff_features = record["diff_features"]
            author = record["author"]
            for test_result in record["test_results"]:
                test_name = test_result["test"]
                failed = test_result["status"] == "failed"
                hist_rate, time_since_fail, dev_rate = tracker.current_features(author, test_name)
                row = build_feature_row(
                    diff_features, changed_files, test_name, hist_rate, time_since_fail, dev_rate
                )
                row["label"] = int(failed)
                rows.append(row)
                tracker.update(author, test_name, failed)
            tracker.advance_commit()
        return pd.DataFrame(rows)

    def train(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict[str, Any]:
        X, y = df[FEATURE_COLUMNS], df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.model = GradientBoostingClassifier(random_state=random_state)
        self.model.fit(X_train, y_train)

        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring="f1")
        y_pred = self.model.predict(X_test)

        metrics = {
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        logger.info("Training complete: %s", metrics)
        logger.info("\n%s", classification_report(y_test, y_pred, zero_division=0))
        return metrics

    def save_model(self) -> str:
        if self.model is None:
            raise RuntimeError("No trained model to save; call train() first.")
        joblib.dump(self.model, self.model_path)
        return str(self.model_path)

    def load_model(self) -> GradientBoostingClassifier:
        self.model = joblib.load(self.model_path)
        return self.model

    def plot_feature_importance(self, output_path: str = "models/feature_importance.png") -> str:
        if self.model is None:
            raise RuntimeError("No trained model; call train() first.")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        importances = self.model.feature_importances_
        order = importances.argsort()[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([FEATURE_COLUMNS[i] for i in order], importances[order])
        ax.set_xlabel("Importance")
        ax.set_title("TestWise AI — Feature Importance")
        fig.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    def predict_proba(self, feature_rows: pd.DataFrame) -> list[float]:
        model = self.model or self.load_model()
        return model.predict_proba(feature_rows[FEATURE_COLUMNS])[:, 1].tolist()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    predictor = TestFailurePredictor()
    data = predictor.load_data()
    matrix = predictor.build_feature_matrix(data)
    predictor.train(matrix)
    predictor.save_model()
    predictor.plot_feature_importance()
    print(f"Model saved to {predictor.model_path}")
