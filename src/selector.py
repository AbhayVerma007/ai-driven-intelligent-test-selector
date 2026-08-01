"""Risk-tiered intelligent test selection for TestWise AI."""

from __future__ import annotations

import json
import logging
import random
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.trainer import FEATURE_COLUMNS, TestFailurePredictor, TestStatsTracker, build_feature_row

logger = logging.getLogger(__name__)

DEFAULT_CRITICAL_PATTERN = re.compile(r"critical|smoke", re.IGNORECASE)


@dataclass
class SelectionConfig:
    high_risk_threshold: float = 0.7
    medium_risk_threshold: float = 0.3
    low_risk_sample_rate: float = 0.10
    max_skip_rate: float = 0.80
    critical_test_names: set[str] = field(default_factory=set)
    critical_test_pattern: re.Pattern = DEFAULT_CRITICAL_PATTERN


class IntelligentTestSelector:
    """Scores tests for failure risk against a code diff and selects a
    reduced test set, enforcing hard safety rules at every step:

      1. Critical/smoke tests always run.
      2. At least `low_risk_sample_rate` of low-risk tests are sampled.
      3. No more than `max_skip_rate` of the full suite is ever skipped.
      4. Config/DB-migration/API changes trigger extra safety-net tests.
      5. Every skip decision is written to an audit log.
    """

    def __init__(
        self,
        model_path: str = "models/test_failure_model.joblib",
        history_path: str = "data/test_history.json",
        audit_log_path: str = "data/audit_log.jsonl",
        config: SelectionConfig | None = None,
    ):
        self.predictor = TestFailurePredictor(model_path)
        self.predictor.load_model()
        self.history_path = Path(history_path)
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config or SelectionConfig()
        self._tracker, self._avg_duration = self._replay_history()

    # ------------------------------------------------------------------
    # History replay (rebuilds the same running stats used at train time)
    # ------------------------------------------------------------------

    def _replay_history(self) -> tuple[TestStatsTracker, dict[str, float]]:
        tracker = TestStatsTracker()
        durations: dict[str, list[float]] = {}
        if not self.history_path.exists():
            return tracker, {}
        records = json.loads(self.history_path.read_text())
        records.sort(key=lambda r: r["timestamp"])
        for record in records:
            author = record["author"]
            for test_result in record["test_results"]:
                test_name = test_result["test"]
                failed = test_result["status"] == "failed"
                durations.setdefault(test_name, []).append(test_result.get("duration", 0.0))
                tracker.update(author, test_name, failed)
            tracker.advance_commit()
        avg_duration = {name: sum(vals) / len(vals) for name, vals in durations.items()}
        return tracker, avg_duration

    # ------------------------------------------------------------------
    # Test discovery
    # ------------------------------------------------------------------

    def discover_tests(self, test_path: str = "tests") -> list[str]:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_path, "--collect-only", "-q"],
            capture_output=True,
            text=True,
        )
        tests = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if "::" in line and not line.startswith(("=", "no tests")):
                tests.append(line)
        return tests or self._tracker.known_tests()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_tests(
        self,
        diff_features: dict[str, Any],
        changed_files: list[str],
        test_names: list[str],
        author: str = "unknown",
    ) -> dict[str, float]:
        rows = []
        for test_name in test_names:
            hist_rate, time_since_fail, dev_rate = self._tracker.current_features(author, test_name)
            rows.append(build_feature_row(diff_features, changed_files, test_name, hist_rate, time_since_fail, dev_rate))
        df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        probs = self.predictor.predict_proba(df)
        return dict(zip(test_names, probs))

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def is_critical(self, test_name: str) -> bool:
        return (
            test_name in self.config.critical_test_names
            or bool(self.config.critical_test_pattern.search(test_name))
        )

    def select_tests(
        self,
        diff_features: dict[str, Any],
        changed_files: list[str],
        test_path: str = "tests",
        author: str = "unknown",
        rng_seed: int | None = None,
    ) -> dict[str, Any]:
        rng = random.Random(rng_seed)
        test_names = self.discover_tests(test_path)
        if not test_names:
            raise RuntimeError(f"No tests discovered under '{test_path}'.")

        scores = self.score_tests(diff_features, changed_files, test_names, author)

        extra_safety_triggered = any(
            diff_features.get(k) for k in ("has_config_change", "has_db_migration", "has_api_change")
        )

        decisions: dict[str, dict[str, Any]] = {}
        low_risk_tests = []

        for test_name in test_names:
            score = scores[test_name]
            file_overlap = build_feature_row(diff_features, changed_files, test_name, 0, 0, 0)["file_overlap_score"]

            if self.is_critical(test_name):
                decisions[test_name] = {"run": True, "tier": "critical", "score": score, "reason": "critical/smoke test — always runs"}
            elif score > self.config.high_risk_threshold:
                decisions[test_name] = {"run": True, "tier": "high_risk", "score": score, "reason": f"failure probability {score:.2f} > {self.config.high_risk_threshold}"}
            elif score >= self.config.medium_risk_threshold:
                covers_change = file_overlap > 0 or extra_safety_triggered
                decisions[test_name] = {
                    "run": covers_change,
                    "tier": "medium_risk",
                    "score": score,
                    "reason": (
                        f"medium risk ({score:.2f}) and {'covers changed code' if file_overlap > 0 else 'extra safety net (config/db/api change)'}"
                        if covers_change
                        else f"medium risk ({score:.2f}) but no file overlap with changed code — skipped"
                    ),
                }
            else:
                low_risk_tests.append(test_name)
                decisions[test_name] = {"run": False, "tier": "low_risk", "score": score, "reason": f"low risk ({score:.2f}) — pending 10% sample"}

        # Rule 2: minimum 10% sample of low-risk tests
        if low_risk_tests:
            n_sample = max(1, round(len(low_risk_tests) * self.config.low_risk_sample_rate))
            sampled = set(rng.sample(low_risk_tests, k=min(n_sample, len(low_risk_tests))))
            for test_name in sampled:
                decisions[test_name]["run"] = True
                decisions[test_name]["reason"] += " — selected in random safety sample"

        # Rule 3: max skip rate cap — never skip more than max_skip_rate of the suite
        total = len(test_names)
        skipped = [t for t, d in decisions.items() if not d["run"]]
        max_skippable = int(total * self.config.max_skip_rate)
        if len(skipped) > max_skippable:
            need_to_promote = len(skipped) - max_skippable
            skipped_sorted = sorted(skipped, key=lambda t: decisions[t]["score"], reverse=True)
            for test_name in skipped_sorted[:need_to_promote]:
                decisions[test_name]["run"] = True
                decisions[test_name]["reason"] += " — promoted to satisfy max 80% skip-rate cap"

        # Rule 5: audit log every decision (especially skips)
        self._write_audit_log(decisions)

        selected = [t for t, d in decisions.items() if d["run"]]
        tier_counts = {
            "high_risk": sum(1 for d in decisions.values() if d["run"] and d["tier"] == "high_risk"),
            "medium_risk": sum(1 for d in decisions.values() if d["run"] and d["tier"] == "medium_risk"),
            "low_risk_sampled": sum(1 for d in decisions.values() if d["run"] and d["tier"] == "low_risk"),
            "critical": sum(1 for d in decisions.values() if d["run"] and d["tier"] == "critical"),
        }

        original_time = sum(self._avg_duration.get(t, 1.0) for t in test_names)
        estimated_time = sum(self._avg_duration.get(t, 1.0) for t in selected)
        savings_pct = round(100 * (1 - estimated_time / original_time), 1) if original_time else 0.0

        result = {
            "selected_tests": selected,
            "selection_summary": {
                "total_tests": total,
                "selected": len(selected),
                **tier_counts,
                "estimated_time_seconds": round(estimated_time, 2),
                "original_time_seconds": round(original_time, 2),
                "savings_percentage": savings_pct,
            },
            "decisions": decisions,
        }
        return result

    def _write_audit_log(self, decisions: dict[str, dict[str, Any]]) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decisions": decisions,
        }
        with self.audit_log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self, result: dict[str, Any]) -> str:
        s = result["selection_summary"]
        lines = [
            "# TestWise AI — Selection Report",
            "",
            f"**Selected {s['selected']} / {s['total_tests']} tests "
            f"({s['savings_percentage']}% reduction in estimated runtime)**",
            "",
            "| Tier | Count |",
            "|---|---|",
            f"| Critical | {s['critical']} |",
            f"| High risk | {s['high_risk']} |",
            f"| Medium risk | {s['medium_risk']} |",
            f"| Low risk (sampled) | {s['low_risk_sampled']} |",
            "",
            f"Estimated time: **{s['estimated_time_seconds']}s** "
            f"(vs {s['original_time_seconds']}s for the full suite)",
            "",
            "## Selected Tests",
            "",
        ]
        for test_name in result["selected_tests"]:
            d = result["decisions"][test_name]
            lines.append(f"- `{test_name}` — {d['tier']} (score={d['score']:.2f}) — {d['reason']}")
        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    selector = IntelligentTestSelector()
    diff_features = {
        "lines_added": 40,
        "lines_deleted": 5,
        "has_config_change": False,
        "has_db_migration": False,
        "has_api_change": True,
    }
    result = selector.select_tests(diff_features, changed_files=["src/auth.py"], test_path="tests")
    print(json.dumps(result["selection_summary"], indent=2))
    Path("data/selection_report.md").write_text(selector.generate_report(result))
