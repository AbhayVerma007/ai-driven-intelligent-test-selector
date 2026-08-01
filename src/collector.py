"""Git history and test-result collection for TestWise AI."""

from __future__ import annotations

import json
import logging
import random
import re
import sqlite3
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_FILE_PATTERNS = re.compile(r"\.(ya?ml|json|ini|cfg|toml|env)$|Dockerfile|docker-compose")
MIGRATION_PATTERNS = re.compile(r"migrat|alembic|schema", re.IGNORECASE)
API_PATTERNS = re.compile(r"route|endpoint|api|views?\.py|urls\.py|controller", re.IGNORECASE)
FUNC_DEF_PATTERN = re.compile(r"^\+\s*def\s+(\w+)\s*\(")
IMPORT_PATTERN = re.compile(r"^\+\s*(?:import|from)\s+([\w.]+)")


@dataclass
class DiffFeatures:
    lines_added: int = 0
    lines_deleted: int = 0
    functions_modified: list[str] = field(default_factory=list)
    imports_changed: list[str] = field(default_factory=list)
    file_types: list[str] = field(default_factory=list)
    has_config_change: bool = False
    has_db_migration: bool = False
    has_api_change: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "lines_added": self.lines_added,
            "lines_deleted": self.lines_deleted,
            "functions_modified": self.functions_modified,
            "imports_changed": self.imports_changed,
            "file_types": self.file_types,
            "has_config_change": self.has_config_change,
            "has_db_migration": self.has_db_migration,
            "has_api_change": self.has_api_change,
        }


class TestHistoryCollector:
    """Collects commit history, diff features, and test results for model training."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS commits (
        commit_hash TEXT PRIMARY KEY,
        timestamp TEXT,
        author TEXT,
        branch TEXT,
        pr_number TEXT,
        changed_files TEXT,
        diff_features TEXT
    );
    CREATE TABLE IF NOT EXISTS test_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        commit_hash TEXT,
        test_name TEXT,
        status TEXT,
        duration REAL,
        error_message TEXT,
        FOREIGN KEY (commit_hash) REFERENCES commits (commit_hash)
    );
    """

    def __init__(self, repo_path: str = ".", db_path: str = "data/test_history.db"):
        self.repo_path = Path(repo_path).resolve()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._validate_git_repo()
        self._init_db()

    def _validate_git_repo(self) -> None:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(
                f"ERROR: '{self.repo_path}' is not a git repository.\n"
                "TestHistoryCollector requires git history to operate.\n\n"
                "Fix it by running:\n"
                "  git init\n"
                "  git add .\n"
                '  git commit -m "Initial commit"\n',
                file=sys.stderr,
            )
            raise SystemExit(1)

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    # ------------------------------------------------------------------
    # Git log / diff parsing
    # ------------------------------------------------------------------

    def collect_commit_history(self, max_commits: int = 200) -> list[dict[str, Any]]:
        """Walk `git log` and extract commit metadata + diff features."""
        log_format = "%H|%an|%ae|%aI"
        result = subprocess.run(
            ["git", "log", f"-{max_commits}", f"--pretty=format:{log_format}"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        commits = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            commit_hash, author_name, author_email, timestamp = line.split("|", 3)
            changed_files = self._get_changed_files(commit_hash)
            diff_text = self._get_diff_for_commit(commit_hash)
            diff_features = self.extract_diff_features(diff_text, changed_files)
            record = {
                "commit": commit_hash,
                "timestamp": timestamp,
                "author": author_email or author_name,
                "branch": self._current_branch(),
                "pr_number": None,
                "changed_files": changed_files,
                "diff_features": diff_features.to_dict(),
            }
            commits.append(record)
            self._save_commit(record)
        return commits

    def _current_branch(self) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"

    def _get_changed_files(self, commit_hash: str) -> list[str]:
        result = subprocess.run(
            ["git", "show", "--pretty=", "--name-only", commit_hash],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        return [f for f in result.stdout.splitlines() if f.strip()]

    def get_diff_between_refs(self, base_ref: str, head_ref: str = "HEAD") -> tuple[list[str], DiffFeatures]:
        """Compute changed files + diff features between two git refs, e.g.
        for scoring a PR's full diff against its base branch."""
        files_result = subprocess.run(
            ["git", "diff", "--name-only", f"{base_ref}...{head_ref}"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        changed_files = [f for f in files_result.stdout.splitlines() if f.strip()]

        diff_result = subprocess.run(
            ["git", "diff", f"{base_ref}...{head_ref}"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        diff_features = self.extract_diff_features(diff_result.stdout, changed_files)
        return changed_files, diff_features

    def _get_diff_for_commit(self, commit_hash: str) -> str:
        result = subprocess.run(
            ["git", "show", commit_hash],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def extract_diff_features(self, diff_text: str, changed_files: list[str]) -> DiffFeatures:
        features = DiffFeatures()
        for raw_line in diff_text.splitlines():
            if raw_line.startswith("+") and not raw_line.startswith("+++"):
                features.lines_added += 1
                func_match = FUNC_DEF_PATTERN.match(raw_line)
                if func_match:
                    features.functions_modified.append(func_match.group(1))
                import_match = IMPORT_PATTERN.match(raw_line)
                if import_match:
                    features.imports_changed.append(import_match.group(1))
            elif raw_line.startswith("-") and not raw_line.startswith("---"):
                features.lines_deleted += 1

        features.file_types = sorted({Path(f).suffix for f in changed_files if Path(f).suffix})
        features.has_config_change = any(CONFIG_FILE_PATTERNS.search(f) for f in changed_files)
        features.has_db_migration = any(MIGRATION_PATTERNS.search(f) for f in changed_files)
        features.has_api_change = any(API_PATTERNS.search(f) for f in changed_files)
        return features

    # ------------------------------------------------------------------
    # Test execution
    # ------------------------------------------------------------------

    def run_test_suite(self, test_path: str = "tests") -> list[dict[str, Any]]:
        """Run pytest against `test_path` and capture per-test results via JUnit XML."""
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
            junit_path = tmp.name
        try:
            subprocess.run(
                [sys.executable, "-m", "pytest", test_path, f"--junitxml={junit_path}", "-q"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            return self._parse_junit_xml(junit_path)
        finally:
            Path(junit_path).unlink(missing_ok=True)

    def _parse_junit_xml(self, junit_path: str) -> list[dict[str, Any]]:
        results = []
        try:
            tree = ET.parse(junit_path)
        except ET.ParseError:
            return results
        for testcase in tree.getroot().iter("testcase"):
            name = f"{testcase.get('classname', '')}::{testcase.get('name', '')}"
            duration = float(testcase.get("time", 0.0))
            status = "passed"
            error_message = None
            failure = testcase.find("failure")
            error = testcase.find("error")
            skipped = testcase.find("skipped")
            if failure is not None:
                status, error_message = "failed", failure.get("message")
            elif error is not None:
                status, error_message = "error", error.get("message")
            elif skipped is not None:
                status = "skipped"
            results.append(
                {"test": name, "status": status, "duration": duration, "error_message": error_message}
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_commit(self, record: dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO commits
                   (commit_hash, timestamp, author, branch, pr_number, changed_files, diff_features)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["commit"],
                    record["timestamp"],
                    record["author"],
                    record["branch"],
                    record["pr_number"],
                    json.dumps(record["changed_files"]),
                    json.dumps(record["diff_features"]),
                ),
            )

    def save_test_results(self, commit_hash: str, test_results: list[dict[str, Any]]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """INSERT INTO test_results (commit_hash, test_name, status, duration, error_message)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (commit_hash, r["test"], r["status"], r["duration"], r.get("error_message"))
                    for r in test_results
                ],
            )

    def export_to_json(self, output_path: str = "data/test_history.json") -> str:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            commits = [dict(row) for row in conn.execute("SELECT * FROM commits")]
            results = [dict(row) for row in conn.execute("SELECT * FROM test_results")]

        by_commit: dict[str, list[dict[str, Any]]] = {}
        for r in results:
            by_commit.setdefault(r["commit_hash"], []).append(
                {"test": r["test_name"], "status": r["status"], "duration": r["duration"]}
            )

        records = []
        for c in commits:
            records.append(
                {
                    "commit": c["commit_hash"],
                    "timestamp": c["timestamp"],
                    "author": c["author"],
                    "changed_files": json.loads(c["changed_files"]),
                    "diff_features": json.loads(c["diff_features"]),
                    "test_results": by_commit.get(c["commit_hash"], []),
                }
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(records, indent=2))
        return output_path

    # ------------------------------------------------------------------
    # Synthetic bootstrap data (cold-start)
    # ------------------------------------------------------------------

    def generate_synthetic_data(self, n_commits: int = 200, n_tests: int = 40, seed: int = 42) -> int:
        """Populate the DB with synthetic commit/test-result history so the model
        has something to train on before enough real CI runs have accumulated."""
        rng = random.Random(seed)
        test_names = [f"tests/test_module_{i // 8}.py::test_case_{i}" for i in range(n_tests)]
        critical_tests = set(rng.sample(test_names, k=max(1, n_tests // 10)))
        authors = ["alice@example.com", "bob@example.com", "carol@example.com"]
        file_pool = [
            "src/auth.py", "src/models.py", "src/api/routes.py", "src/utils.py",
            "src/db/migrations/0001_init.py", "config/settings.yaml", "src/collector.py",
        ]
        base_time = datetime.now(timezone.utc) - timedelta(days=n_commits)

        with sqlite3.connect(self.db_path) as conn:
            for i in range(n_commits):
                commit_hash = f"synthetic{i:06x}"
                timestamp = (base_time + timedelta(hours=i * 6)).isoformat()
                author = rng.choice(authors)
                changed_files = rng.sample(file_pool, k=rng.randint(1, 3))
                features = DiffFeatures(
                    lines_added=rng.randint(1, 120),
                    lines_deleted=rng.randint(0, 60),
                    functions_modified=[f"func_{rng.randint(0, 20)}"],
                    imports_changed=[],
                    file_types=sorted({Path(f).suffix for f in changed_files}),
                    has_config_change=any("config" in f or f.endswith((".yaml", ".yml")) for f in changed_files),
                    has_db_migration=any("migrat" in f for f in changed_files),
                    has_api_change=any("api" in f or "routes" in f for f in changed_files),
                )
                conn.execute(
                    """INSERT OR REPLACE INTO commits
                       (commit_hash, timestamp, author, branch, pr_number, changed_files, diff_features)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (commit_hash, timestamp, author, "main", None,
                     json.dumps(changed_files), json.dumps(features.to_dict())),
                )

                # riskier commits (config/db/api changes, big churn) fail more tests
                risk_bump = 0.0
                if features.has_config_change:
                    risk_bump += 0.15
                if features.has_db_migration:
                    risk_bump += 0.2
                if features.has_api_change:
                    risk_bump += 0.1
                if features.lines_added + features.lines_deleted > 100:
                    risk_bump += 0.1

                rows = []
                for test_name in test_names:
                    base_fail_rate = 0.25 if test_name in critical_tests else 0.08
                    fail_prob = min(0.9, base_fail_rate + risk_bump)
                    status = "failed" if rng.random() < fail_prob else "passed"
                    duration = round(rng.uniform(0.05, 3.0), 3)
                    rows.append((commit_hash, test_name, status, duration, None))
                conn.executemany(
                    """INSERT INTO test_results (commit_hash, test_name, status, duration, error_message)
                       VALUES (?, ?, ?, ?, ?)""",
                    rows,
                )
        logger.info("Generated %d synthetic commits x %d tests", n_commits, n_tests)
        return n_commits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = TestHistoryCollector()
    collector.generate_synthetic_data(n_commits=200)
    path = collector.export_to_json()
    print(f"Synthetic training data written to {path}")
