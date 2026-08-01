"""Command-line interface for TestWise AI: collect, train, select."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.collector import TestHistoryCollector
from src.selector import IntelligentTestSelector
from src.trainer import TestFailurePredictor

logger = logging.getLogger(__name__)


def cmd_collect(args: argparse.Namespace) -> None:
    collector = TestHistoryCollector(repo_path=args.repo_path, db_path=args.db_path)
    if args.synthetic:
        collector.generate_synthetic_data(n_commits=args.max_commits)
    else:
        collector.collect_commit_history(max_commits=args.max_commits)
    output_path = collector.export_to_json(output_path=args.output_json)
    print(f"Wrote training data to {output_path}")


def cmd_train(args: argparse.Namespace) -> None:
    predictor = TestFailurePredictor(model_path=args.model_path)
    data = predictor.load_data(json_path=args.history_json)
    matrix = predictor.build_feature_matrix(data)
    metrics = predictor.train(matrix)
    predictor.save_model()
    predictor.plot_feature_importance(output_path=args.feature_importance_path)
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to {predictor.model_path}")


def cmd_select(args: argparse.Namespace) -> None:
    collector = TestHistoryCollector(repo_path=args.repo_path, db_path=args.db_path)
    changed_files, diff_features = collector.get_diff_between_refs(args.base_ref, args.head_ref)

    selector = IntelligentTestSelector(
        model_path=args.model_path,
        history_path=args.history_json,
        audit_log_path=args.audit_log,
    )
    result = selector.select_tests(
        diff_features.to_dict(),
        changed_files=changed_files,
        test_path=args.test_path,
    )

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2))

    report = selector.generate_report(result)
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_report).write_text(report)

    print(json.dumps(result["selection_summary"], indent=2))
    print(f"Selected tests written to {args.output_json}")
    print(f"Report written to {args.output_report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="testwise", description="TestWise AI CLI")
    parser.add_argument("--repo-path", default=".")
    parser.add_argument("--db-path", default="data/test_history.db")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_collect = subparsers.add_parser("collect", help="Collect commit/test history")
    p_collect.add_argument("--max-commits", type=int, default=200)
    p_collect.add_argument("--synthetic", action="store_true", help="Generate synthetic bootstrap data")
    p_collect.add_argument("--output-json", default="data/test_history.json")
    p_collect.set_defaults(func=cmd_collect)

    p_train = subparsers.add_parser("train", help="Train the failure-prediction model")
    p_train.add_argument("--history-json", default="data/test_history.json")
    p_train.add_argument("--model-path", default="models/test_failure_model.joblib")
    p_train.add_argument("--feature-importance-path", default="models/feature_importance.png")
    p_train.set_defaults(func=cmd_train)

    p_select = subparsers.add_parser("select", help="Select tests for a diff")
    p_select.add_argument("--base-ref", required=True)
    p_select.add_argument("--head-ref", default="HEAD")
    p_select.add_argument("--test-path", default="tests")
    p_select.add_argument("--model-path", default="models/test_failure_model.joblib")
    p_select.add_argument("--history-json", default="data/test_history.json")
    p_select.add_argument("--audit-log", default="data/audit_log.jsonl")
    p_select.add_argument("--output-json", default="data/selected_tests.json")
    p_select.add_argument("--output-report", default="data/selection_report.md")
    p_select.set_defaults(func=cmd_select)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
