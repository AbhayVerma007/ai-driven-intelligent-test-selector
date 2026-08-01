# TestWise AI — Selection Report

**Selected 9 / 9 tests (0.0% reduction in estimated runtime)**

| Tier | Count |
|---|---|
| Critical | 1 |
| High risk | 0 |
| Medium risk | 8 |
| Low risk (sampled) | 0 |

Estimated time: **9.0s** (vs 9.0s for the full suite)

## Selected Tests

- `tests/test_auth.py::test_login` — medium_risk (score=0.59) — medium risk (0.59) and covers changed code
- `tests/test_collector.py::test_validate_git_repo_raises_outside_repo` — medium_risk (score=0.50) — medium risk (0.50) and covers changed code
- `tests/test_collector.py::test_extract_diff_features_detects_config_and_api_changes` — medium_risk (score=0.50) — medium risk (0.50) and covers changed code
- `tests/test_collector.py::test_generate_synthetic_data_populates_db` — medium_risk (score=0.50) — medium risk (0.50) and covers changed code
- `tests/test_selector.py::test_critical_test_always_selected_even_if_scored_low` — critical (score=0.50) — critical/smoke test — always runs
- `tests/test_selector.py::test_max_skip_rate_cap_enforced` — medium_risk (score=0.50) — medium risk (0.50) and covers changed code
- `tests/test_selector.py::test_low_risk_minimum_sample_rate_enforced` — medium_risk (score=0.50) — medium risk (0.50) and covers changed code
- `tests/test_selector.py::test_audit_log_written_for_every_selection` — medium_risk (score=0.50) — medium risk (0.50) and covers changed code
- `tests/test_selector.py::test_generate_report_contains_summary` — medium_risk (score=0.50) — medium risk (0.50) and covers changed code