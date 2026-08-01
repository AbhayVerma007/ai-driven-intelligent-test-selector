# TestWise AI — Review & Implementation Plan

## 1. Summary of the Ask

Build "TestWise AI": a system that reads git diffs, predicts which tests are likely
to fail using a trained ML model, and selects a reduced test subset for CI —
targeting >70% CI time reduction while keeping bug-catch rate >98%. Scope includes:
data collection, model training, a selector with tiered risk logic, semantic
analysis via CodeBERT, a feedback/retraining loop, cost tracking, GitHub Actions
integration, a Flask demo app, Docker packaging, notebooks, and a polished README.

This is a full product build (~20+ files across ML, CI/CD, and app layers), not a
small task. Below is a review of the request as given, followed by a phased plan.

## 2. Review Notes / Risks

- **Cold-start problem**: `TestFailurePredictor` needs historical commit+test-result
  data to train on. A brand-new repo (like this one) has none. The plan must include
  a synthetic/bootstrap dataset generator or the model has nothing to learn from until
  many real CI runs accumulate.
- **CodeBERT cost**: `transformers` + `torch` + `microsoft/codebert-base` is a heavy
  dependency (multi-GB download, slow CPU inference). Spec itself says "use CodeBERT
  only if model accuracy < 85%" — so semantic_analyzer.py should be implemented but
  gated/optional, not on the critical path for MVP.
- **Multi-output prediction ambiguity**: "one model per test suite or single model
  with test labels" — for an MVP, a single model with (change-features × test-features)
  → failure-probability rows is simpler and more general than N per-suite models.
  Plan will default to that unless told otherwise.
- **GitHub Actions workflows can't be verified locally** — they'll be written to spec
  but only truly validated once pushed to an actual GitHub repo with Actions enabled.
- **Safety requirements** (never skip critical/smoke tests, always sample 10% of
  low-risk, audit log, override switch) are correctness-critical and will be enforced
  in `selector.py` directly, not left as documentation-only promises.
- **No git repo yet** in this directory — needed for `collector.py` to have real
  commit history to parse. Demo/bootstrap data will stand in until real history exists.
- **Large dependency footprint** (torch, xgboost, mlflow, fastapi) — Docker setup
  matters more than local venv for reproducibility; will keep requirements honest
  about what's actually imported by MVP code vs. optional extras.

## 3. Proposed Phasing (matches spec's priority order)

**Phase 1 — Core (MVP, no ML dependencies beyond sklearn)**
- `src/collector.py` — TestHistoryCollector (git log parsing, diff feature extraction, pytest result capture, JSON/SQLite storage)
- `src/trainer.py` — TestFailurePredictor (feature matrix, GradientBoostingClassifier, joblib persistence, CV metrics)
- `src/selector.py` — IntelligentTestSelector (tiered risk selection, safety nets, JSON + Markdown report output)
- Basic unit tests for each (`tests/test_collector.py`, `tests/test_selector.py`)
- `requirements.txt`, `setup.py`, project skeleton per the file tree

**Phase 2 — Integration**
- `.github/workflows/intelligent-testing.yml` and `model-retraining.yml`
- Simple CLI entrypoint to run collection/train/select locally
- `examples/basic_demo.py`

**Phase 3 — Intelligence**
- `src/semantic_analyzer.py` — CodeBERT embeddings, AST-based function extraction, test↔source mapping (optional/gated dependency)
- `src/feedback_loop.py` — ContinuousLearner (buffer, drift detection, retrain trigger, accuracy tracking)

**Phase 4 — Polish**
- `src/cost_tracker.py` — savings math, executive report, charts
- `examples/flask_app_demo/` — Flask CRUD app + 50+ tests + preloaded history + demo showing the reduction
- `Dockerfile`, `docker-compose.yml` (app + Redis, optional MLflow)
- `notebooks/01_exploratory_analysis.ipynb`, `02_model_training_demo.ipynb`
- README with architecture diagram, badges, benchmarks table, quick start

## 4. Open Questions Before Implementation

1. Should this repo be `git init`-ed now so `collector.py` has real commits to work
   against as we build, or do we generate synthetic training data only?
2. XGBoost vs GradientBoostingClassifier (sklearn) for the actual MVP model — spec
   lists both; sklearn has fewer install headaches and is the stated MVP default.
3. Is CodeBERT integration actually needed for the first working version, or should
   Phase 3 stay stubbed/optional given the dependency weight and the spec's own
   "only if accuracy < 85%" condition?
4. Any target Python version / CI provider constraints (spec assumes GitHub Actions
   only — confirm that's the only CI target, no GitLab/Jenkins needed for v1)?

## 5. Next Step

Awaiting go-ahead to start Phase 1. Once confirmed, will scaffold the full directory
tree and implement `collector.py`, `trainer.py`, `selector.py` first, with real unit
tests, before touching CI workflows or the demo app.
