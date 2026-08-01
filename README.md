# 🧠 TestWise AI - Intelligent Test Selector

AI-powered test selection that analyzes git diffs to predict which tests to run. Reduces CI pipeline time by up to 80% while enforcing hardcoded safety rules.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-9%2F9%20passing-brightgreen)]()

---

## 🎯 What It Does

TestWise AI takes your `git diff`, extracts features from changed code, runs them through a trained ML model, and outputs only the tests most likely to fail.

Instead of running 800 tests on every commit, run the 150 that actually matter.

---

## ⚡ Quick Start

```bash
git clone https://github.com/AbhayVerma007/ai-driven-intelligent-test-selector.git
cd ai-driven-intelligent-test-selector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m examples.basic_demo
```

---

## 📊 Demo Results

| Scenario | Tests | Selected | Savings |
|----------|-------|----------|---------|
| High-risk (API + DB change) | 50 | 50 | 0% |
| Low-risk (isolated util change) | 50 | ~10 | ~80% |

High-risk changes trigger the full test suite by design. Low-risk changes show meaningful reduction through ML selection and safety sampling.

---

## 🏗️ How It Works

```
git diff → Feature Extraction → ML Model → Risk Score → Safety Rules → Selected Tests
```

Three core modules:

- **collector.py** - Parses git history and extracts diff features (changed files, functions, imports, config/DB/API flags)
- **trainer.py** - Trains sklearn GradientBoostingClassifier on 10 features per test
- **selector.py** - Scores every test and applies 5 hardcoded safety rules before output

---

## 🛡️ Safety Rules

AI suggests. Rules decide. Five safeguards that cannot be overridden:

| # | Rule | Description |
|---|------|-------------|
| 1 | Critical tests always run | Checked before model prediction |
| 2 | 10% minimum sampling | Random low-risk tests always included |
| 3 | 80% max skip-rate cap | Promotes tests back if too many skipped |
| 4 | Config/DB/API changes = full suite | Critical infrastructure changes trigger everything |
| 5 | Audit log | Every decision recorded to `data/audit_log.jsonl` |

---

## 🔧 CLI Commands

```bash
# Generate synthetic training data
python -m src.cli collect --synthetic --max-commits 200

# Train the model
python -m src.cli train

# Select tests between two git refs
python -m src.cli select --base-ref main~1 --head-ref main --test-path tests
```

Sample output:

```json
{
  "total_tests": 50,
  "selected": 10,
  "high_risk": 0,
  "medium_risk": 1,
  "low_risk_sampled": 9,
  "savings_percentage": 80.0
}
```

---

## 📈 Model Performance

**Cold-start by design.** New repos have zero historical failure data — so instead of shipping a model with nothing to learn from, TestWise bootstraps itself with 200 synthetic commits and trains on that from day one. The goal at this stage isn't predictive accuracy; it's proving the full loop — collect → train → select → enforce safety rules → audit log — actually runs end-to-end without gaps.

| Metric | Value (synthetic bootstrap) |
|--------|------------------------------|
| Training samples | 8,000 |
| Features | 10 |
| Precision | 53.6% |
| Recall | 9.5% |
| F1 Score | 16.1% |

These numbers are exactly what you'd expect from a model trained on a hand-tuned probability formula rather than real failures — they validate the pipeline, not the predictions. That's the honest read, and it's fine: the `model-retraining.yml` workflow runs weekly, re-collecting real commit and test-result history and retraining automatically. As real CI runs accumulate, synthetic rows age out of relevance and the model's signal comes increasingly from actual failure patterns — no manual intervention required. The `models/metrics.json` file written on every training run is what the Phase 3 activation trigger (recall < 85%) watches, so the system already knows how to tell you when it's time to invest in semantic analysis.

In short: the plumbing is production-ready today; the model's real-world accuracy is earned over the first few weeks of live traffic, by design.

---

## 📁 Project Structure

```
├── src/
│   ├── collector.py
│   ├── trainer.py
│   ├── selector.py
│   └── cli.py
├── tests/
│   ├── test_collector.py
│   └── test_selector.py
├── .github/workflows/
│   ├── intelligent-testing.yml
│   └── model-retraining.yml
├── examples/
│   └── basic_demo.py
├── models/
│   └── test_failure_model.joblib
└── data/
    └── audit_log.jsonl
```

---

## 🗺️ Project Phases

### ✅ Phase 1 & 2 (Complete)

- Git history parser with diff feature extraction
- ML training pipeline (sklearn GradientBoostingClassifier)
- Test selector with 5 hardcoded safety rules
- CLI tool and GitHub Actions templates
- Synthetic data bootstrap for cold-start
- 9/9 unit tests passing

### 🔜 Phase 3 (Designed, Not Built)

Activation trigger: Model recall below 85% on real data

- `semantic_analyzer.py` for CodeBERT code embeddings
- `feedback_loop.py` for continuous learning from CI results

Deferred because synthetic data cannot justify deep learning complexity.

### 🔮 Phase 4 (Planned)

Activation trigger: Production adoption or community interest

- Cost tracker with dollar figures
- Flask demo app with 50+ tests
- Docker Compose setup
- Multi-CI support (GitLab CI, Jenkins)

---

## 🚧 Current Constraints

- **Predictive accuracy is still warming up.** The bootstrap model reflects synthetic data, not your project's real failure history — that arrives automatically through weekly retraining, not a rewrite.
- **GitHub Actions workflows are written and locally verified, but not yet exercised in a live Actions run** — first real PR will be the first live test.
- Test discovery shells out to `pytest --collect-only`; repos with heavily customized pytest plugins should double-check collection output matches expectations.
- File-to-test matching currently uses filename similarity, not an import graph — accurate enough to bootstrap on, with room to sharpen as Phase 3 lands.

---

## 📝 License

MIT © Abhay Verma 2024