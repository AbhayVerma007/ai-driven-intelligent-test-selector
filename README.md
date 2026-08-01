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

| Metric | Value |
|--------|-------|
| Training samples | 8,000 |
| Features | 10 |
| Precision | 53.6% |
| Recall | 9.5% |
| F1 Score | 16.1% |

Model is trained on synthetic bootstrap data, not real failure patterns. These numbers prove the pipeline works end-to-end. Meaningful predictions require real project CI history.

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

## 🚧 Limitations

- Trained on synthetic data only, not real failure patterns
- GitHub Actions workflows not yet tested in live environment
- Test discovery uses string parsing, custom pytest plugins may break
- File matching uses filename similarity, not import graphs

---

## 📝 License

MIT © Abhay Verma 2024
