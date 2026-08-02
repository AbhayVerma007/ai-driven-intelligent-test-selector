# 🚀 TestWise AI - Setup Guide

A step-by-step guide to get TestWise AI running on your project.

---

## 📋 Prerequisites

- Python 3.11 or higher
- Git installed
- A project with pytest tests

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/AbhayVerma007/ai-driven-intelligent-test-selector.git
cd ai-driven-intelligent-test-selector
```

---

## Step 2: Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Step 3: Run the Demo (Verify It Works)

```bash
python -m examples.basic_demo
```

**Expected output:**
```
1. Bootstrapping 200 synthetic commits...
2. Training the failure-prediction model...
   Metrics: {
     "precision": 0.83,
     "recall": 0.80,
     "f1": 0.82
   }
3. Selecting tests for scenario: low_risk...
   {
     "total_tests": 49,
     "selected": 11,
     "savings_percentage": 77.6
   }
```

**If you see savings_percentage around 77%, the tool works.**

---

## Step 4: Use on Your Own Project

Copy these files into your project:

```
src/collector.py
src/trainer.py
src/selector.py
src/cli.py
requirements.txt
```

Then run:

```bash
# Collect your git history
python -m src.cli collect --max-commits 500

# Train the model
python -m src.cli train

# Select tests for a change
python -m src.cli select --base-ref main~1 --head-ref main --test-path tests
```

**Sample output:**
```json
{
  "total_tests": 150,
  "selected": 35,
  "savings_percentage": 76.7
}
```

---

## Step 5: Add GitHub Actions (Optional)

Copy `.github/workflows/intelligent-testing.yml` into your repo's `.github/workflows/` folder.

Now every push to a feature branch or pull request will:
- Run AI test selection automatically
- Execute only the selected tests
- Post a summary with model health and skipped test reasons

---

## 🔍 How to Know It's Working

| Check | What You'll See |
|-------|-----------------|
| Demo runs | Terminal shows 77% savings |
| CLI select command | JSON output with selected tests |
| GitHub Actions | Green checkmark on PR |
| CI Summary | Scroll to bottom of Actions log for full report |

---

## ❓ Common Questions

**Q: The model has low accuracy on my project. Why?**
A: The model improves with real CI history. Run `python -m src.cli collect` regularly and retrain weekly. Accuracy climbs automatically.

**Q: What if no tests are selected?**
A: The safety fallback runs the full test suite. Your pipeline never skips everything.

**Q: How do I mark tests as critical?**
A: Tests with "critical" or "smoke" in their name are automatically treated as critical and always run.

---

## 📝 License

MIT © Abhay Verma 2024