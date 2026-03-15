# ⚾ MLB Prediction Models — NRFI/YRFI + Starter K Total

[![Run Daily Predictions](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/daily_predictions.yml/badge.svg)](../../actions/workflows/daily_predictions.yml)

> **Real data. No hallucinations. Full 2024 backtest vs actuals.**

---

## ▶️ Run in Google Colab (easiest — no install)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/mlb_models.ipynb)

Click the badge above → Runtime → Run All.  
First run downloads 2024 Statcast data (~20 min). Subsequent runs use cache.

---

## 💻 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
python mlb_models.py --mode backtest   # Full 2024 validation
python mlb_models.py --mode predict    # Train + prepare for today's slate
```

---

## 🤖 GitHub Actions (Auto-runs daily)

The workflow in `.github/workflows/daily_predictions.yml`:
- Triggers **every day at 1pm ET**
- Caches Statcast data so it doesn't re-download every run
- Saves prediction CSVs to `/predictions/YYYY-MM-DD/`
- Uploads the results chart as a downloadable Actions artifact

**To enable:** Go to your repo → Actions tab → Enable workflows.  
**To run manually:** Actions → "MLB Daily Predictions" → "Run workflow"

---

## 📊 Models

### NRFI/YRFI (Logistic Regression)
| Feature | Direction |
|---|---|
| Starter 1st-inning K% | Higher K% → NRFI |
| Starter 1st-inning BB% | Higher BB% → YRFI |
| Opposing wOBA (1st inning) | Higher wOBA → YRFI |
| Park factor | High-offense park → YRFI |
| Weather offense factor | Hot + out-blowing wind → YRFI |

### K Total O/U (Ridge Regression)
| Feature | Direction |
|---|---|
| Pitcher avg Ks/game | → More Ks |
| Pitcher K% | → More Ks |
| K consistency (low variance) | → Reliable over |
| Opponent K% (how often they K) | → More Ks |
| Park factor | Low-offense park → More Ks |

---

## 📁 Output Files

| File | Contents |
|---|---|
| `mlb_model_results_2024.png` | 7-panel backtest dashboard |
| `nrfi_backtest_2024.csv` | Game-by-game predictions vs actuals |
| `k_backtest_2024.csv` | Start-by-start K predictions vs actuals |
| `predictions/YYYY-MM-DD/` | Daily prediction logs (via Actions) |

---

## ⚙️ Setup Notes

- Replace `YOUR_USERNAME/YOUR_REPO` in this README with your actual GitHub path
- The Colab badge auto-links to your notebook once you update the URL
- No API keys required — all data via `pybaseball` (Baseball Savant + FanGraphs)
