NBA Moneyline Tool

Overview
--------
This repository builds a moneyline (win-probability) prediction pipeline for NBA games. The main script (`moneyline.py`) and the helper module (`util.py`) perform data ingestion, feature engineering, model training/evaluation, and real-time prediction against sportsbook odds.

What the code does
------------------
- Loads historical NBA game data (from `data/RawDF.csv` or fetched via the NBA API).
- Filters regular-season games and computes running per-team season statistics (moving averages of box-score stats, win percentages, streaks, and home-win percentage).
- Builds paired matchup feature rows (home vs away) and prepares training/test datasets.
- Trains several classifiers (Logistic Regression, MLP, KNN, Random Forest, Gradient Boosting). Optionally ensembles models.
- Evaluates and prints training/test accuracy and ROC scores; displays coefficient and calibration plots.
- Fetches today's moneyline odds from an odds API, maps bookmaker team names to local abbreviations, composes home/away feature vectors from the latest team stats, predicts calibrated win probabilities, and estimates expected returns and standard deviations for a $100 bet.
- Writes per-team prediction results to `data/Results.xlsx` (and prints a preview to console).

Primary files
-------------
- `moneyline.py` — main script that runs end-to-end: data loading, preprocessing, training, calibration, and real-time predictions.
- `util.py` — helper functions: data ingestion, feature engineering, model training, evaluation, odds fetching and name-mapping, prediction assembly.
- `data/RawDF.csv` — historical game data (used as default input).
- `data/DraftkingsNameMatcher.json` — mapping from odds-provider (DraftKings) team names to local `TEAM_ABBREVIATION` values.
- `data/Results.xlsx` — output produced by `moneyline.py` with predictions and derived metrics.

Output format (example columns)
-------------------------------
The produced `data/Results.xlsx` contains rows per team entry with columns such as:
- `Home Team`, `Away Team` — team abbreviations (e.g., `LAL`, `BKN`).
- `Home Odds`, `Away Odds` — American-format odds from the bookmaker.
- `Home Prob`, `Away Prob` — calibrated model probabilities for each side to win.
- `Home LB Return`, `Away LB Return` — expected return for a $100 bet (with a liquidity buffer adjustment).
- `Home Std`, `Away Std` — estimated standard deviation of the return.

Common issues and notes
-----------------------
- Team name mapping: the script maps bookmaker names to local abbreviations using `data/DraftkingsNameMatcher.json` plus a fuzzy/normalized lookup. If a bookmaker name is not mapped, the prediction row will contain NaNs and the script prints a warning. Update or extend `data/DraftkingsNameMatcher.json` when you encounter unmapped names.
- API key: The odds API key is currently provided in `util.py` as the `apiKey` parameter. For security, consider moving this key into an environment variable or a config file and updating `util.py` to read it from there.
- Dependencies: this project uses Python packages including `pandas`, `numpy`, `scikit-learn`, `nba_api`, `requests`, `beautifulsoup4`, `tqdm`, `seaborn`, `matplotlib`, and `openpyxl` (for Excel output). Install them via pip.

How to run
----------
1. (Optional) Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn nba_api requests beautifulsoup4 tqdm seaborn matplotlib openpyxl
```

2. Run the main script from the repository root:

```bash
python3 moneyline.py
```

This will train models (or re-use precomputed data where applicable), fetch today's odds, make predictions, print diagnostics, and save results to `data/Results.xlsx`.

Suggestions
-----------
- Move the odds API key to an environment variable and read it from `os.environ` inside `util.py`.
- Log unmapped bookmaker names to a file to easily extend `DraftkingsNameMatcher.json`.
- Persist trained models (pickle) if you plan to reuse them frequently without retraining.

Contact / Author
----------------
Repository owner: atticusdrex

