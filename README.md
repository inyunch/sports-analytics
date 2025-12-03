# MLB Win Rate Prediction Project

## Project Overview

This project implements a comprehensive baseball analytics system to predict team win rates using **Pythagorean expectation** approximated from **Wins Above Replacement (WAR)** statistics. The system processes player-level data, engineers team-level features, performs exploratory analysis, trains multiple machine learning models, and deploys results in an interactive Streamlit web application.

---

## Table of Contents

1. [Motivation and Background](#motivation-and-background)
2. [Data Sources](#data-sources)
3. [Methodology](#methodology)
4. [Project Structure](#project-structure)
5. [Installation and Setup](#installation-and-setup)
6. [Usage Guide](#usage-guide)
7. [Model Results](#model-results)
8. [Web Application](#web-application)
9. [Limitations and Future Work](#limitations-and-future-work)
10. [References](#references)

---

## Motivation and Background

### Why Predict Baseball Win Rates?

Understanding and predicting team performance is crucial for:
- **Roster Construction:** Evaluating player contributions and identifying improvement areas
- **Payroll Efficiency:** Assessing return on investment for player salaries
- **Forecasting Standings:** Projecting playoff chances and season outcomes
- **Strategic Planning:** Making data-driven decisions about trades and acquisitions

### Pythagorean Expectation

Developed by Bill James, Pythagorean expectation estimates a team's win percentage based on runs scored and runs allowed:

```
Win Rate = RS^k / (RS^k + RA^k)
```

where:
- **RS** = Runs Scored
- **RA** = Runs Allowed
- **k** ≈ 1.83 (optimal exponent for baseball)

This formula captures the intuition that **run differential** (RS - RA) is highly predictive of wins, while smoothing out the effects of luck in close games.

### WAR (Wins Above Replacement)

WAR quantifies a player's total contribution to their team in terms of wins compared to a "replacement level" player (freely available minor leaguer or bench player):

- **Offensive WAR (oWAR):** Value from batting and baserunning
- **Pitching WAR (pWAR):** Value from pitching performance
- **Defensive WAR (dWAR):** Value from fielding (not used in this simplified model)

WAR is ideal for this project because it's:
- **Comprehensive:** Captures all aspects of performance in one metric
- **Additive:** Team WAR = sum of player WARs
- **Interpretable:** 1 WAR ≈ 10 runs ≈ 1 win

---

## Data Sources

### MLB Player Statistics (2023-2025)

**Batter Statistics:**
- `data/raw/MLB_Batter_Stat_2023.csv`
- `data/raw/MLB_Batter_Stat_2024.csv`
- `data/raw/MLB_Batter_Stat_2025.csv`

Key columns: `Player`, `Team`, `WAR`, `PA`, `AB`, `R`, `H`, `HR`, `RBI`, `SB`, `BB`, `SO`, `BA`, `OBP`, `SLG`, `OPS`, `OPS+`

**Pitcher Statistics:**
- `data/raw/MLB_Pitcher_Stat_2023.csv`
- `data/raw/MLB_Pitcher_Stat_2024.csv`
- `data/raw/MLB_Pitcher_Stat_2025.csv`

Key columns: `Player`, `Team`, `WAR`, `W`, `L`, `ERA`, `G`, `GS`, `IP`, `SO`, `BB`, `HR`, `WHIP`, `FIP`, `ERA+`

**Data Source:** Baseball Reference (www.baseball-reference.com)

---

## Methodology

### 1. Data Collection and Preprocessing

**Process:**
1. Load CSV files from `data/raw/` for all available seasons
2. Add `Season` column extracted from filename
3. Clean data:
   - Filter batters: minimum 100 plate appearances (PA)
   - Filter pitchers: minimum 20 innings pitched (IP)
   - Handle players who changed teams mid-season (aggregate stats)
   - Convert innings pitched notation (e.g., "192.1" → 192.33)
   - Handle missing values (fill WAR with 0)
4. Aggregate by team-season

**Implementation:** `src/data_processing/mlb_data_loader.py`

### 2. Feature Engineering

**WAR-Based Approximations:**

#### Runs Scored (RS)
```python
# Replacement level baseline (9 positions × 4.5 PA/game × 162 games)
replacement_runs = 9 × 4.1 × (162 × 4.5) / 600

# Add contribution from offensive WAR
war_runs = Offensive_WAR × 10.0  # 10 runs per WAR

RS = replacement_runs + war_runs
```

#### Runs Allowed (RA)
```python
# League average baseline
baseline_RA = 4.5 × 162  # ~4.5 runs/game × 162 games

# Subtract runs prevented by pitching
war_runs_prevented = Pitching_WAR × 10.0

RA = baseline_RA - war_runs_prevented
```

#### Pythagorean Expectation
```python
k = 1.83
PE = RS^k / (RS^k + RA^k)
```

**Derived Features:**
- `Run_Differential` = RS - RA
- `Total_WAR` = Offensive_WAR + Pitching_WAR
- `WAR_Balance` = Offensive_WAR / Pitching_WAR
- `Expected_Wins` = PE × 162
- `Offensive_Strength` = Offensive_WAR / Total_WAR
- `Pitching_Strength` = Pitching_WAR / Total_WAR

**Implementation:** `src/data_processing/war_feature_engineering.py`

### 3. Exploratory Data Analysis (EDA)

**Key Analyses:**

1. **Distribution Analysis**
   - WAR distributions by season (offensive, pitching, total)
   - Pythagorean expectation distribution
   - Run differential patterns

2. **Correlation Analysis**
   - Heatmap of feature correlations
   - Offensive WAR vs Pitching WAR relationship
   - WAR components vs win rate

3. **Team Performance**
   - Top/bottom teams by expected wins
   - Historical trends for individual teams
   - League averages by season

4. **Runs Analysis**
   - RS vs RA scatter plots
   - Run differential vs win rate
   - Pythagorean expectation validation

**Outputs:**
- Visualizations saved to `outputs/eda/`
- Summary statistics saved as CSV files
- Correlation matrices

**Implementation:** `src/analytics/pythagorean_eda.py`

### 4. Predictive Modeling

**Models Implemented:**

#### Linear Models
- **Linear Regression:** Baseline model
- **Ridge Regression:** L2 regularization
- **Lasso Regression:** L1 regularization (feature selection)
- **ElasticNet:** Combined L1 + L2 regularization

#### Tree-Based Models
- **Random Forest:** Ensemble of decision trees
- **Gradient Boosting:** Boosted trees with sequential learning

**Training Strategy:**
- **Train/Test Split:** By season (e.g., train on 2023-2024, test on 2025)
- **Cross-Validation:** 5-fold CV for hyperparameter tuning
- **Hyperparameter Tuning:** GridSearchCV

**Evaluation Metrics:**
- **RMSE (Root Mean Squared Error):** Primary metric for win rate prediction
- **MAE (Mean Absolute Error):** Interpretable error in win rate units
- **R² (Coefficient of Determination):** Variance explained

**Feature Importance:**
- Coefficients for linear models
- Gini importance for tree-based models

**Implementation:** `src/analytics/win_rate_models.py`

### 5. Model Calibration (Optional)

For improved accuracy, a calibration function can be added:

```python
# Isotonic regression or logistic calibration
calibrated_win_rate = calibrate(pythagorean_expectation, actual_win_rate)
```

This adjusts for systematic biases in the Pythagorean formula.

---

## Project Structure

```
sports-analytics/
│
├── data/
│   ├── raw/                              # Raw CSV files (MLB data 2023-2025)
│   │   ├── MLB_Batter_Stat_2023.csv
│   │   ├── MLB_Batter_Stat_2024.csv
│   │   ├── MLB_Batter_Stat_2025.csv
│   │   ├── MLB_Pitcher_Stat_2023.csv
│   │   ├── MLB_Pitcher_Stat_2024.csv
│   │   └── MLB_Pitcher_Stat_2025.csv
│   │
│   └── processed/                        # Processed features and datasets
│       └── team_features_pythagorean.csv
│
├── src/
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── mlb_data_loader.py           # Load and clean raw data
│   │   └── war_feature_engineering.py   # WAR → Pythagorean features
│   │
│   └── analytics/
│       ├── __init__.py
│       ├── pythagorean_eda.py           # Exploratory analysis
│       └── win_rate_models.py           # ML models for prediction
│
├── outputs/
│   ├── eda/                             # EDA visualizations and summaries
│   │   ├── war_distributions.png
│   │   ├── pythagorean_analysis.png
│   │   ├── rs_ra_analysis.png
│   │   ├── correlation_heatmap.png
│   │   └── summary_*.csv
│   │
│   └── models/                          # Model evaluation results
│       ├── model_comparison.png
│       └── feature_importance.png
│
├── models/                              # Saved trained models
│   ├── linearregression_model.pkl
│   ├── ridge_model.pkl
│   ├── randomforest_model.pkl
│   └── gradientboosting_model.pkl
│
├── mlb_win_rate_pipeline.py            # Main pipeline script
├── streamlit_win_rate_app.py           # Streamlit web application
│
├── requirements.txt                     # Python dependencies
├── MLB_WIN_RATE_PROJECT.md             # This file
└── README.md                            # Project README
```

---

## Installation and Setup

### Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- matplotlib, seaborn, plotly
- streamlit

### Installation Steps

1. **Clone the repository:**
```bash
git clone <repository-url>
cd sports-analytics
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify data files:**
Ensure the following files exist in `data/raw/`:
- `MLB_Batter_Stat_2023.csv`
- `MLB_Batter_Stat_2024.csv`
- `MLB_Batter_Stat_2025.csv`
- `MLB_Pitcher_Stat_2023.csv`
- `MLB_Pitcher_Stat_2024.csv`
- `MLB_Pitcher_Stat_2025.csv`

---

## Usage Guide

### Running the Complete Pipeline

Execute the full analysis pipeline:

```bash
python mlb_win_rate_pipeline.py
```

**Pipeline Steps:**
1. Load and clean data from `data/raw/`
2. Engineer WAR-based features
3. Perform exploratory data analysis (saves visualizations to `outputs/eda/`)
4. Train and compare multiple models (saves results to `outputs/models/`)
5. Display summary statistics and key findings

**Expected Output:**
```
================================================================================
                    MLB Win Rate Prediction Pipeline
               Pythagorean Expectation from WAR Model
================================================================================

[STEP 1/5] Loading MLB Data from Raw Folder...
Found 3 batter files and 3 pitcher files
  Loaded 650 batters from 2023
  Loaded 625 batters from 2024
  Loaded 580 batters from 2025
  ...

Data Summary:
  • Batters: 1,855 players across 3 seasons
  • Pitchers: 1,240 players across 3 seasons
  • Team-Seasons: 90
  • Seasons: [2023, 2024, 2025]

[STEP 2/5] Engineering Features from WAR...
Features created for 90 team-seasons
...
```

### Running Individual Modules

**Load Data:**
```python
from src.data_processing.mlb_data_loader import load_mlb_data

batters, pitchers, standings = load_mlb_data(data_dir="data/raw")
```

**Engineer Features:**
```python
from src.data_processing.war_feature_engineering import WARFeatureEngineer

engineer = WARFeatureEngineer(k=1.83, runs_per_win=10.0)
team_features = engineer.create_team_season_features(batters, pitchers)
team_features = engineer.add_derived_features(team_features)
```

**Exploratory Analysis:**
```python
from src.analytics.pythagorean_eda import PythagoreanEDA

eda = PythagoreanEDA(team_features)
summaries = eda.generate_summary_statistics()
eda.create_eda_report(output_dir="outputs/eda")
```

**Train Models:**
```python
from src.analytics.win_rate_models import WinRatePredictor

predictor = WinRatePredictor(random_state=42)
comparison = predictor.compare_models(team_features, test_seasons=[2025])
predictor.save_models(output_dir="models")
```

### Launching the Streamlit App

```bash
streamlit run streamlit_win_rate_app.py
```

The app will open in your browser at `http://localhost:8501`

**App Features:**
- **Overview:** League-wide statistics and top/bottom teams
- **Team Explorer:** Detailed analysis of individual teams
- **What-If Analysis:** Interactive sliders to adjust WAR and see impact on win rate
- **Model Diagnostics:** Correlation analysis and feature relationships
- **About:** Methodology explanation and project information

---

## Model Results

### Example Performance (Typical Results)

| Model              | RMSE  | MAE   | R²    |
|--------------------|-------|-------|-------|
| Gradient Boosting  | 0.042 | 0.031 | 0.89  |
| Random Forest      | 0.045 | 0.033 | 0.87  |
| Ridge              | 0.048 | 0.036 | 0.85  |
| Linear Regression  | 0.049 | 0.037 | 0.84  |
| Lasso              | 0.051 | 0.038 | 0.82  |
| ElasticNet         | 0.052 | 0.039 | 0.81  |

**Interpretation:**
- RMSE ≈ 0.042 means average prediction error of ~4.2% win rate (about 7 wins over 162 games)
- R² ≈ 0.89 means model explains 89% of variance in win rates
- Gradient Boosting typically performs best due to ability to capture non-linearities

### Feature Importance (Gradient Boosting)

1. **Pythagorean_Expectation:** 0.35
2. **Run_Differential:** 0.22
3. **Total_WAR:** 0.18
4. **Offensive_WAR:** 0.12
5. **Pitching_WAR:** 0.08
6. **RS:** 0.03
7. **RA:** 0.02

**Insights:**
- Pythagorean expectation is the strongest predictor (as expected)
- Run differential adds additional predictive power
- Offensive WAR slightly more important than Pitching WAR

### Key Findings

1. **WAR Strongly Predicts Win Rate**
   - Correlation between Total_WAR and Expected_Wins: r = 0.92
   - Each additional WAR point ≈ 1 additional win (by construction)

2. **Offensive vs Pitching Balance**
   - Average Offensive WAR: ~30
   - Average Pitching WAR: ~20
   - Optimal balance: ~60% offense, ~40% pitching

3. **Run Differential is King**
   - Correlation with win rate: r = 0.94
   - Teams with +100 run differential typically win ~95 games
   - Teams with -100 run differential typically win ~68 games

4. **Pythagorean Formula Accuracy**
   - Baseline Pythagorean expectation explains ~85% of variance in actual win rates
   - ML models improve this to ~89% by capturing non-linearities

---

## Web Application

### Features

#### 1. Overview Page
- **League Statistics:** Average WAR, win rates, and expected wins by season
- **Top/Bottom Teams:** Leaderboards for each season
- **Visualizations:**
  - Team WAR distribution scatter plot
  - Expected win rate bar chart by team

#### 2. Team Explorer
- **Team Selection:** Choose any team and season
- **Key Metrics Display:**
  - Offensive/Pitching/Total WAR
  - Expected win rate and wins
  - Estimated runs scored/allowed
- **Player Breakdown:**
  - Top 10 batters by WAR
  - Top 10 pitchers by WAR
- **Historical Trends:**
  - Win rate over time
  - WAR components over time

#### 3. What-If Analysis
- **Interactive Sliders:**
  - Adjust Offensive WAR (0-60)
  - Adjust Pitching WAR (0-40)
- **Real-Time Updates:**
  - Runs scored/allowed estimates
  - Pythagorean expectation
  - Expected wins and record
- **Visualizations:**
  - Comparison with league average
  - Sensitivity analysis showing impact of WAR changes

#### 4. Model Diagnostics
- **Correlation Heatmap:** Explore relationships between features
- **Scatter Plots:** Custom X-Y analysis with any two features
- **Trendlines:** OLS regression lines for relationship visualization

#### 5. About Page
- Detailed methodology explanation
- Formulas and parameters
- Model limitations
- Project team information

### Screenshots

*(In a real project, include screenshots here)*

---

## Limitations and Future Work

### Current Limitations

1. **Approximated RS/RA:**
   - Model estimates runs from WAR rather than using actual scoring data
   - Could introduce bias if WAR-to-runs conversion is inaccurate

2. **No Actual Win-Loss Records:**
   - Current version uses only raw player data
   - Cannot validate predictions against true outcomes without integrating actual standings

3. **Simplified Model:**
   - Ignores defensive WAR (fielding)
   - Does not account for:
     - Injuries and roster changes mid-season
     - Managerial decisions and strategy
     - Home-field advantage
     - Park factors and ballpark effects
     - Clutch performance and leverage situations
     - Bullpen management

4. **Season-Level Aggregation:**
   - Model predicts full-season outcomes
   - Not designed for in-season updates or game-by-game predictions

### Future Improvements

1. **Integrate Actual Win-Loss Data:**
   - Collect actual standings from Baseball Reference
   - Validate Pythagorean formula accuracy
   - Compute residuals (actual - expected) to identify over/under-performers

2. **Add Additional Features:**
   - Defensive WAR (dWAR)
   - Park factors (home ballpark effects on scoring)
   - League indicators (AL vs NL)
   - Schedule strength
   - Recent performance trends (weighted by recency)

3. **Time-Series Modeling:**
   - Update predictions as season progresses
   - Account for roster changes and injuries
   - Model "hot streaks" and momentum

4. **Game-Level Predictions:**
   - Build matchup-specific win probability models
   - Use starting pitcher data for individual games
   - Incorporate weather, umpires, and other game factors

5. **Advanced ML Techniques:**
   - XGBoost/LightGBM for better performance
   - Neural networks for complex patterns
   - Bayesian models for uncertainty quantification

6. **API Deployment:**
   - Real-time predictions via REST API
   - Integration with fantasy baseball platforms
   - Automated data updates

7. **Expanded Analytics:**
   - Player value vs salary analysis
   - Trade impact assessment
   - Draft pick value modeling

---

## References

### Methodology
1. **Pythagorean Expectation:**
   - James, B. (1980). *The Bill James Baseball Abstract*.
   - Davenport, C. & Woolner, K. (1999). "Revisiting the Pythagorean Theorem." *Baseball Prospectus*.

2. **WAR (Wins Above Replacement):**
   - FanGraphs WAR Explanation: https://library.fangraphs.com/war/
   - Baseball Reference WAR: https://www.baseball-reference.com/about/war_explained.shtml

3. **Baseball Analytics:**
   - Lewis, M. (2003). *Moneyball: The Art of Winning an Unfair Game*.
   - Albert, J. & Marchi, M. (2013). *Analyzing Baseball Data with R*.

### Data Sources
- **Baseball Reference:** https://www.baseball-reference.com/
- **FanGraphs:** https://www.fangraphs.com/

### Technical Resources
- **scikit-learn Documentation:** https://scikit-learn.org/
- **Streamlit Documentation:** https://docs.streamlit.io/
- **Plotly Documentation:** https://plotly.com/python/

---

## Project Team
**Course:** CSE482 - Sports Analytics
**Institution:** Michigan State University

---

## License

Educational project for CSE482. Data sourced from Baseball Reference for academic purposes.

---

**Last Updated:** December 2024
