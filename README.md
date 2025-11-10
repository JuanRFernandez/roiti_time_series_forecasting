# Day-Ahead Renewable Energy Forecasting

Machine learning solution for predicting hourly renewable energy generation in German electricity markets.

---

## Problem Statement

Power grid operators run daily "day-ahead" auctions where energy producers must nominate their hourly production for the next day by 12:00 CET. Inaccurate predictions result in penalties of €50 per MWh error.

**Challenge:** Renewable sources (solar, wind) are highly variable and difficult to forecast.

**Goal:** Develop forecasting model to minimize prediction errors and reduce penalty costs.

---

## Solution Overview

### Approach
- **Data:** 3 years of hourly German renewable generation (2022-2024)
- **Models:** SARIMAX (statistical) and SVR (machine learning) with feature engineering
- **Validation:** 7-day walk-forward backtest simulating real auction conditions

### Key Results
- **34% cost reduction** vs naive baseline
- **€2.1 million daily savings**
- **€781 million projected yearly savings**

---

## Repository Structure

```
├── notebooks/
│   ├── 01_Renewable_Energy_Forecasting.ipynb    # EDA and patterns
│   └── 02_Improved_Model.ipynb                  # Model development
├── src/
│   ├── data_loader.py                           # Data ingestion
│   ├── feature_engineer.py                      # Feature creation
│   ├── models.py                                # Model implementations
│   └── 20251109_roiti_case_study_juan_Fernandez.pptx  # Presentation
├── data/
│   ├── raw/                                     # Original CSVs
│   └── processed/                               # Preprocessed data
├── results/                                     # Model outputs
├── run_forecast.py                              # Main pipeline script
├── DEPLOYMENT.md                                # Production architecture
└── pyproject.toml                               # Dependencies
```

---

## Quick Start

### Setup

```bash
# Install dependencies
uv sync

# Activate environment
.venv\Scripts\activate
```

### Run Analysis

**Option 1: Notebooks (Interactive)**
```bash
jupyter notebook notebooks/
```
- Start with `01_Renewable_Energy_Forecasting.ipynb` for EDA
- Run `02_Improved_Model.ipynb` for model results

**Option 2: Python Script (Reproducible)**
```bash
python run_forecast.py
```
Generates model comparison and saves results to `results/model_comparison.csv`

---

## Methodology

### Feature Engineering
- **Temporal lags:** 24h, 48h, 168h (yesterday, 2 days, last week)
- **Rolling statistics:** 24h/48h mean and std deviation
- **Time features:** Hour, day of week, month with cyclical encoding
- **Purpose:** Capture trends and seasonal patterns beyond simple lag

### Models Implemented

**1. Baseline (Naive)**
- Prediction: Tomorrow = Yesterday (24-hour lag)
- MAE: 5,284 MW | Daily cost: €6.3 million

**2. SARIMAX (Statistical)**
- Seasonal ARIMA with 24-hour cycles
- Walk-forward validation (predict 24h, retrain, repeat)
- MAE: 3,501 MW | Daily cost: €4.2 million | **Winner**

**3. SVR (Machine Learning)**  
- Support Vector Regression with engineered features
- RBF kernel with feature scaling
- MAE: 5,255 MW | Daily cost: €6.3 million

### Validation Strategy
- **Test period:** 7 days (168 hours)
- **Approach:** Walk-forward validation
  - Predict next 24 hours
  - Evaluate performance
  - Update training set
  - Repeat
- **Prevents data leakage:** Only uses past information

---

## Key Findings

### Why Baseline is Strong
- High autocorrelation at 24-hour lag (>0.80)
- Weather patterns persist day-to-day
- Difficult to beat without weather forecasts

### Missing Critical Data: Weather Forecasts
- **What:** Wind speed, solar irradiance, cloud cover
- **Impact:** Would enable 20-30% additional accuracy improvement
- **Integration plan:** See `DEPLOYMENT.md`

---

## Production Deployment

See `DEPLOYMENT.md` for complete architecture including:
- Daily prediction pipeline (10:00 CET)
- Monitoring and alerting system
- Retraining strategy (weekly/daily)
- Weather API integration roadmap
- Scalability considerations
- Risk management

**Infrastructure:** Docker containers, PostgreSQL, Grafana monitoring

---

## Technical Details

### Dependencies
- pandas, numpy (data processing)
- scikit-learn (ML models, preprocessing)
- statsmodels (SARIMAX)
- matplotlib, seaborn (visualization)

Managed via `pyproject.toml` with [uv](https://github.com/astral-sh/uv)

### Python Modules

**`src/data_loader.py`**
- `RenewableDataLoader`: Load and preprocess CSV files
- Handles TSO aggregation and hourly resampling

**`src/feature_engineer.py`**
- `TimeSeriesFeatureEngineer`: Create model features
- Modular feature creation (lags, rolling stats, time)

**`src/models.py`**
- `BaselineModel`, `SARIMAXModel`, `SVRModel`: Model implementations
- `ModelEvaluator`: Performance metrics and comparison

**`run_forecast.py`**
- Complete pipeline from data loading to evaluation
- Reproducible execution

---

## Results Summary

| Metric | Baseline | SARIMAX | SVR |
|--------|----------|---------|-----|
| MAE (MW) | 5,284 | **3,501** | 5,255 |
| Daily Cost | €6.3M | **€4.2M** | €6.3M |
| Improvement | - | **34%** | 0.5% |

**Best Model:** SARIMAX with day-ahead walk-forward validation

**Yearly Impact:** €781 million cost reduction

---

## Future Enhancements

1. **Weather Integration** (High Priority)
   - DWD (German Weather Service) API
   - Expected +20-30% accuracy gain

2. **Ensemble Methods**
   - Combine SARIMAX + SVR + XGBoost
   - Weighted by recent performance

3. **Uncertainty Quantification**
   - Prediction intervals
   - Risk-adjusted bidding strategy

4. **Disaggregation**
   - Site-specific models
   - Regional forecasts

---

## License

MIT License

---

## Contact

Juan R. Fernandez  
November 2025
