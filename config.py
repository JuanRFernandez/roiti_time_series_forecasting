"""
Configuration file for renewable energy forecasting
"""

# Data paths
DATA_RAW_DIR = 'data/raw'
DATA_PROCESSED_DIR = 'data/processed'
RESULTS_DIR = 'results'

# Model parameters
SARIMAX_ORDER = (2, 1, 2)
SARIMAX_SEASONAL_ORDER = (1, 0, 1, 24)

SVR_KERNEL = 'rbf'
SVR_C = 100
SVR_EPSILON = 0.1

# Feature engineering
LAG_PERIODS = [24, 48, 168]  # hours
ROLLING_WINDOWS = [24, 48]   # hours

# Validation
TEST_DAYS = 7
FORECAST_HORIZON = 24  # hours (day-ahead)

# Business parameters
PENALTY_PER_MWH = 50  # EUR

# Data years
DATA_YEARS = [2022, 2023, 2024]

# Logging
LOG_LEVEL = 'INFO'

