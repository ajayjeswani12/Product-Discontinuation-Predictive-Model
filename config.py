"""
Configuration file for Product Discontinuation Prediction
"""

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data files
PRODUCT_DETAILS_FILE = DATA_DIR / "ProductDetails.csv"
CATALOGUE_DISCONTINUATION_FILE = DATA_DIR / "CatalogueDiscontinuation.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature engineering parameters
MIN_WEEKS_OUT_FOR_TRAINING = 1  # Only use pre-decision data
FORECAST_ACCURACY_WINDOW = 4  # Weeks to look back for accuracy calculation

# Model selection
MODELS_TO_EVALUATE = [
    'logistic_regression',
    'random_forest',
    'xgboost',
    'lightgbm'
]

# Business metrics thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.8
LOW_CONFIDENCE_THRESHOLD = 0.2

# Logging configuration
LOGGING_LEVEL = "INFO"