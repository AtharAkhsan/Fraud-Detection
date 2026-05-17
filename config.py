"""
Centralized configuration for the Fraud Detection project.
All constants and paths in one place — no more hardcoded strings.
"""
import os

# ─── FILE PATHS ───────────────────────────────────────────
RAW_CSV_PATH = os.getenv('RAW_CSV_PATH', 'PS_20174392719_1491204439457_log.csv')
PROCESSED_CSV_PATH = os.getenv('PROCESSED_CSV_PATH', 'processed_fraud_data.csv')
MODEL_PATH = 'model_fraud.pkl'
SCALER_PATH = 'scaler_fraud.pkl'
FEATURES_PATH = 'feature_columns.pkl'
METRICS_PATH = 'training_metrics.json'

# ─── DATABASE (Optional — only for ingest_data.py) ───────
DB_CONNECTION_STR = os.getenv(
    'DB_CONNECTION_STR',
    'mysql+mysqlconnector://root:@localhost/fraud_project'
)

# ─── DATA PROCESSING ─────────────────────────────────────
LEGIT_SAMPLE_SIZE = 50000
CHUNK_SIZE = 50000

COLUMN_MAPPING = {
    'step': 'step',
    'type': 'trans_type',
    'amount': 'amount',
    'nameOrig': 'name_orig',
    'oldbalanceOrg': 'old_balance_orig',
    'newbalanceOrig': 'new_balance_orig',
    'nameDest': 'name_dest',
    'oldbalanceDest': 'old_balance_dest',
    'newbalanceDest': 'new_balance_dest',
    'isFraud': 'is_fraud',
    'isFlaggedFraud': 'is_flagged_fraud'
}

# ─── MODEL TRAINING ──────────────────────────────────────
HIDDEN_LAYERS = (128, 64, 32)
ACTIVATION = 'relu'
SOLVER = 'adam'
MAX_ITER = 500
LEARNING_RATE_INIT = 0.001
EARLY_STOPPING = True
VALIDATION_FRACTION = 0.1
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
