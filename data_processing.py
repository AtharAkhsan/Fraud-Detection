"""
Data Processing Pipeline for Fraud Detection.
Reads raw CSV directly (no MySQL needed), performs forensic feature engineering,
and saves processed data ready for model training.
"""
import pandas as pd
import numpy as np
import logging
from config import (
    RAW_CSV_PATH, PROCESSED_CSV_PATH, COLUMN_MAPPING,
    LEGIT_SAMPLE_SIZE, CHUNK_SIZE, RANDOM_STATE
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_and_sample(csv_path: str) -> pd.DataFrame:
    """
    Memory-efficient loading with smart undersampling.
    Keeps ALL fraud rows, randomly samples legit rows.
    """
    logger.info(f"📂 Reading: {csv_path}")

    fraud_chunks = []
    legit_chunks = []
    total = 0

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
        chunk = chunk.rename(columns=COLUMN_MAPPING)
        total += len(chunk)
        fraud_chunks.append(chunk[chunk['is_fraud'] == 1])
        legit_chunks.append(chunk[chunk['is_fraud'] == 0])
        logger.info(f"   Processed {total:,} rows...")

    df_fraud = pd.concat(fraud_chunks, ignore_index=True)
    df_legit = pd.concat(legit_chunks, ignore_index=True)

    if len(df_legit) > LEGIT_SAMPLE_SIZE:
        df_legit = df_legit.sample(n=LEGIT_SAMPLE_SIZE, random_state=RANDOM_STATE)

    df = pd.concat([df_fraud, df_legit], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    logger.info(f"✅ Loaded {len(df):,} rows — Fraud: {len(df_fraud):,} | Legit: {len(df_legit):,}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Forensic feature engineering — creating fraud-indicative signals."""
    logger.info("⚙️ Engineering forensic features...")

    # Balance error origin (manipulation indicator)
    df['error_bal_orig'] = df['new_balance_orig'] + df['amount'] - df['old_balance_orig']

    # Balance error destination
    df['error_bal_dest'] = df['old_balance_dest'] + df['amount'] - df['new_balance_dest']

    # Amount-to-balance ratio (disproportionate transfer?)
    df['amount_bal_ratio'] = np.where(
        df['old_balance_orig'] > 0,
        df['amount'] / df['old_balance_orig'],
        0.0
    )

    # Account zeroed flag (drained to zero)
    df['is_orig_zeroed'] = (df['new_balance_orig'] == 0).astype(int)

    # Destination started empty (mule account indicator)
    df['is_dest_empty_start'] = (df['old_balance_dest'] == 0).astype(int)

    # One-Hot encode transaction type
    df = pd.get_dummies(df, columns=['trans_type'], drop_first=True)
    for col in [c for c in df.columns if c.startswith('trans_type_')]:
        df[col] = df[col].astype(int)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove irrelevant columns and validate data quality."""
    drop = [c for c in ['name_orig', 'name_dest', 'is_flagged_fraud', 'created_at'] if c in df.columns]
    df = df.drop(columns=drop)

    nulls = df.isnull().sum().sum()
    if nulls > 0:
        logger.warning(f"⚠️ {nulls} null values found — filling with 0")
        df = df.fillna(0)

    # Drop any remaining non-numeric columns
    non_numeric = df.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        logger.warning(f"⚠️ Dropping non-numeric columns: {non_numeric}")
        df = df.drop(columns=non_numeric)

    logger.info(f"✅ Clean data: {df.shape[0]:,} rows × {df.shape[1]} features")
    return df


def main():
    logger.info("=" * 55)
    logger.info("  FRAUD DETECTION — Data Processing Pipeline")
    logger.info("=" * 55)

    df = load_and_sample(RAW_CSV_PATH)
    df = engineer_features(df)
    df = clean_data(df)

    df.to_csv(PROCESSED_CSV_PATH, index=False)
    logger.info(f"💾 Saved → '{PROCESSED_CSV_PATH}'")

    print("\n[Preview]")
    print(df[['amount', 'error_bal_orig', 'amount_bal_ratio', 'is_orig_zeroed', 'is_fraud']].head())
    print(f"\n[Features]: {list(df.columns)}")


if __name__ == '__main__':
    main()