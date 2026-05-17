"""
Model Training Pipeline for Fraud Detection.
MLP Backpropagation with SMOTE balancing, cross-validation, and comprehensive metrics.
Saves model artifacts + metrics JSON for the dashboard.
"""
import pandas as pd
import numpy as np
import json
import logging
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
)
from imblearn.over_sampling import SMOTE
from config import (
    PROCESSED_CSV_PATH, MODEL_PATH, SCALER_PATH, FEATURES_PATH,
    METRICS_PATH, HIDDEN_LAYERS, ACTIVATION, SOLVER, MAX_ITER,
    LEARNING_RATE_INIT, EARLY_STOPPING, VALIDATION_FRACTION,
    TEST_SIZE, RANDOM_STATE, CV_FOLDS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 55)
    logger.info("  FRAUD DETECTION — Model Training Pipeline")
    logger.info("=" * 55)

    # ── 1. LOAD DATA ──────────────────────────────────────
    logger.info("📂 Loading processed data...")
    df = pd.read_csv(PROCESSED_CSV_PATH)

    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    feature_columns = list(X.columns)

    logger.info(f"   Features: {len(feature_columns)}")
    logger.info(f"   Class balance: Legit={sum(y == 0):,} | Fraud={sum(y == 1):,}")

    # ── 2. TRAIN/TEST SPLIT ───────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"📊 Split: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test")

    # ── 3. SMOTE (Balance training set) ───────────────────
    logger.info("⚖️ Applying SMOTE to training data...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info(f"   After SMOTE: Legit={sum(y_train_res == 0):,} | Fraud={sum(y_train_res == 1):,}")

    # ── 4. SCALE ──────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # ── 5. TRAIN ──────────────────────────────────────────
    logger.info(f"🧠 Training MLP {HIDDEN_LAYERS}...")
    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation=ACTIVATION,
        solver=SOLVER,
        max_iter=MAX_ITER,
        learning_rate_init=LEARNING_RATE_INIT,
        early_stopping=EARLY_STOPPING,
        validation_fraction=VALIDATION_FRACTION,
        random_state=RANDOM_STATE,
        verbose=True
    )
    model.fit(X_train_scaled, y_train_res)

    # ── 6. EVALUATE ───────────────────────────────────────
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 55)
    print("  CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"[TN] True Negative  (Legit->Legit): {tn:,}")
    print(f"[FP] False Positive (Legit->Fraud): {fp:,}")
    print(f"[FN] False Negative (Fraud->Legit): {fn:,}  <- DANGEROUS")
    print(f"[TP] True Positive  (Fraud->Fraud): {tp:,}")
    print(f"\n[*] ROC-AUC: {auc:.4f}")
    print(f"[*] F1:      {f1:.4f}")
    print(f"[*] Accuracy: {acc:.4f}")

    # ── 7. CROSS-VALIDATION (lightweight — scoring only) ──
    logger.info(f"\n🔄 {CV_FOLDS}-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    # Use a simpler estimator for CV speed (scale + model, no SMOTE in CV for speed)
    from sklearn.pipeline import Pipeline as SkPipeline
    cv_pipe = SkPipeline([
        ('scaler', StandardScaler()),
        ('model', MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYERS,
            activation=ACTIVATION, solver=SOLVER,
            max_iter=MAX_ITER, early_stopping=EARLY_STOPPING,
            validation_fraction=VALIDATION_FRACTION,
            random_state=RANDOM_STATE
        ))
    ])
    cv_scores = cross_val_score(cv_pipe, X, y, cv=cv, scoring='f1', n_jobs=-1)
    print(f"\n[CV] F1 Scores: {np.round(cv_scores, 4)}")
    print(f"CV F1 Mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 8. SAVE ARTIFACTS ─────────────────────────────────
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_columns, FEATURES_PATH)

    metrics = {
        'accuracy': round(acc, 4),
        'f1_score': round(f1, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'roc_auc': round(auc, 4),
        'cv_f1_mean': round(float(cv_scores.mean()), 4),
        'cv_f1_std': round(float(cv_scores.std()), 4),
        'confusion_matrix': {
            'true_negative': int(tn), 'false_positive': int(fp),
            'false_negative': int(fn), 'true_positive': int(tp)
        },
        'model_config': {
            'hidden_layers': list(HIDDEN_LAYERS),
            'activation': ACTIVATION,
            'max_iter': MAX_ITER,
            'smote': True,
            'features': feature_columns
        }
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\n💾 Saved: {MODEL_PATH}, {SCALER_PATH}, {FEATURES_PATH}, {METRICS_PATH}")
    logger.info("🎉 Training complete!")


if __name__ == '__main__':
    main()