import logging
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import mlflow
import mlflow.sklearn

# -------------------------------------------------
# Logging Configuration
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------
# Paths & Experiment Name
# -------------------------------------------------
DATA_PATH = "data/processed/train_with_target.csv"
MLFLOW_EXPERIMENT_NAME = "Credit Risk Model Training"

# -------------------------------------------------
# Evaluation Function
# -------------------------------------------------
def evaluate_model(model, X_test, y_test):
    """Evaluate classification model using standard metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }


# -------------------------------------------------
# Main Training Pipeline
# -------------------------------------------------
def main():
    logging.info("Starting model training pipeline")

    # -----------------------------
    # Load Data
    # -----------------------------
    df = pd.read_csv(DATA_PATH)
    logging.info(f"Data loaded: {df.shape}")

    # -----------------------------
    # Feature / Target Separation
    # -----------------------------
    X = df.drop(columns=["is_high_risk", "CustomerId"], errors="ignore")
    y = df["is_high_risk"]

    # IMPORTANT: keep numeric features only
    X = X.select_dtypes(include=["number"])
    logging.info(f"Features after numeric filtering: {X.shape}")

    # -----------------------------
    # Train-Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    logging.info("Train-test split completed")

    # -----------------------------
    # MLflow Setup
    # -----------------------------
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # =================================================
    # Logistic Regression + Grid Search
    # =================================================
    with mlflow.start_run(run_name="Logistic Regression"):

        log_model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        log_param_grid = {
            "C": [0.01, 0.1, 1, 10]
        }

        log_grid = GridSearchCV(
            log_model,
            log_param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )

        log_grid.fit(X_train, y_train)
        best_log_model = log_grid.best_estimator_

        metrics = evaluate_model(best_log_model, X_test, y_test)

        mlflow.log_params(log_grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_log_model, "model")

        logging.info("Logistic Regression training completed")
        logging.info(f"Logistic Regression metrics: {metrics}")

    # =================================================
    # Random Forest + Random Search
    # =================================================
    with mlflow.start_run(run_name="Random Forest"):

        rf_model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )

        rf_param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10]
        }

        rf_random = RandomizedSearchCV(
            rf_model,
            rf_param_dist,
            n_iter=10,
            cv=5,
            scoring="roc_auc",
            random_state=42,
            n_jobs=-1
        )

        rf_random.fit(X_train, y_train)
        best_rf_model = rf_random.best_estimator_

        metrics = evaluate_model(best_rf_model, X_test, y_test)

        mlflow.log_params(rf_random.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_rf_model, "model")

        logging.info("Random Forest training completed")
        logging.info(f"Random Forest metrics: {metrics}")

    logging.info("Model training pipeline completed successfully")


# -------------------------------------------------
# Entry Point
# -------------------------------------------------
if __name__ == "__main__":
    main()
