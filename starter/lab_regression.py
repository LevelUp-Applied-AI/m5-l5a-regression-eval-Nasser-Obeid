"""
Module 5 Week A — Lab: Regression & Evaluation

Build and evaluate logistic and linear regression models on the
Petra Telecom customer churn dataset.

Run: python lab_regression.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, r2_score)


def load_data(filepath="data/telecom_churn.csv"):
    """Load the telecom churn dataset.

    Returns:
        DataFrame with all columns.
    """
    df = pd.read_csv(filepath)
    return df


def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into train and test sets with stratification.

    Args:
        df: DataFrame with features and target.
        target_col: Name of the target column.
        test_size: Fraction for test set.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    # Only stratify when the target is categorical (few unique values)
    strat = y if y.nunique() <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    return X_train, X_test, y_train, y_test


def build_logistic_pipeline():
    """Build a Pipeline with StandardScaler and LogisticRegression.

    Returns:
        sklearn Pipeline object.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42, max_iter=1000))
    ])
    return pipe


def build_ridge_pipeline():
    """Build a Pipeline with StandardScaler and Ridge regression.

    Returns:
        sklearn Pipeline object.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0))
    ])
    return pipe


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return classification metrics.

    Args:
        pipeline: sklearn Pipeline with a classifier.
        X_train, X_test: Feature arrays.
        y_train, y_test: Label arrays.

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "accuracy": round(report["accuracy"], 4),
        "precision": round(report["weighted avg"]["precision"], 4),
        "recall": round(report["weighted avg"]["recall"], 4),
        "f1": round(report["weighted avg"]["f1-score"], 4)
    }
    return metrics


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return regression metrics.

    Args:
        pipeline: sklearn Pipeline with a regressor.
        X_train, X_test: Feature arrays.
        y_train, y_test: Target arrays.

    Returns:
        Dictionary with keys: 'mae', 'r2'.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = {
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "r2": round(r2_score(y_test, y_pred), 4)
    }
    return metrics


def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation on the pipeline.

    Args:
        pipeline: sklearn Pipeline.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of folds.

    Returns:
        Array of cross-validation scores.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="accuracy")
    return scores


if __name__ == "__main__":
    # ── Task 1: Load Data and Basic EDA ──────────────────────────────
    df = load_data()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nChurn distribution:\n{df['churned'].value_counts()}")
    print(f"Churn rate: {df['churned'].mean():.2%}")

    # ── Task 2: Split the Data ───────────────────────────────────────
    numeric_features = ["tenure", "monthly_charges", "total_charges",
                        "num_support_calls", "senior_citizen",
                        "has_partner", "has_dependents"]

    df_cls = df[numeric_features + ["churned"]].dropna()
    X_train, X_test, y_train, y_test = split_data(df_cls, "churned")
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train churn rate: {y_train.mean():.2%}")
    print(f"Test  churn rate: {y_test.mean():.2%}")

    # ── Task 3: Logistic Regression Pipeline ─────────────────────────
    pipe = build_logistic_pipeline()
    metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
    print(f"\nLogistic Regression Metrics: {metrics}")

    y_pred = pipe.predict(X_test)
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # ── Task 4: Ridge Regression Pipeline ────────────────────────────
    reg_features = ["tenure", "total_charges", "num_support_calls",
                    "senior_citizen", "has_partner", "has_dependents"]

    df_reg = df[reg_features + ["monthly_charges"]].dropna()
    X_tr, X_te, y_tr, y_te = split_data(df_reg, "monthly_charges")

    ridge_pipe = build_ridge_pipeline()
    reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
    print(f"\nRidge Regression Metrics: {reg_metrics}")

    # ── Task 5: Lasso Regularization Comparison ──────────────────────
    lasso_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Lasso(alpha=0.1))
    ])
    lasso_metrics = evaluate_regressor(lasso_pipe, X_tr, X_te, y_tr, y_te)
    print(f"Lasso Regression Metrics: {lasso_metrics}")

    ridge_coefs = ridge_pipe.named_steps["reg"].coef_
    lasso_coefs = lasso_pipe.named_steps["reg"].coef_

    print(f"\n{'Feature':<22} {'Ridge':>10} {'Lasso':>10}")
    print("-" * 44)
    for feat, rc, lc in zip(reg_features, ridge_coefs, lasso_coefs):
        marker = "  <-- zero" if abs(lc) < 1e-6 else ""
        print(f"{feat:<22} {rc:>10.4f} {lc:>10.4f}{marker}")

    # Features driven to zero by Lasso are likely uninformative for
    # predicting monthly_charges. For example, has_partner and
    # has_dependents may have little relationship with how much a
    # customer pays monthly, while tenure and total_charges carry
    # the strongest signal.

    # ── Task 6: Cross-Validation ─────────────────────────────────────
    scores = run_cross_validation(build_logistic_pipeline(), X_train, y_train)
    print(f"\n5-Fold CV Scores: {scores}")
    print(f"CV Mean: {scores.mean():.3f} +/- {scores.std():.3f}")


"""
── Task 7: Summary of Findings ────────────────────────────────────

1. Most important features for predicting churn:
   tenure and monthly_charges tend to have the largest logistic
   regression coefficients (in absolute value). Customers with
   shorter tenure and higher monthly charges are more likely to
   churn. num_support_calls also contributes — more calls may
   signal dissatisfaction.

2. Model performance:
   The logistic regression achieves reasonable accuracy and weighted
   F1. For a churn problem, recall on the churned class is more
   concerning than precision — missing a customer who will churn
   (false negative) means losing the chance to intervene, which is
   typically costlier than a false alarm (false positive).

3. Recommended next steps:
   - Engineer additional features (e.g., contract_type, internet_service)
     via one-hot encoding to give the model more signal.
   - Try class-weight balancing (class_weight='balanced') to boost
     recall on the minority churn class.
   - Experiment with non-linear models (Random Forest, Gradient
     Boosting) and hyperparameter tuning via GridSearchCV.
   - Evaluate using precision-recall curves and AUC-ROC alongside
     accuracy for a fuller picture.
"""