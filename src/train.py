"""
Train and evaluate heart disease classifiers.

Usage:
    python -m src.train

Outputs:
    - models/best_heart_model.joblib
    - models/metrics_comparison.csv
    - models/confusion_matrices.png
    - models/roc_curves.png
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.database import load_data
from src.preprocess import clean_data, prepare_features

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_heart_model.joblib"
METRICS_CSV = MODEL_DIR / "metrics_comparison.csv"
CM_PNG = MODEL_DIR / "confusion_matrices.png"
ROC_PNG = MODEL_DIR / "roc_curves.png"


# --------------------------------------------------------------------------- #
# 1. Load + clean
# --------------------------------------------------------------------------- #
def load_training_data() -> pd.DataFrame:
    print("[1/5] Loading data from PostgreSQL...")
    df_raw = load_data("SELECT * FROM heart_disease")
    print(f"      raw rows: {len(df_raw)}")

    df = clean_data(df_raw)
    print(f"      clean rows: {len(df)}  (dropped {len(df_raw) - len(df)})")
    return df


# --------------------------------------------------------------------------- #
# 2. Build pipelines
# --------------------------------------------------------------------------- #
def build_models(preprocessor, scale_pos_weight: float) -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("pre", preprocessor),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=3,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "XGBoost": Pipeline([
            ("pre", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }


# --------------------------------------------------------------------------- #
# 3. Evaluate a single fitted pipeline
# --------------------------------------------------------------------------- #
def evaluate_model(name, pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics, y_pred, y_proba


# --------------------------------------------------------------------------- #
# 4. Plots
# --------------------------------------------------------------------------- #
def plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, (_, y_pred, _)) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"], ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=120)
    plt.close()
    print(f"      saved {CM_PNG}")


def plot_roc_curves(results, y_test):
    plt.figure(figsize=(7, 6))
    for name, (metrics, _, y_proba) in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Heart Disease Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_PNG, dpi=120)
    plt.close()
    print(f"      saved {ROC_PNG}")


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main():
    # --- data ---
    df = load_training_data()
    X, y, preprocessor = prepare_features(df)
    print(f"[2/5] Features: {X.shape}  |  positives: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"      train: {len(X_train)}  test: {len(X_test)}")

    # balance ratio for XGBoost
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"      scale_pos_weight = {scale_pos_weight:.2f}")

    # --- train + evaluate ---
    print("[3/5] Training models...")
    models = build_models(preprocessor, scale_pos_weight)
    results = {}
    rows = []

    for name, pipeline in models.items():
        print(f"      >> {name}")
        pipeline.fit(X_train, y_train)

        metrics, y_pred, y_proba = evaluate_model(name, pipeline, X_test, y_test)

        # 5-fold CV on AUC for a more robust number
        cv_auc = cross_val_score(
            pipeline, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring="roc_auc", n_jobs=-1,
        )
        metrics["cv_auc_mean"] = cv_auc.mean()
        metrics["cv_auc_std"] = cv_auc.std()

        rows.append(metrics)
        results[name] = (metrics, y_pred, y_proba)

        print(f"         AUC={metrics['roc_auc']:.3f}  "
              f"Acc={metrics['accuracy']:.3f}  "
              f"Recall={metrics['recall']:.3f}  "
              f"CV-AUC={cv_auc.mean():.3f}±{cv_auc.std():.3f}")
        print(classification_report(y_test, y_pred,
                                    target_names=["No Disease", "Disease"]))

    # --- comparison table ---
    print("[4/5] Writing comparison table...")
    df_metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    df_metrics.to_csv(METRICS_CSV, index=False)
    print(df_metrics.to_string(index=False, float_format="%.4f"))
    print(f"      saved {METRICS_CSV}")

    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)

    # --- pick and save the best ---
    print("[5/5] Selecting best model...")
    best_name = df_metrics.iloc[0]["model"]
    best_pipeline = models[best_name]
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print(f"      best = {best_name}  →  {BEST_MODEL_PATH}")

    # small metadata sidecar (helps the Streamlit app)
    meta = {
        "best_model": best_name,
        "roc_auc": float(df_metrics.iloc[0]["roc_auc"]),
        "feature_columns": list(X.columns),
        "trained_on_rows": int(len(X_train)),
    }
    with open(MODEL_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()