
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT / "data" / "processed" / "training" /
    "student_test" / "all_student_test_labeled_candidates.csv"
)

MODEL_DIR = ROOT / "models"

REPORT_DIR = ROOT / "reports" / "student_test" / "evaluation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

VISUAL_DIR = ROOT / "visualizations" / "student_test"
VISUAL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FEATURES AND MODELS
# ============================================================

FEATURES = [
    "distance_to_arc_m",
    "projection_fraction",
    "candidate_rank",
    "gps_speed_mps",
    "gps_heading_deg",
    "road_length_m",
    "road_bearing_deg",
    "heading_difference_deg"
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "Extra Trees": "extra_trees.joblib",
    "HistGradientBoosting": "histgradientboosting.joblib"
}


# ============================================================
# LOAD STUDENT-TEST DATA
# ============================================================

print("=" * 65)
print("PATHSNAP - STUDENT-TEST MODEL EVALUATION")
print("=" * 65)

print("\nLoading student-test dataset...")

df = pd.read_csv(DATA_FILE)

X = df[FEATURES]
y = df["label"].astype(int)

print("Candidate rows:", len(df))
print("Student-test trips:", df["trip_id"].nunique())
print("GPS observations:", df["gps_index"].nunique())


# ============================================================
# VERIFY DATA
# ============================================================

if X.isna().any().any():
    raise ValueError("Missing feature values detected")

if set(y.unique()) != {0, 1}:
    raise ValueError("Both binary classes must be present")

if df.groupby(["trip_id", "gps_index"])["label"].max().ne(1).any():
    raise ValueError(
        "Some GPS observations do not have a positive candidate"
    )


# ============================================================
# EVALUATE FIVE SAVED MODELS
# ============================================================

results = []
trip_results = []

for model_name, filename in MODEL_FILES.items():

    print("\n" + "=" * 60)
    print("Evaluating:", model_name)

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = joblib.load(model_path)

    if model.n_features_in_ != len(FEATURES):
        raise ValueError(
            f"{model_name} expects {model.n_features_in_} features"
        )

    # Prediction timing
    start_time = time.perf_counter()

    predictions = model.predict(X)
    probabilities_all = model.predict_proba(X)

    inference_time = time.perf_counter() - start_time

    # Locate positive-class probability
    classes = model.classes_
    positive_index = list(classes).index(1)

    probabilities = probabilities_all[:, positive_index]

    # Candidate-level metrics
    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y, predictions, zero_division=0
    )

    recall = recall_score(
        y, predictions, zero_division=0
    )

    macro_f1 = f1_score(
        y, predictions,
        average="macro",
        zero_division=0
    )

    positive_f1 = f1_score(
        y, predictions,
        average="binary",
        zero_division=0
    )

    loss = log_loss(
        y,
        probabilities_all,
        labels=classes
    )

    auc = roc_auc_score(y, probabilities)

    # Top-1 candidate accuracy per GPS observation
    prediction_data = df[
        ["trip_id", "gps_index", "arc_id", "label"]
    ].copy()

    prediction_data["probability"] = probabilities

    best_candidates = (
        prediction_data
        .sort_values(
            ["trip_id", "gps_index", "probability"],
            ascending=[True, True, False]
        )
        .drop_duplicates(
            ["trip_id", "gps_index"]
        )
    )

    top1_accuracy = best_candidates["label"].mean()

    # Per-trip Top-1 results
    per_trip = (
        best_candidates
        .groupby("trip_id")["label"]
        .mean()
        .reset_index(name="Top1_Accuracy")
    )

    per_trip["Model"] = model_name
    trip_results.append(per_trip)

    # Save metrics
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro_F1": macro_f1,
        "Positive_F1": positive_f1,
        "Top1_Accuracy": top1_accuracy,
        "ROC_AUC": auc,
        "Log_Loss": loss,
        "Inference_Time_Sec": inference_time,
        "Inference_Time_Per_Row_Ms":
            inference_time / len(X) * 1000
    })

    # Save confusion matrix
    cm = confusion_matrix(y, predictions)

    pd.DataFrame(
        cm,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"]
    ).to_csv(
        REPORT_DIR /
        f"{filename.replace('.joblib', '')}_confusion_matrix.csv"
    )

    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Log Loss:       {loss:.5f}")
    print(f"Inference time: {inference_time:.4f} seconds")


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Macro_F1",
    ascending=False
)

results_df.to_csv(
    REPORT_DIR / "student_test_model_metrics.csv",
    index=False
)

trip_df = pd.concat(
    trip_results,
    ignore_index=True
)

trip_df.to_csv(
    REPORT_DIR / "student_test_per_trip_metrics.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("STUDENT-TEST EVALUATION SUMMARY")
print("=" * 65)

print(results_df.to_string(index=False))

print("\nMetrics saved:")
print(REPORT_DIR / "student_test_model_metrics.csv")

print("\nPer-trip results saved:")
print(REPORT_DIR / "student_test_per_trip_metrics.csv")

print("\nSTUDENT-TEST EVALUATION COMPLETED")
