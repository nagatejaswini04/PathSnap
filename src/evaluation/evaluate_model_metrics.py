
from pathlib import Path
import time

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT / "data" / "processed" / "training" /
    "train" / "all_train_labeled_candidates.csv"
)

MODEL_DIR = ROOT / "models"

OUTPUT_DIR = ROOT / "reports" / "model_evaluation"
PLOT_DIR = ROOT / "visualizations" / "model_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

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

X = df[FEATURES]
y = df["label"].astype(int)
groups = df["trip_id"]


# Same group split as compare_models.py
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

print("Testing trips:", groups.iloc[test_idx].nunique())
print("Testing rows:", len(X_test))


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "Extra Trees": "extra_trees.joblib",
    "HistGradientBoosting": "hist_gradientboosting.joblib"
}


# ============================================================
# EVALUATION
# ============================================================

results = []

sns.set_theme(style="whitegrid")

for name, filename in MODEL_FILES.items():

    print(f"\nEvaluating {name}...")

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        print("Model file missing:", model_path)
        continue

    model = joblib.load(model_path)

    # Measure prediction time.
    start_time = time.perf_counter()

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    inference_time = time.perf_counter() - start_time

    # Metrics.
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Macro_F1": f1_score(
            y_test, predictions, average="macro",
            zero_division=0
        ),
        "Positive_F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "Log_Loss": log_loss(
            y_test, probabilities, labels=[0, 1]
        ),
        "Inference_Time_Sec": inference_time,
        "Inference_Time_Per_Row_Ms": (
            inference_time / len(X_test) * 1000
        )
    }

    results.append(metrics)

    print("Macro F1:", round(metrics["Macro_F1"], 4))
    print("Log Loss:", round(metrics["Log_Loss"], 5))
    print("Inference time:", round(inference_time, 4), "seconds")

    # Confusion matrix.
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plot_name = name.lower().replace(" ", "_")

    plt.savefig(
        PLOT_DIR / f"{plot_name}_confusion_matrix.png",
        dpi=300
    )

    plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "model_evaluation_metrics.csv",
    index=False
)

# Log Loss comparison.
if not results_df.empty:

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=results_df,
        x="Model",
        y="Log_Loss"
    )

    plt.title("Log Loss Comparison")
    plt.ylabel("Log Loss")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "log_loss_comparison.png",
        dpi=300
    )

    plt.close()

    # Inference time comparison.
    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=results_df,
        x="Model",
        y="Inference_Time_Per_Row_Ms"
    )

    plt.title("Inference Time Per Candidate")
    plt.ylabel("Milliseconds per candidate")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "inference_time_comparison.png",
        dpi=300
    )

    plt.close()


print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETED")
print("=" * 60)

if not results_df.empty:
    print(results_df.to_string(index=False))

print("\nMetrics saved:")
print(OUTPUT_DIR / "model_evaluation_metrics.csv")

print("\nVisualisations saved:")
print(PLOT_DIR)
