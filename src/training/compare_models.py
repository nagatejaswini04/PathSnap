
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score
)


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT / "data" / "processed" / "training" /
    "train" / "all_train_labeled_candidates.csv"
)

OUTPUT_DIR = ROOT / "reports" / "model_comparison"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


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


# ============================================================
# GROUP-AWARE SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("Training trips:", groups.iloc[train_idx].nunique())
print("Testing trips:", groups.iloc[test_idx].nunique())


# ============================================================
# FIVE MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=300,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=15,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        learning_rate=0.02171994,
        max_iter=150,
        max_leaf_nodes=54,
        max_depth=3,
        min_samples_leaf=29,
        l2_regularization=0.3636047,
        random_state=42
    )
}


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

results = []

for name, classifier in models.items():

    print(f"\nTraining {name}...")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC AUC": roc_auc_score(y_test, probabilities)
    }

    results.append(metrics)

    print(
        f"Accuracy: {metrics['Accuracy']:.4f} | "
        f"F1: {metrics['F1 Score']:.4f} | "
        f"ROC-AUC: {metrics['ROC AUC']:.4f}"
    )

    joblib.dump(
        model,
        MODEL_DIR / (
            name.lower().replace(" ", "_") + ".joblib"
        )
    )


# ============================================================
# SAVE COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

results_df.to_csv(
    OUTPUT_DIR / "model_comparison_results.csv",
    index=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))

print("\nResults saved at:")
print(OUTPUT_DIR / "model_comparison_results.csv")

print("\nAll five models trained successfully.")
