
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT / "data" / "processed" / "training" /
    "train" / "all_train_labeled_candidates.csv"
)

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)
print("Number of trips:", df["trip_id"].nunique())


# ============================================================
# FEATURES AND TARGET
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

TARGET = "label"

X = df[FEATURES].copy()
y = df[TARGET].astype(int)
groups = df["trip_id"]


# ============================================================
# GROUP-AWARE TRAIN / TEST SPLIT
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

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

print(
    "Training trips:",
    groups.iloc[train_idx].nunique()
)

print(
    "Testing trips:",
    groups.iloc[test_idx].nunique()
)


# ============================================================
# HIST GRADIENT BOOSTING MODEL
# ============================================================

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "classifier",
        HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42
        )
    )
])


# ============================================================
# TRAIN
# ============================================================

print("\nTraining HistGradientBoosting...")

model.fit(X_train, y_train)


# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating model...")

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

if y_test.nunique() == 2:
    print(
        "\nROC-AUC:",
        roc_auc_score(y_test, probabilities)
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "hist_gradientboosting.joblib"

joblib.dump(model, model_path)

print("\nModel saved at:")
print(model_path)

print("\nHistGradientBoosting training completed.")
