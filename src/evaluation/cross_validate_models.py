
from pathlib import Path

import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)


# Paths
ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT / "data" / "processed" / "training" /
    "train" / "all_train_labeled_candidates.csv"
)

OUTPUT_DIR = ROOT / "reports" / "cross_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load data
print("Loading training dataset...")

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

print("Total rows:", len(df))
print("Total trips:", groups.nunique())


# Five models: same configurations as compare_models.py
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


# Group-aware 5-fold cross-validation
cv = GroupKFold(n_splits=5)

fold_results = []

for model_name, classifier in models.items():

    print("\n" + "=" * 60)
    print("MODEL:", model_name)
    print("=" * 60)

    fold_number = 1

    for train_idx, val_idx in cv.split(X, y, groups):

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        train_groups = groups.iloc[train_idx]
        val_groups = groups.iloc[val_idx]

        print(
            f"\nFold {fold_number}: "
            f"Train trips={train_groups.nunique()}, "
            f"Validation trips={val_groups.nunique()}"
        )

        # Verify no trip leakage
        assert set(train_groups).isdisjoint(set(val_groups))

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier)
        ])

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0
        )

        positive_f1 = f1_score(
            y_val,
            predictions,
            average="binary",
            zero_division=0
        )

        fold_results.append({
            "Model": model_name,
            "Fold": fold_number,
            "Training_Trips": train_groups.nunique(),
            "Validation_Trips": val_groups.nunique(),
            "Macro_F1": macro_f1,
            "Positive_F1": positive_f1
        })

        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Positive F1: {positive_f1:.4f}")

        fold_number += 1


# Save fold-wise results
fold_df = pd.DataFrame(fold_results)

fold_df.to_csv(
    OUTPUT_DIR / "cross_validation_fold_results.csv",
    index=False
)


# Calculate average and standard deviation
summary_df = (
    fold_df.groupby("Model")
    .agg(
        Mean_Macro_F1=("Macro_F1", "mean"),
        Std_Macro_F1=("Macro_F1", "std"),
        Mean_Positive_F1=("Positive_F1", "mean")
    )
    .reset_index()
    .sort_values("Mean_Macro_F1", ascending=False)
)

summary_df.to_csv(
    OUTPUT_DIR / "cross_validation_summary.csv",
    index=False
)


print("\n" + "=" * 60)
print("CROSS-VALIDATION SUMMARY")
print("=" * 60)

print(summary_df.to_string(index=False))

print("\nFiles saved:")
print(OUTPUT_DIR / "cross_validation_fold_results.csv")
print(OUTPUT_DIR / "cross_validation_summary.csv")

print("\n5-FOLD GROUP CROSS-VALIDATION COMPLETED.")
