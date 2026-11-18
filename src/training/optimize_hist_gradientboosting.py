
from pathlib import Path

import joblib
import optuna
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT / "data" / "processed" / "training" /
    "train" / "all_train_labeled_candidates.csv"
)

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "model_optimization"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

N_TRIALS = 30
RANDOM_STATE = 42


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
# GROUP-AWARE TRAIN / VALIDATION SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, val_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

print("Training trips:", groups.iloc[train_idx].nunique())
print("Validation trips:", groups.iloc[val_idx].nunique())


# ============================================================
# OBJECTIVE FUNCTION
# ============================================================

def objective(trial):

    params = {
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.02, 0.2, log=True
        ),

        "max_iter": trial.suggest_int(
            "max_iter", 100, 350, step=50
        ),

        "max_leaf_nodes": trial.suggest_int(
            "max_leaf_nodes", 15, 63
        ),

        "max_depth": trial.suggest_int(
            "max_depth", 3, 12
        ),

        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 20, 100
        ),

        "l2_regularization": trial.suggest_float(
            "l2_regularization", 0.001, 10.0, log=True
        ),

        "random_state": RANDOM_STATE
    }

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),

        ("classifier", HistGradientBoostingClassifier(**params))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    score = f1_score(
        y_val,
        predictions,
        pos_label=1,
        zero_division=0
    )

    print(
        f"Trial {trial.number + 1}: "
        f"F1 = {score:.5f}"
    )

    return score


# ============================================================
# BAYESIAN OPTIMIZATION
# ============================================================

print("\nStarting Optuna Bayesian optimization...")

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
)

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True
)


# ============================================================
# SAVE OPTIMIZATION RESULTS
# ============================================================

print("\nBest parameters:")
print(study.best_params)

print("\nBest validation F1:", study.best_value)

study.trials_dataframe().to_csv(
    REPORT_DIR / "optuna_trials.csv",
    index=False
)

pd.DataFrame([
    {
        "best_validation_f1": study.best_value,
        **study.best_params
    }
]).to_csv(
    REPORT_DIR / "best_hyperparameters.csv",
    index=False
)


# ============================================================
# TRAIN BEST MODEL
# ============================================================

print("\nTraining final tuned model...")

best_params = study.best_params.copy()
best_params["random_state"] = RANDOM_STATE

best_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),

    ("classifier", HistGradientBoostingClassifier(**best_params))
])

best_model.fit(X_train, y_train)

# Validation evaluation
predictions = best_model.predict(X_val)
probabilities = best_model.predict_proba(X_val)[:, 1]

metrics = {
    "precision": precision_score(
        y_val, predictions, zero_division=0
    ),
    "recall": recall_score(
        y_val, predictions, zero_division=0
    ),
    "f1_score": f1_score(
        y_val, predictions, zero_division=0
    ),
    "roc_auc": roc_auc_score(
        y_val, probabilities
    )
}

print("\nTuned model validation results:")

for name, value in metrics.items():
    print(f"{name}: {value:.5f}")

pd.DataFrame([metrics]).to_csv(
    REPORT_DIR / "tuned_model_metrics.csv",
    index=False
)


# ============================================================
# SAVE TUNED MODEL
# ============================================================

model_path = MODEL_DIR / "hist_gradientboosting_tuned.joblib"

joblib.dump(best_model, model_path)

print("\nTuned model saved at:")
print(model_path)

print("\nBayesian optimization completed successfully.")
