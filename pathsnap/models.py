"""Model zoo used in the comparison experiment (all binary candidate scorers)."""
from __future__ import annotations

import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

SEED = 42


def zoo() -> dict:
    return {
        "LogisticRegression": dict(
            family="Linear / probabilistic",
            config="StandardScaler + L2, C=1.0, lbfgs, max_iter=1000",
            make=lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))),
        "GaussianNB": dict(
            family="Probabilistic (generative)",
            config="QuantileTransformer(normal) + GaussianNB, var_smoothing=1e-9",
            make=lambda: make_pipeline(QuantileTransformer(output_distribution="normal", subsample=200_000,
                                                           random_state=SEED), GaussianNB())),
        "KNN": dict(
            family="Instance-based",
            config="StandardScaler + k=31, distance-weighted, euclidean",
            make=lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=31, weights="distance",
                                                                              n_jobs=2))),
        "RandomForest": dict(
            family="Ensemble (bagging)",
            config="200 trees, min_samples_leaf=5, max_features=sqrt",
            make=lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=2,
                                                random_state=SEED)),
        "HistGradientBoosting": dict(
            family="Boosting",
            config="lr=0.1, max_iter=300, max_leaf_nodes=31, early stop on 10% val",
            make=lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, max_leaf_nodes=31,
                                                        early_stopping=True, validation_fraction=0.1,
                                                        random_state=SEED)),
        "LightGBM": dict(
            family="Boosting (leaf-wise GBDT)",
            config="lr=0.05, 400 rounds, num_leaves=31, subsample=0.8, colsample=0.8",
            make=lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                            random_state=SEED, n_jobs=2, verbose=-1)),
        "MLP": dict(
            family="Neural network",
            config="StandardScaler + MLP(64,32), relu, adam lr=1e-3, batch 512, early stopping",
            make=lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                                       batch_size=512, learning_rate_init=1e-3,
                                                                       max_iter=60, early_stopping=True,
                                                                       n_iter_no_change=6, random_state=SEED))),
    }
