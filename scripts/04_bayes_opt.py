"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected model (LightGBM).

Objective : mean GroupKFold(5) macro-F1 on the 80 training trips (same stride-3,
            reliable-label data as the model comparison).
Search    : see SPACE below. Trials and parameters are logged to
            results/bayes_opt/trials.csv and the Optuna SQLite study.
Then      : refit best config on all training data, evaluate on student_test,
            compare with the untuned model, save final model artifact.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.features import FEATURES
from pathsnap.metrics import candidate_metrics
from pathsnap.models import SEED, zoo

N_TRIALS = int(os.environ.get("N_TRIALS", 40))
STRIDE = 3
OUT = "results/bayes_opt"; os.makedirs(OUT, exist_ok=True); os.makedirs("models/final", exist_ok=True)

tr = pd.read_parquet("data/processed/train_cands.parquet")
tr_full = tr[tr.label_ok].reset_index(drop=True)
tr = tr_full[tr_full.obs % STRIDE == 0].reset_index(drop=True)
te = pd.read_parquet("data/processed/student_test_cands.parquet")
te = te[te.label_ok].reset_index(drop=True)
X, y, g = tr[FEATURES].values, tr.y.values, tr.trip_id.values
splits = list(GroupKFold(n_splits=5).split(X, y, g))


def make(params):
    return lgb.LGBMClassifier(**params, subsample_freq=1, random_state=SEED, n_jobs=2, verbose=-1)


def cv_score(model_fn, trial=None):
    f1s, lls, t1 = [], [], []
    for k, (a, b) in enumerate(splits):
        m = model_fn(); m.fit(X[a], y[a])
        s = candidate_metrics(tr.iloc[b], m.predict_proba(X[b])[:, 1])
        f1s.append(s["macro_f1"]); lls.append(s["log_loss"]); t1.append(s["top1_acc"])
        if trial is not None:
            trial.report(float(np.mean(f1s)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(f1s)), float(np.std(f1s)), float(np.mean(lls)), float(np.mean(t1))


def objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 300, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    )
    t0 = time.time()
    f1, f1sd, ll, t1 = cv_score(lambda: make(params), trial)
    trial.set_user_attr("cv_f1_std", f1sd); trial.set_user_attr("cv_log_loss", ll)
    trial.set_user_attr("cv_top1", t1); trial.set_user_attr("seconds", time.time() - t0)
    return f1


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="pathsnap_lgbm",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
                                storage=f"sqlite:///{OUT}/optuna_study.db", load_if_exists=True)
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    opt_seconds = time.time() - t0
    study.trials_dataframe().to_csv(f"{OUT}/trials.csv", index=False)
    best = study.best_params
    json.dump({"best_params": best, "best_cv_macro_f1": study.best_value, "n_trials": len(study.trials),
               "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
               "optimisation_seconds": opt_seconds}, open(f"{OUT}/best.json", "w"), indent=2)

    # ---------------- untuned vs tuned on identical data / metrics
    rows = []
    for label, fn in [("LightGBM (untuned)", zoo()["LightGBM"]["make"]), ("LightGBM (tuned)", lambda: make(best))]:
        f1, f1sd, ll, t1 = cv_score(fn)
        m = fn(); t = time.time(); m.fit(X, y); fit_s = time.time() - t
        t = time.time(); p = m.predict_proba(te[FEATURES].values)[:, 1]; inf = time.time() - t
        s = candidate_metrics(te, p)
        rows.append(dict(model=label, cv_macro_f1=f1, cv_macro_f1_std=f1sd, cv_log_loss=ll, cv_top1=t1,
                         test_top1=s["top1_acc"], test_macro_f1=s["macro_f1"], test_log_loss=s["log_loss"],
                         fit_seconds=fit_s, infer_ms_per_obs=1000 * inf / s["n_obs"]))
        print(rows[-1])
    # final deployed model: tuned config refit on ALL reliable training observations
    m = make(best); t = time.time(); m.fit(tr_full[FEATURES].values, tr_full.y.values); fit_s = time.time() - t
    t = time.time(); p = m.predict_proba(te[FEATURES].values)[:, 1]; inf = time.time() - t
    s = candidate_metrics(te, p)
    rows.append(dict(model="LightGBM (tuned, refit on all train obs) [deployed]", cv_macro_f1=np.nan,
                     cv_macro_f1_std=np.nan, cv_log_loss=np.nan, cv_top1=np.nan, test_top1=s["top1_acc"],
                     test_macro_f1=s["macro_f1"], test_log_loss=s["log_loss"], fit_seconds=fit_s,
                     infer_ms_per_obs=1000 * inf / s["n_obs"]))
    pd.DataFrame(rows).to_csv(f"{OUT}/tuned_vs_untuned.csv", index=False)
    joblib.dump({"model": m, "features": FEATURES, "params": best}, "models/final/classifier.joblib", compress=3)
    np.save(f"{OUT}/test_proba_final.npy", p)
    imp = pd.DataFrame({"feature": FEATURES, "gain": m.booster_.feature_importance("gain")}).sort_values("gain", ascending=False)
    imp.to_csv(f"{OUT}/feature_importance.csv", index=False)
    print("best", best, study.best_value)
