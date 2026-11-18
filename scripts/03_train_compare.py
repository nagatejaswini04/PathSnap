"""Train and compare candidate-scoring models.

- Group-aware 5-fold CV over the 80 training trips (GroupKFold by trip).
- Fit on all training trips, evaluate on the 10 student-testing trips.
- Only observations with a reliable reference label (<= 50 m) are used for
  fitting and point-level scoring.
- Training data: every 3rd observation per trip (1 Hz points are highly
  autocorrelated); applied identically to every model.
Outputs: results/models/*.csv|json, models/experiments/*.joblib, reports/figures/models_*.png
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from pathsnap.features import FEATURES
from pathsnap.io import load_trip
from pathsnap.metrics import candidate_metrics, continuity, top_candidate_mask, obs_key
from pathsnap.models import SEED, zoo

STRIDE = 3
RES, FIG, MOD = "results/models", "reports/figures", "models/experiments"
for d in (RES, FIG, MOD):
    os.makedirs(d, exist_ok=True)
only = sys.argv[1:]  # optional subset of model names

tr = pd.read_parquet("data/processed/train_cands.parquet")
tr = tr[tr.label_ok & (tr.obs % STRIDE == 0)].reset_index(drop=True)
te_all = pd.read_parquet("data/processed/student_test_cands.parquet")
te = te_all[te_all.label_ok].reset_index(drop=True)
X, y, groups = tr[FEATURES].values, tr.y.values, tr.trip_id.values
Xte = te[FEATURES].values
print("train rows", len(tr), "obs", tr.groupby(["trip_id", "obs"]).ngroups, "| test rows", len(te))

arcs_te = {t: load_trip("data/raw/student_test", t, with_route=False).arcs for t in te.trip_id.unique()}


def route_continuity(df, prob):
    top = top_candidate_mask(obs_key(df), prob)
    p = df[top].sort_values(["trip_id", "obs"])
    return float(np.mean([continuity(g.arc.values, arcs_te[t]) for t, g in p.groupby("trip_id")]))


gkf = GroupKFold(n_splits=5)
summary, folds = [], []
for name, spec in zoo().items():
    if only and name not in only:
        continue
    print("==", name)
    fold_scores = []
    for k, (a, b) in enumerate(gkf.split(X, y, groups)):
        m = spec["make"]()
        t0 = time.time(); m.fit(X[a], y[a]); tfit = time.time() - t0
        p = m.predict_proba(X[b])[:, 1]
        s = candidate_metrics(tr.iloc[b], p); s.pop("cm")
        s.update(model=name, fold=k, fit_s=tfit); folds.append(s); fold_scores.append(s)
        print(f"  fold {k}: F1={s['macro_f1']:.4f} top1={s['top1_acc']:.4f} ll={s['log_loss']:.4f} ({tfit:.0f}s)")
    m = spec["make"]()
    t0 = time.time(); m.fit(X, y); tfit = time.time() - t0
    p_tr = m.predict_proba(X)[:, 1]
    tr_s = candidate_metrics(tr, p_tr)
    t0 = time.time(); p = m.predict_proba(Xte)[:, 1]; tinf = time.time() - t0
    s = candidate_metrics(te, p)
    cont = route_continuity(te, p)
    joblib.dump(m, f"{MOD}/{name}.joblib", compress=3)
    np.save(f"{RES}/test_proba_{name}.npy", p)
    f = pd.DataFrame(fold_scores)
    row = dict(model=name, family=spec["family"], config=spec["config"],
               cv_macro_f1=f.macro_f1.mean(), cv_macro_f1_std=f.macro_f1.std(),
               cv_top1=f.top1_acc.mean(), cv_log_loss=f.log_loss.mean(),
               train_macro_f1=tr_s["macro_f1"], train_top1=tr_s["top1_acc"], train_log_loss=tr_s["log_loss"],
               test_top1=s["top1_acc"], test_macro_f1=s["macro_f1"], test_precision=s["precision"],
               test_recall=s["recall"], test_log_loss=s["log_loss"], test_route_continuity=cont,
               fit_seconds=tfit, infer_ms_per_obs=1000 * tinf / s["n_obs"], test_obs=s["n_obs"],
               test_cm=json.dumps(s["cm"]))
    summary.append(row)
    print("  TEST", {k: round(v, 4) for k, v in row.items() if isinstance(v, float)})
    pd.DataFrame(summary).to_csv(f"{RES}/comparison{'_' + '_'.join(only) if only else ''}.csv", index=False)
    pd.DataFrame(folds).to_csv(f"{RES}/cv_folds{'_' + '_'.join(only) if only else ''}.csv", index=False)

if only:
    sys.exit(0)

# --------------------------------------------- epoch-wise curves (where they exist)
# Hold out 16 training trips as a validation set (group split) to record histories.
trips = np.array(sorted(tr.trip_id.unique()))
rng = np.random.RandomState(SEED); val_trips = set(rng.choice(trips, 16, replace=False))
va = tr.trip_id.isin(val_trips).values
# LightGBM: boosting rounds
m = zoo()["LightGBM"]["make"]()
m.fit(X[~va], y[~va], eval_set=[(X[~va], y[~va]), (X[va], y[va])], eval_names=["train", "valid"], eval_metric="binary_logloss")
ev = m.evals_result_
pd.DataFrame({"round": np.arange(1, len(ev["train"]["binary_logloss"]) + 1), "train_logloss": ev["train"]["binary_logloss"],
              "valid_logloss": ev["valid"]["binary_logloss"]}).to_csv(f"{RES}/curve_LightGBM.csv", index=False)
# HistGradientBoosting: internal train/validation scores (neg log loss)
m = zoo()["HistGradientBoosting"]["make"]()
m.fit(X[~va], y[~va])
pd.DataFrame({"iter": np.arange(len(m.train_score_)), "train_logloss": -m.train_score_,
              "valid_logloss": -m.validation_score_}).to_csv(f"{RES}/curve_HistGradientBoosting.csv", index=False)
# MLP: manual epochs with partial_fit, train/valid log loss per epoch
sc = StandardScaler().fit(X[~va]); Xa, Xv = sc.transform(X[~va]), sc.transform(X[va])
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512, learning_rate_init=1e-3, random_state=SEED)
hist = []
for ep in range(40):
    idx = rng.permutation(len(Xa))
    mlp.partial_fit(Xa[idx], y[~va][idx], classes=[0, 1])
    hist.append((ep + 1, log_loss(y[~va], mlp.predict_proba(Xa)[:, 1]), log_loss(y[va], mlp.predict_proba(Xv)[:, 1])))
pd.DataFrame(hist, columns=["epoch", "train_logloss", "valid_logloss"]).to_csv(f"{RES}/curve_MLP.csv", index=False)
print("done")
