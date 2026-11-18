"""All result figures, generated only from saved experiment outputs in results/."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = "reports/figures"; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
C1, C2, C3 = "#2a6fdb", "#e07b00", "#2a9d8f"

cmp_ = pd.read_csv("results/models/comparison.csv")
folds = pd.read_csv("results/models/cv_folds.csv")
order = cmp_.sort_values("cv_macro_f1").model.tolist()

# 1. CV macro F1 (mean +- std, with fold points)
fig, ax = plt.subplots(figsize=(8, 4))
c = cmp_.set_index("model").loc[order]
ax.barh(order, c.cv_macro_f1, xerr=c.cv_macro_f1_std, color=C1, alpha=.85, capsize=3)
for i, m in enumerate(order):
    v = folds[folds.model == m].macro_f1
    ax.scatter(v, [i] * len(v), s=10, c="k", zorder=3)
    ax.text(0.962, i, f"{c.cv_macro_f1[m]:.4f} ± {c.cv_macro_f1_std[m]:.4f}", va="center", fontsize=8.5)
ax.set_xlim(min(.84, c.cv_macro_f1.min() - .03), 1.0); ax.set_xlabel("Macro F1 (GroupKFold-5 on 80 training trips)")
ax.set_title("Cross-validation macro F1 by model (dots = folds)")
plt.tight_layout(); plt.savefig(f"{FIG}/models_cv_macro_f1.png"); plt.close()

# 2. student-test top-1 and macro F1
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(order)); w = .38
ax.bar(x - w / 2, c.test_top1, w, label="Top-1 accuracy", color=C1)
ax.bar(x + w / 2, c.test_macro_f1, w, label="Macro F1", color=C2)
for i, m in enumerate(order):
    ax.text(i - w / 2, c.test_top1[m] + .004, f"{c.test_top1[m]:.3f}", ha="center", fontsize=7.5)
    ax.text(i + w / 2, c.test_macro_f1[m] + .004, f"{c.test_macro_f1[m]:.3f}", ha="center", fontsize=7.5)
ax.set_xticks(x, order, rotation=20); ax.set_ylim(.7, 1.0); ax.legend(loc="lower right")
ax.set_title("Student-testing partition (10 trips): classifier-only Top-1 and macro F1")
plt.tight_layout(); plt.savefig(f"{FIG}/models_test_top1_f1.png"); plt.close()

# 3. log loss
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar(x - w / 2, c.cv_log_loss, w, label="CV", color=C1); ax.bar(x + w / 2, c.test_log_loss, w, label="Student-test", color=C2)
ax.set_yscale("log"); ax.set_xticks(x, order, rotation=20); ax.set_ylabel("binary log loss (log scale)"); ax.legend()
ax.set_title("Candidate-level log loss (lower is better)")
plt.tight_layout(); plt.savefig(f"{FIG}/models_log_loss.png"); plt.close()

# 4. over/under-fitting: train vs CV vs test
fig, ax = plt.subplots(figsize=(9, 3.8)); w = .27
ax.bar(x - w, c.train_macro_f1, w, label="Train (in-sample)", color="#888")
ax.bar(x, c.cv_macro_f1, w, label="CV", color=C1); ax.bar(x + w, c.test_macro_f1, w, label="Student-test", color=C2)
ax.set_xticks(x, order, rotation=20); ax.set_ylim(.8, 1.0); ax.legend(ncol=3, loc="lower right")
ax.set_title("Generalisation gap: in-sample vs CV vs student-test macro F1")
plt.tight_layout(); plt.savefig(f"{FIG}/models_generalisation_gap.png"); plt.close()

# 5. inference time vs accuracy
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.scatter(c.infer_ms_per_obs, c.test_top1, s=40, c=C1)
for m in order:
    ax.annotate(m, (c.infer_ms_per_obs[m], c.test_top1[m]), fontsize=8, xytext=(4, 3), textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("inference time per GPS observation (ms, log)"); ax.set_ylabel("student-test Top-1")
ax.set_title("Accuracy vs inference cost")
plt.tight_layout(); plt.savefig(f"{FIG}/models_speed_accuracy.png"); plt.close()

# 6. confusion matrices
n = len(order); fig, axs = plt.subplots(1, n, figsize=(2.6 * n, 2.8))
for a, m in zip(axs, order[::-1]):
    cm = np.array(json.loads(c.test_cm[m])); cmn = cm / cm.sum(1, keepdims=True)
    a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]:,}\n({cmn[i, j]:.2f})", ha="center", va="center", fontsize=7,
                   color="white" if cmn[i, j] > .5 else "black")
    a.set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=7); a.set_yticks([0, 1], ["true 0", "true 1"], fontsize=7)
    a.set_title(m, fontsize=8)
plt.suptitle("Candidate-level confusion matrices on student-test (top-candidate decision rule)", fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIG}/models_confusion.png"); plt.close()

# 7. training curves (only models with real iteration histories)
curves = [("LightGBM", "round", "boosting round", "validation (16 held-out trips)"),
          ("HistGradientBoosting", "iter", "boosting iteration", "internal 10% early-stopping split"),
          ("MLP", "epoch", "epoch", "validation (16 held-out trips)")]
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for a, (m, col, xl, vl) in zip(axs, curves):
    p = f"results/models/curve_{m}.csv"
    if not os.path.exists(p):
        continue
    d = pd.read_csv(p)
    a.plot(d[col], d.train_logloss, label="train", color=C1); a.plot(d[col], d.valid_logloss, label=vl, color=C2)
    a.set_title(m); a.set_xlabel(xl); a.set_ylabel("log loss"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/models_training_curves.png"); plt.close()

# 8. Bayesian optimisation
if os.path.exists("results/bayes_opt/trials.csv"):
    t = pd.read_csv("results/bayes_opt/trials.csv")
    done = t[t.state == "COMPLETE"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    axs[0].scatter(done.number, done.value, s=14, c=C1, label="completed trial")
    pr = t[t.state == "PRUNED"]
    axs[0].scatter(pr.number, [done.value.min()] * len(pr), marker="x", c="#999", s=14, label="pruned (median rule)")
    axs[0].plot(done.number, done.value.cummax(), c=C2, label="best so far")
    axs[0].set_xlabel("trial"); axs[0].set_ylabel("CV macro F1"); axs[0].legend(fontsize=8); axs[0].set_title("Optuna TPE optimisation history")
    tv = pd.read_csv("results/bayes_opt/tuned_vs_untuned.csv").iloc[:2]
    mets = ["cv_macro_f1", "test_top1", "test_macro_f1"]
    xx = np.arange(len(mets))
    axs[1].bar(xx - .2, tv.iloc[0][mets].astype(float), .4, label="untuned", color="#999")
    axs[1].bar(xx + .2, tv.iloc[1][mets].astype(float), .4, label="tuned", color=C3)
    for i, mm in enumerate(mets):
        axs[1].text(i - .2, float(tv.iloc[0][mm]) + .002, f"{float(tv.iloc[0][mm]):.4f}", ha="center", fontsize=7.5)
        axs[1].text(i + .2, float(tv.iloc[1][mm]) + .002, f"{float(tv.iloc[1][mm]):.4f}", ha="center", fontsize=7.5)
    lo = min(tv[mets].astype(float).min()) - .03
    axs[1].set_ylim(lo, 1); axs[1].set_xticks(xx, ["CV macro F1", "Test Top-1", "Test macro F1"]); axs[1].legend()
    axs[1].set_title("LightGBM: untuned vs Bayesian-tuned")
    plt.tight_layout(); plt.savefig(f"{FIG}/bayes_opt.png"); plt.close()
    fi = pd.read_csv("results/bayes_opt/feature_importance.csv")
    plt.figure(figsize=(7, 5)); plt.barh(fi.feature[::-1], fi.gain[::-1] / fi.gain.sum(), color=C1)
    plt.xlabel("share of total split gain"); plt.title("Final LightGBM feature importance")
    plt.tight_layout(); plt.savefig(f"{FIG}/feature_importance.png"); plt.close()

# 9. HMM
if os.path.exists("results/hmm/student_test_per_trip.csv"):
    pt = pd.read_csv("results/hmm/student_test_per_trip.csv", dtype={"trip": str})
    piv = pt.pivot(index="trip", columns="method", values="top1")
    meths = ["Classifier only", "HMM + Gaussian emission (baseline)", "HMM + classifier emission"]
    fig, ax = plt.subplots(figsize=(11, 3.8)); xx = np.arange(len(piv)); w = .27
    for k, (mm, col) in enumerate(zip(meths, ["#999", C1, C2])):
        ax.bar(xx + (k - 1) * w, piv[mm], w, label=mm, color=col)
    ax.set_xticks(xx, piv.index, rotation=0); ax.set_ylim(max(0, piv.values.min() - .05), 1.0)
    ax.set_ylabel("Top-1 (reliable obs)"); ax.legend(fontsize=8, ncol=3, loc="lower right")
    ax.set_title("Student-test per-trip Top-1: classifier-only vs HMM decoding")
    plt.tight_layout(); plt.savefig(f"{FIG}/hmm_per_trip.png"); plt.close()
    g = pd.read_csv("results/hmm/decoder_grid_train_oof.csv")
    h = g[g.emission == "clf"].pivot(index="beta", columns="w_emit", values="train_top1_micro")
    base = g[g.emission == "classifier-only"].train_top1_micro.iloc[0]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(h.values, cmap="viridis", aspect="auto")
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            ax.text(j, i, f"{h.values[i, j]:.4f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_xticks(range(h.shape[1]), h.columns); ax.set_yticks(range(h.shape[0]), h.index)
    ax.set_xlabel("emission weight w_e"); ax.set_ylabel("transition scale beta (m)")
    ax.set_title(f"Decoder grid, train OOF Top-1\n(classifier-only = {base:.4f})", fontsize=9)
    plt.colorbar(im); plt.tight_layout(); plt.savefig(f"{FIG}/hmm_grid.png"); plt.close()
print("figures written")
