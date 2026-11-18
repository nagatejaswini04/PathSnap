"""Evaluation metrics.

Decision rule: within each observation, the candidate with the highest score is
the prediction (observation-level Top-1). Candidate-level precision / recall /
macro-F1 are computed with that same rule (top candidate = positive, others =
negative), so they are threshold-free and consistent with Top-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score


def top_candidate_mask(obs_key: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Boolean mask marking the argmax candidate within each observation group.

    obs_key must be grouped contiguously (rows of one observation adjacent).
    """
    df = pd.DataFrame({"k": obs_key, "s": score, "i": np.arange(len(score))})
    idx = df.sort_values(["k", "s"], ascending=[True, False]).drop_duplicates("k")["i"].values
    m = np.zeros(len(score), bool)
    m[idx] = True
    return m


def obs_key(df: pd.DataFrame) -> np.ndarray:
    return pd.factorize(df["trip_id"].astype(str) + ":" + df["obs"].astype(str))[0]


def candidate_metrics(df: pd.DataFrame, prob: np.ndarray) -> dict:
    """df: candidate rows (already restricted to reliable observations)."""
    key = obs_key(df)
    top = top_candidate_mask(key, prob)
    y = df["y"].values
    n_obs = len(np.unique(key))
    out = {
        "top1_acc": float(y[top].sum() / n_obs),
        "macro_f1": float(f1_score(y, top, average="macro")),
        "precision": float(precision_score(y, top, zero_division=0)),
        "recall": float(recall_score(y, top, zero_division=0)),
        "log_loss": float(log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "n_obs": int(n_obs),
    }
    out["cm"] = confusion_matrix(y, top, labels=[0, 1]).tolist()
    return out


def continuity(pred_arcs: np.ndarray, arcs: np.ndarray) -> float:
    """Share of consecutive observation pairs whose predicted arcs are the same
    arc or directly connected (end node of previous == start node of next)."""
    if len(pred_arcs) < 2:
        return 1.0
    a, b = pred_arcs[:-1], pred_arcs[1:]
    ok = (a == b) | (arcs[a, 1] == arcs[b, 0])
    return float(ok.mean())


def route_set_metrics(pred_arcs: np.ndarray, ref_route: np.ndarray, arc_len: np.ndarray) -> dict:
    """Sequence-level comparison of the reconstructed arc set vs the reference route
    (length-weighted precision / recall / F1)."""
    P, R = set(np.unique(pred_arcs).tolist()), set(np.unique(ref_route).tolist())
    inter = P & R
    lp = sum(arc_len[list(P)]) or 1e-9
    lr = sum(arc_len[list(R)]) or 1e-9
    li = sum(arc_len[list(inter)]) if inter else 0.0
    prec, rec = li / lp, li / lr
    return {"route_precision": prec, "route_recall": rec,
            "route_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def traversable_continuity(ctx, pred_arcs: np.ndarray) -> float:
    """Share of consecutive observation pairs whose predicted arcs can actually be
    traversed in sequence on the directed network within a physically plausible
    distance (same arc, or a path end(a) -> start(b) shorter than 3*d_gc + 200 m)."""
    n = len(pred_arcs)
    if n < 2:
        return 1.0
    ok = 0
    for i in range(1, n):
        a, b = int(pred_arcs[i - 1]), int(pred_arcs[i])
        if a == b or ctx.arcs[a, 1] == ctx.arcs[b, 0]:
            ok += 1
            continue
        dgc = float(np.hypot(ctx.px[i] - ctx.px[i - 1], ctx.py[i] - ctx.py[i - 1]))
        dist, _ = ctx.node_dists(int(ctx.arcs[a, 1]), min(5000.0, 3 * dgc + 200.0))
        d = dist.get(int(ctx.arcs[b, 0]))
        ok += d is not None and d <= 3 * dgc + 200.0
    return ok / (n - 1)
