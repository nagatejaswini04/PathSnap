"""End-to-end inference pipeline (shared by evaluation scripts and the API).

trip files (track/nodes/arcs) -> clean -> candidates -> features -> classifier
probabilities -> HMM/Viterbi -> per-observation arcs -> connected route.
The reference route is NOT read here; scoring is done separately (score_trip).
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np

from .features import FEATURES, build_rows
from .hmm import reconstruct_route, viterbi
from .io import Trip, load_trip
from .metrics import continuity, route_set_metrics, traversable_continuity
from .network import TripContext

LABEL_OK_M = 50.0


class PathSnapModel:
    def __init__(self, model_dir: str = "models/final"):
        art = joblib.load(os.path.join(model_dir, "classifier.joblib"))
        self.clf, self.features = art["model"], art["features"]
        cfg_path = os.path.join(model_dir, "hmm_config.json")
        self.hmm = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        assert self.features == FEATURES, "feature list mismatch between artifact and code"

    def predict_trip(self, trip: Trip, use_hmm: bool = True) -> dict:
        t0 = time.time()
        trip = Trip(trip.trip_id, trip.track, trip.nodes, trip.arcs, None)  # never pass labels in
        ctx = TripContext(trip)
        cands = ctx.candidates()
        df = build_rows(ctx, cands, None)
        p = self.clf.predict_proba(df[FEATURES].values)[:, 1] if len(df) else np.array([])
        probs, k = [], 0
        for c in cands:
            probs.append(p[k:k + len(c)]); k += len(c)
        clf_pred = np.array([c[int(np.argmax(q))] if len(c) else -1 for c, q in zip(cands, probs)])
        t1 = time.time()
        if use_hmm:
            pred, breaks = viterbi(ctx, cands, probs, self.hmm.get("params"), self.hmm.get("emission", "clf"))
        else:
            pred, breaks = clf_pred, 0
        route = reconstruct_route(ctx, pred)
        t2 = time.time()
        return {"ctx": ctx, "cands": cands, "probs": probs, "clf_pred": clf_pred, "pred": pred,
                "route": route, "breaks": breaks, "t_classify": t1 - t0, "t_decode": t2 - t1}


def score(ctx: TripContext, pred: np.ndarray, route_pred: list[int], ref_route: np.ndarray) -> dict:
    """Server-side scoring against the reference route (labels derived by DP alignment)."""
    tmp = TripContext.__new__(TripContext)
    tmp.__dict__.update(ctx.__dict__)
    tmp.route = ref_route
    lab, _, dl = tmp.point_labels()
    ok = dl <= LABEL_OK_M
    out = {
        "n_obs": int(len(pred)), "n_scored_obs": int(ok.sum()),
        "top1_accuracy": float(np.mean(pred[ok] == lab[ok])) if ok.any() else None,
        "top1_accuracy_all_obs": float(np.mean(pred == lab)),
        "continuity_adjacent": continuity(pred, ctx.arcs),
        "continuity_traversable": traversable_continuity(ctx, pred),
    }
    out.update(route_set_metrics(np.array(route_pred, dtype=np.int64), ref_route, ctx.alen))
    return out
