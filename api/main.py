"""PathSnap FastAPI backend (Laptop 1).

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:  PATHSNAP_EVAL_DIR   folder with the evaluation trips (<id>/<id>.track|nodes|arcs|route)
      PATHSNAP_MODEL_DIR  folder with classifier.joblib + hmm_config.json (default models/final)

The .route files are read ONLY by the scoring step, after the prediction has been
produced from track/nodes/arcs. Reference routes are never returned to the UI.
"""
from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.io import Trip, list_trips, load_trip
from pathsnap.pipeline import PathSnapModel, score

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", "data/raw/student_test")
MODEL_DIR = os.environ.get("PATHSNAP_MODEL_DIR", "models/final")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
MODEL = PathSnapModel(MODEL_DIR)
log.info("Model loaded from %s | HMM config %s | eval dir %s", MODEL_DIR, MODEL.hmm, EVAL_DIR)


class PredictRequest(BaseModel):
    trip_id: str
    use_hmm: bool = True


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    log.info("%s %s from %s -> %s (%.0f ms)", request.method, request.url.path,
             request.client.host if request.client else "?", resp.status_code, 1000 * (time.time() - t0))
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "eval_dir": EVAL_DIR, "hmm": MODEL.hmm}


@app.get("/trips")
def trips():
    out = []
    for tid in list_trips(EVAL_DIR):
        p = os.path.join(EVAL_DIR, tid, tid + ".track")
        n = sum(1 for _ in open(p)) if os.path.exists(p) else 0
        out.append({"trip_id": tid, "n_obs": n})
    return {"trips": out}


def _line(ctx, e):
    a, b = ctx.arcs[e]
    return [[float(ctx.nodes_ll[a, 1]), float(ctx.nodes_ll[a, 0])], [float(ctx.nodes_ll[b, 1]), float(ctx.nodes_ll[b, 0])]]


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in list_trips(EVAL_DIR):
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    log.info("PREDICT request trip=%s use_hmm=%s", req.trip_id, req.use_hmm)
    t0 = time.time()
    raw = load_trip(EVAL_DIR, req.trip_id, with_route=False)        # prediction inputs only
    res = MODEL.predict_trip(Trip(raw.trip_id, raw.track, raw.nodes, raw.arcs), use_hmm=req.use_hmm)
    ctx, pred = res["ctx"], res["pred"]
    t_pred = time.time() - t0

    # ---- server-side scoring (reference route loaded only now, never returned)
    metrics = None
    ref_path = os.path.join(EVAL_DIR, req.trip_id, req.trip_id + ".route")
    if os.path.exists(ref_path):
        ref = np.loadtxt(ref_path, dtype=np.int64, ndmin=1)
        metrics = score(ctx, pred, res["route"], ref)
        clf_metrics = score(ctx, res["clf_pred"], [int(a) for a in res["clf_pred"]], ref)
        metrics["classifier_only_top1_accuracy"] = clf_metrics["top1_accuracy"]
    step = max(1, len(ctx.track) // 2000)
    out = {
        "trip_id": req.trip_id,
        "n_obs": int(len(ctx.track)),
        "gps": [[float(la), float(lo)] for lo, la in ctx.track[::step, :2]],
        "route": [_line(ctx, e) for e in res["route"]],
        "route_arc_ids": [int(e) for e in res["route"]],
        "n_route_arcs": len(res["route"]),
        "hmm_breaks": int(res["breaks"]),
        "metrics": metrics,
        "timing_s": {"classify": round(res["t_classify"], 3), "decode": round(res["t_decode"], 3),
                     "predict_total": round(t_pred, 3), "total": round(time.time() - t0, 3)},
    }
    if metrics:
        log.info("RESULT trip=%s obs=%d scored=%d top1=%.4f (classifier-only %.4f) route_F1=%.4f "
                 "continuity=%.4f breaks=%d time=%.2fs", req.trip_id, metrics["n_obs"], metrics["n_scored_obs"],
                 metrics["top1_accuracy"], metrics["classifier_only_top1_accuracy"], metrics["route_f1"],
                 metrics["continuity_traversable"], res["breaks"], out["timing_s"]["total"])
    return out
