"""Candidate-level feature engineering.

One row = (GPS observation i, candidate arc a). Target = 1 if a is the reference
arc derived for observation i. Features capture geometry (distance), temporal
context (neighbouring points' distance to the same arc), direction agreement
(motion heading vs directed-arc bearing), and local network structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import angle_diff, point_segment
from .network import TripContext

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio",
    "win_dist", "win_rank", "win_minus_min",
    "d_prev", "d_next",
    "head_diff", "head_cos", "head_valid", "speed",
    "arc_len", "proj_t", "end_dist",
    "n_cand", "out_deg", "in_deg", "has_twin",
    "dt_prev", "step_prev",
]
WIN = 5  # +/- observations used for the windowed distance


def build_rows(ctx: TripContext, cands: list[np.ndarray], labels: np.ndarray | None = None) -> pd.DataFrame:
    N = len(ctx.px)
    obs = np.repeat(np.arange(N), [len(c) for c in cands])
    if len(obs) == 0:
        return pd.DataFrame(columns=["trip_id", "obs", "arc", "y"] + FEATURES)
    arc = np.concatenate(cands).astype(np.int64)
    ax, ay, bx, by = ctx.ax[arc], ctx.ay[arc], ctx.bx[arc], ctx.by[arc]

    d, t, tu = point_segment(ctx.px[obs], ctx.py[obs], ax, ay, bx, by)
    # windowed distances: same arc, neighbouring observations
    offs = np.arange(-WIN, WIN + 1)
    W = np.empty((len(obs), len(offs)))
    for k, o in enumerate(offs):
        j = np.clip(obs + o, 0, N - 1)
        W[:, k], _, _ = point_segment(ctx.px[j], ctx.py[j], ax, ay, bx, by)
    win = W.mean(1)
    d_prev, d_next = W[:, WIN - 1], W[:, WIN + 1]

    df = pd.DataFrame({"obs": obs, "arc": arc, "dist": d, "win_dist": win,
                       "d_prev": d_prev, "d_next": d_next})
    g = df.groupby("obs")
    df["dist_rank"] = g["dist"].rank(method="first") - 1
    mn = g["dist"].transform("min")
    df["dist_minus_min"] = df["dist"] - mn
    df["dist_ratio"] = (df["dist"] + 1.0) / (mn + 1.0)
    df["win_rank"] = g["win_dist"].rank(method="first") - 1
    df["win_minus_min"] = df["win_dist"] - g["win_dist"].transform("min")
    df["n_cand"] = g["arc"].transform("size")

    hd = angle_diff(ctx.heading[obs], ctx.abear[arc])
    hv = ctx.heading_valid[obs]
    df["head_diff"] = np.where(hv, hd, 90.0)   # neutral value when stationary
    df["head_cos"] = np.where(hv, np.cos(np.deg2rad(hd)), 0.0)
    df["head_valid"] = hv.astype(float)
    df["speed"] = ctx.speed[obs]
    df["arc_len"] = ctx.alen[arc]
    df["proj_t"] = t
    df["end_dist"] = np.maximum(0, np.maximum(-tu, tu - 1)) * ctx.alen[arc]
    df["out_deg"] = ctx.out_deg[ctx.arcs[arc, 1]]
    df["in_deg"] = ctx.in_deg[ctx.arcs[arc, 0]]
    df["has_twin"] = ctx.has_twin[arc].astype(float)
    df["dt_prev"] = np.minimum(ctx.dt_prev[obs], 600.0)
    df["step_prev"] = np.minimum(ctx.step_prev[obs], 5000.0)
    df["trip_id"] = ctx.trip_id
    df["y"] = (arc == labels[obs]).astype(np.int8) if labels is not None else -1
    return df[["trip_id", "obs", "arc", "y"] + FEATURES]
