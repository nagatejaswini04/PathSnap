"""Raw-data readers for PathSnap trips.

Each trip folder <root>/<trip_id>/ contains:
  <trip_id>.track  lon lat t        (GPS observations, ~1 Hz)
  <trip_id>.nodes  lon lat          (row index = node_id)
  <trip_id>.arcs   from to          (row index = arc_id, directed)
  <trip_id>.route  arc_id           (reference arc sequence; optional / server-side only)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray          # (N,3) lon, lat, t
    nodes: np.ndarray          # (V,2) lon, lat
    arcs: np.ndarray           # (E,2) from, to
    route: np.ndarray | None = None  # (R,) arc ids
    meta: dict = field(default_factory=dict)


def list_trips(root: str) -> list[str]:
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def _load(path, dtype=float, ndmin=2):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    return np.loadtxt(path, dtype=dtype, ndmin=ndmin)


def load_trip(root: str, trip_id: str, with_route: bool = True) -> Trip:
    p = os.path.join(root, trip_id, trip_id)
    track = _load(p + ".track")
    nodes = _load(p + ".nodes")
    arcs = _load(p + ".arcs", dtype=np.int64)
    route = _load(p + ".route", dtype=np.int64, ndmin=1) if with_route else None
    return Trip(trip_id, track, nodes, arcs, route)


def clean_track(track: np.ndarray) -> tuple[np.ndarray, dict]:
    """Drop invalid / duplicate GPS rows and enforce increasing timestamps.

    Returns cleaned track and a dict describing what was removed (for EDA logs).
    """
    info = {"n_raw": int(len(track))}
    ok = np.all(np.isfinite(track), axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    ok &= ~((track[:, 0] == 0) & (track[:, 1] == 0))
    info["n_invalid"] = int((~ok).sum())
    tr = track[ok]
    order = np.argsort(tr[:, 2], kind="stable")
    info["n_out_of_order"] = int((order != np.arange(len(order))).sum())
    tr = tr[order]
    keep = np.r_[True, np.diff(tr[:, 2]) > 0]
    info["n_dup_time"] = int((~keep).sum())
    tr = tr[keep]
    info["n_clean"] = int(len(tr))
    return tr, info
