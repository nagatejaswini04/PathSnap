"""Per-trip road-network context: projection, spatial index, candidates, labels.

Every trip ships its own directed road network, so all structures are built
per trip. Only arcs near the trajectory (bbox + margin) are indexed.
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree

from .geo import LocalProjector, angle_diff, bearing, point_segment
from .io import Trip, clean_track

DEFAULT_CFG = {
    "bbox_margin_m": 500.0,     # arcs considered around the trajectory
    "densify_step_m": 15.0,     # spacing of arc sample points in the KD-tree
    "cand_radius_m": 60.0,      # candidate search radius
    "cand_k": 8,                # max candidates per observation
    "fallback_radius_m": 300.0, # if nothing within radius, take nearest within this
    "fallback_k": 3,
    "label_skip_pen_m": 2.0,    # DP penalty per skipped route arc
}


class TripContext:
    def __init__(self, trip: Trip, cfg: dict | None = None):
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.trip_id = trip.trip_id
        self.track, self.clean_info = clean_track(trip.track)
        self.route = trip.route
        lon, lat, t = self.track.T
        self.t = t
        self.proj = LocalProjector(np.median(lon), np.median(lat))
        self.px, self.py = self.proj.fwd(lon, lat)
        self.nodes_ll = trip.nodes
        self.nx, self.ny = self.proj.fwd(trip.nodes[:, 0], trip.nodes[:, 1])
        self.arcs = trip.arcs
        a, b = trip.arcs[:, 0], trip.arcs[:, 1]
        self.ax, self.ay, self.bx, self.by = self.nx[a], self.ny[a], self.nx[b], self.ny[b]
        self.alen = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.abear = bearing(self.bx - self.ax, self.by - self.ay)
        V = len(trip.nodes)
        self.out_deg = np.bincount(a, minlength=V)
        self.in_deg = np.bincount(b, minlength=V)
        pairs = set(zip(a.tolist(), b.tolist()))
        self.has_twin = np.array([(y, x) in pairs for x, y in zip(a.tolist(), b.tolist())])
        self._build_index()
        self._adj = None
        self._sp_cache: dict = {}
        self._motion()

    # ------------------------------------------------------------------ index
    def _build_index(self):
        m = self.cfg["bbox_margin_m"]
        x0, x1 = self.px.min() - m, self.px.max() + m
        y0, y1 = self.py.min() - m, self.py.max() + m
        sx0, sx1 = np.minimum(self.ax, self.bx), np.maximum(self.ax, self.bx)
        sy0, sy1 = np.minimum(self.ay, self.by), np.maximum(self.ay, self.by)
        active = (sx1 >= x0) & (sx0 <= x1) & (sy1 >= y0) & (sy0 <= y1)
        self.active = np.flatnonzero(active)
        step = self.cfg["densify_step_m"]
        nseg = np.maximum(1, np.ceil(self.alen[self.active] / step).astype(int))
        ids = np.repeat(self.active, nseg + 1)
        frac = np.concatenate([np.linspace(0, 1, n + 1) for n in nseg]) if len(nseg) else np.array([])
        sx = self.ax[ids] + frac * (self.bx[ids] - self.ax[ids])
        sy = self.ay[ids] + frac * (self.by[ids] - self.ay[ids])
        self.sample_arc = ids
        self.tree = cKDTree(np.c_[sx, sy]) if len(ids) else None

    def arcs_near(self, x, y, r):
        if self.tree is None:
            return np.array([], int)
        idx = self.tree.query_ball_point([x, y], r + self.cfg["densify_step_m"] / 2 + 1)
        return np.unique(self.sample_arc[idx]) if idx else np.array([], int)

    # ------------------------------------------------------------ candidates
    def candidates(self):
        """Return list (per observation) of candidate arc-id arrays, sorted by distance."""
        R, K = self.cfg["cand_radius_m"], self.cfg["cand_k"]
        FR, FK = self.cfg["fallback_radius_m"], self.cfg["fallback_k"]
        out = []
        for x, y in zip(self.px, self.py):
            arcs = self.arcs_near(x, y, R)
            if len(arcs):
                d, _, _ = point_segment(x, y, self.ax[arcs], self.ay[arcs], self.bx[arcs], self.by[arcs])
                keep = d <= R
                arcs, d = arcs[keep], d[keep]
            if len(arcs) == 0:
                arcs = self.arcs_near(x, y, FR)
                if len(arcs):
                    d, _, _ = point_segment(x, y, self.ax[arcs], self.ay[arcs], self.bx[arcs], self.by[arcs])
                    o = np.argsort(d)[:FK]
                    arcs, d = arcs[o], d[o]
                    keep = d <= FR
                    arcs, d = arcs[keep], d[keep]
                out.append(arcs)
                continue
            o = np.argsort(d, kind="stable")[:K]
            out.append(arcs[o])
        return out

    # ---------------------------------------------------------------- labels
    def point_labels(self):
        """Derive a reference arc for each GPS observation.

        The .route file is an ordered arc sequence, not per-point labels. We align
        observations to the route with a monotone DP (route position can only move
        forward), minimising total point-to-arc distance with a small penalty for
        skipping route arcs. Returns (arc_id per obs, route position per obs,
        distance to that arc).
        """
        r = self.route
        N, M = len(self.px), len(r)
        pen = self.cfg["label_skip_pen_m"]
        D, _, _ = point_segment(self.px[:, None], self.py[:, None],
                                self.ax[r][None], self.ay[r][None], self.bx[r][None], self.by[r][None])
        jidx = np.arange(M)
        back = np.zeros((N, M), dtype=np.int32)
        P = D[0] + pen * jidx  # start anywhere, pay for skipped leading arcs
        back[0] = -1
        for i in range(1, N):
            q = P - pen * jidx
            pm = np.minimum.accumulate(q)
            am = np.zeros(M, dtype=np.int32)
            # vectorised running argmin of the prefix (used shifted: k < j)
            is_new = np.r_[True, q[1:] < pm[:-1]]
            best_idx = np.maximum.accumulate(np.where(is_new, jidx, 0))
            adv = np.full(M, np.inf)
            adv[1:] = pm[:-1] + pen * (jidx[1:] - 1)
            am[1:] = best_idx[:-1]
            stay = P
            use_adv = adv < stay
            P = np.where(use_adv, adv, stay) + D[i]
            back[i] = np.where(use_adv, am, jidx)
        P = P + pen * (M - 1 - jidx)  # pay for skipped trailing arcs
        j = int(np.argmin(P))
        pos = np.zeros(N, dtype=np.int64)
        for i in range(N - 1, -1, -1):
            pos[i] = j
            if i > 0:
                j = int(back[i, j])
        return r[pos], pos, D[np.arange(N), pos]

    # ---------------------------------------------------------------- motion
    def _motion(self, w=2):
        N = len(self.px)
        i0 = np.clip(np.arange(N) - w, 0, N - 1)
        i1 = np.clip(np.arange(N) + w, 0, N - 1)
        dx, dy = self.px[i1] - self.px[i0], self.py[i1] - self.py[i0]
        disp = np.hypot(dx, dy)
        dt = np.maximum(self.t[i1] - self.t[i0], 1e-6)
        self.speed = disp / dt
        self.heading = bearing(dx, dy)
        self.heading_valid = disp > 3.0
        self.dt_prev = np.r_[1.0, np.diff(self.t)]
        self.step_prev = np.r_[0.0, np.hypot(np.diff(self.px), np.diff(self.py))]

    # ----------------------------------------------------------------- graph
    def adjacency(self):
        if self._adj is None:
            adj: dict[int, list] = {}
            for e in self.active:
                u, v = int(self.arcs[e, 0]), int(self.arcs[e, 1])
                adj.setdefault(u, []).append((v, float(self.alen[e]), int(e)))
            self._adj = adj
        return self._adj

    def node_dists(self, src: int, cutoff: float):
        """Bounded Dijkstra over active arcs (cached by source/cutoff bucket).

        Returns (dist dict node->metres, pred dict node->(prev_node, arc_id)).
        """
        key = (src, int(np.ceil(max(cutoff, 1.0) / 250.0)))
        hit = self._sp_cache.get(key)
        if hit is not None:
            return hit
        cutoff = key[1] * 250.0
        adj = self.adjacency()
        dist, pred = {src: 0.0}, {}
        h = [(0.0, src)]
        while h:
            d, u = heapq.heappop(h)
            if d > dist.get(u, np.inf):
                continue
            for v, w, e in adj.get(u, ()):
                nd = d + w
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    pred[v] = (u, e)
                    heapq.heappush(h, (nd, v))
        if len(self._sp_cache) > 200_000:
            self._sp_cache.clear()
        self._sp_cache[key] = (dist, pred)
        return dist, pred

    def path_arcs(self, a: int, b: int, cutoff: float = 5000.0) -> list[int] | None:
        """Arc ids strictly between arc a and arc b on the shortest connecting path."""
        if a == b or self.arcs[a, 1] == self.arcs[b, 0]:
            return []
        src, dst = int(self.arcs[a, 1]), int(self.arcs[b, 0])
        dist, pred = self.node_dists(src, cutoff)
        if dst not in dist:
            return None
        out, v = [], dst
        while v != src:
            u, e = pred[v]
            out.append(e)
            v = u
        return out[::-1]

    def project(self, i: int, arcs: np.ndarray):
        d, t, _ = point_segment(self.px[i], self.py[i], self.ax[arcs], self.ay[arcs], self.bx[arcs], self.by[arcs])
        return d, t
