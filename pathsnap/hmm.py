"""HMM + Viterbi route reconstruction on top of classifier candidate scores.

Hidden state at observation i : one candidate arc c in C_i, with the GPS point's
                                projection offset s_i(c) along that arc.
Emission  log P(z_i | c)      : w_e * log p_clf(c)  (classifier probability), or
                                a Gaussian on perpendicular distance (baseline).
Transition log P(c' -> c)     : -|d_route - d_gc| / beta   (Newson & Krumm style)
    d_gc    = straight-line distance between consecutive GPS points
    d_route = network distance from (c', s') to (c, s) over the directed graph
              (same arc: s - s', small backward moves tolerated as GPS jitter)
    unreachable within the search cutoff -> transition impossible (-inf)
Breaks: if dt > gap_s, or no finite transition exists (disconnected segments),
        the chain is cut and Viterbi restarts from emissions only, so a single
        bad step cannot corrupt the whole route.
"""
from __future__ import annotations

import numpy as np

from .network import TripContext

DEFAULT_HMM = {
    "beta": 10.0,          # metres, transition scale
    "w_emit": 1.0,         # weight on log classifier probability
    "sigma": 8.0,          # metres, only for emission="gauss"
    "gap_s": 60.0,         # time gap that forces a chain break
    "backtrack_tol": 15.0, # metres of backward motion on same arc treated as noise
    "max_cutoff": 5000.0,
}


def viterbi(ctx: TripContext, cands: list[np.ndarray], probs: list[np.ndarray], params: dict | None = None,
            emission: str = "clf"):
    """Return (predicted arc per observation [-1 if no candidate], number of chain breaks)."""
    p = {**DEFAULT_HMM, **(params or {})}
    N = len(cands)
    pred = np.full(N, -1, dtype=np.int64)
    states = []
    for i in range(N):
        c = np.asarray(cands[i], dtype=np.int64)
        if len(c) == 0:
            states.append(None); continue
        d, t = ctx.project(i, c)
        if emission == "clf":
            e = p["w_emit"] * np.log(np.clip(probs[i], 1e-6, 1.0))
        else:
            e = -0.5 * (d / p["sigma"]) ** 2
        states.append((c, t * ctx.alen[c], e))

    breaks = 0
    seg_start = None
    delta, back = None, []

    def flush(end):
        """Backtrack the current segment [seg_start, end)."""
        if seg_start is None or delta is None:
            return
        k = int(np.argmax(delta))
        for j in range(end - 1, seg_start - 1, -1):
            st = states[j]
            if st is None:
                continue
            pred[j] = st[0][k]
            if j > seg_start:
                k = int(back[j - seg_start][k])

    prev_i = None
    for i in range(N):
        st = states[i]
        if st is None:
            continue  # observation without candidates: carried over below
        if delta is not None:
            gap = ctx.t[i] - ctx.t[prev_i] > p["gap_s"]
            T = None if gap else _transition_idx(ctx, i, prev_i, states, p)
            if T is not None:
                score = delta[:, None] + T
                bp = np.argmax(score, axis=0)
                best = score[bp, np.arange(score.shape[1])]
            if T is None or not np.isfinite(best).any():
                breaks += 1
                flush(prev_i + 1)
                seg_start, delta, back = None, None, []
            else:
                # fill skipped (no-candidate) observations with a back-pointer identity
                for _ in range(prev_i + 1, i):
                    back.append(None)
                back.append(bp)
                delta = best + st[2]
                prev_i = i
                continue
        seg_start, delta, back = i, st[2].copy(), [None]
        prev_i = i
    flush(prev_i + 1 if prev_i is not None else 0)
    # observations with no candidates inherit the previous prediction
    last = -1
    for i in range(N):
        if pred[i] == -1:
            pred[i] = last
        else:
            last = pred[i]
    return pred, breaks


def _transition_idx(ctx, i, j, states, p):
    """Log transition matrix between observation j (previous with candidates) and i."""
    prev, cur = states[j], states[i]
    (pa, ps, _), (ca, cs, _) = prev, cur
    dgc = float(np.hypot(ctx.px[i] - ctx.px[j], ctx.py[i] - ctx.py[j]))
    cutoff = min(p["max_cutoff"], 3 * dgc + 200.0)
    T = np.full((len(pa), len(ca)), -np.inf)
    from_nodes = ctx.arcs[ca, 0]
    for u, (a, s) in enumerate(zip(pa, ps)):
        rem = ctx.alen[a] - s
        dist, _ = ctx.node_dists(int(ctx.arcs[a, 1]), cutoff)
        for v, (b, s2) in enumerate(zip(ca, cs)):
            if a == b:
                dr = s2 - s
                if dr < -p["backtrack_tol"]:
                    continue
                dr = abs(dr)
            else:
                dn = dist.get(int(from_nodes[v]))
                if dn is None:
                    continue
                dr = rem + dn + s2
            T[u, v] = -abs(dr - dgc) / p["beta"]
    return T


def reconstruct_route(ctx: TripContext, pred: np.ndarray, plausibility: bool = True) -> list[int]:
    """Turn per-observation arcs into an arc sequence by filling the shortest path
    between consecutive distinct predicted arcs.

    With plausibility=True a connecting path is only inserted if its length is
    consistent with the GPS displacement between the two observations
    (<= 3 * d_gc + 200 m, the same bound the transition model uses). This stops
    chain breaks (e.g. a vehicle parked during a long gap and then leaving on the
    reverse twin arc) from being "repaired" with long detours that were never
    driven; the route is left discontinuous there instead.
    """
    out, prev, prev_i = [], None, None
    for i, a in enumerate(pred):
        a = int(a)
        if a < 0:
            continue
        if a == prev:
            prev_i = i  # last observation on the previous arc
            continue
        if prev is not None:
            dgc = float(np.hypot(ctx.px[i] - ctx.px[prev_i], ctx.py[i] - ctx.py[prev_i]))
            mid = ctx.path_arcs(prev, a)
            if mid and (not plausibility or ctx.alen[mid].sum() <= 3 * dgc + 200.0):
                out.extend(mid)
        out.append(a)
        prev, prev_i = a, i
    return out
