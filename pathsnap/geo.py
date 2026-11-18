"""Geometry helpers.

Trips are a few km across, so we use a local azimuthal-equidistant style tangent
plane (equirectangular about the trip centroid). At <20 km extent the scale
error is well below GPS noise (<0.1 %), and it lets us use fast vectorised
Euclidean point-to-segment geometry in metres. Raw degrees are never used as
distances.
"""
from __future__ import annotations

import numpy as np

R_EARTH = 6_371_008.8


class LocalProjector:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = np.deg2rad(1.0) * R_EARTH * np.cos(np.deg2rad(lat0))
        self.ky = np.deg2rad(1.0) * R_EARTH

    def fwd(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def inv(self, x, y):
        return np.asarray(x) / self.kx + self.lon0, np.asarray(y) / self.ky + self.lat0


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def point_segment(px, py, ax, ay, bx, by):
    """Broadcasted point->segment projection.

    Returns (distance, t in [0,1] along segment, unclipped t).
    """
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    L2 = np.where(L2 < 1e-9, 1e-9, L2)
    tu = ((px - ax) * dx + (py - ay) * dy) / L2
    t = np.clip(tu, 0.0, 1.0)
    qx, qy = ax + t * dx, ay + t * dy
    return np.hypot(px - qx, py - qy), t, tu


def bearing(dx, dy):
    """Bearing in degrees (0 = north, clockwise) of vector in local metres."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.where(d > 180, 360 - d, d)
