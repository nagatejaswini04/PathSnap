
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRIP_ID = "00000028"

trip_dir = ROOT / "data" / "train" / TRIP_ID

track = pd.read_csv(
    trip_dir / f"{TRIP_ID}.track",
    sep=r"\s+", header=None
)

nodes = pd.read_csv(
    trip_dir / f"{TRIP_ID}.nodes",
    sep=r"\s+", header=None
)

arcs = pd.read_csv(
    trip_dir / f"{TRIP_ID}.arcs",
    sep=r"\s+", header=None
)

route = pd.read_csv(
    trip_dir / f"{TRIP_ID}.route",
    sep=r"\s+", header=None
).iloc[:, 0].astype(int).to_numpy()

reference_lat = track.iloc[:, 1].mean()

def project(lon, lat):
    return np.column_stack((
        np.asarray(lon) * 111320 * np.cos(
            np.radians(reference_lat)
        ),
        np.asarray(lat) * 110540
    ))

gps = project(track.iloc[:, 0], track.iloc[:, 1])
node_xy = project(nodes.iloc[:, 0], nodes.iloc[:, 1])

route_from = arcs.iloc[route, 0].to_numpy(dtype=int)
route_to = arcs.iloc[route, 1].to_numpy(dtype=int)

route_start = node_xy[route_from]
route_end = node_xy[route_to]

# Check whether consecutive route arcs connect.
connected = route_to[:-1] == route_from[1:]

print("\n========== ROUTE CONTINUITY ==========")
print("Route arc occurrences:", len(route))
print("Transitions:", len(connected))
print("Connected transitions:", connected.sum())
print("Disconnected transitions:", (~connected).sum())
print("Continuity ratio:", connected.mean())

# Calculate route arc midpoint positions.
midpoints = (route_start + route_end) / 2

# Check how GPS progresses along the route.
gps_indices = np.arange(len(gps))

# For each GPS point, find the nearest route arc midpoint.
from scipy.spatial import cKDTree

tree = cKDTree(midpoints)
distances, route_positions = tree.query(gps)

print("\n========== GPS TO ROUTE PROGRESSION ==========")
print("GPS observations:", len(gps))
print("Mean nearest route midpoint distance (m):",
      distances.mean())
print("Median nearest route midpoint distance (m):",
      np.median(distances))

print("Route positions assigned to GPS observations:")
print("First 20:", route_positions[:20].tolist())

# Check whether assigned route positions generally progress.
differences = np.diff(route_positions)

print("Non-decreasing GPS-to-route transitions:",
      np.sum(differences >= 0), "/", len(differences))

print("Negative route-position transitions:",
      np.sum(differences < 0))

print("\nDiagnostic complete.")
