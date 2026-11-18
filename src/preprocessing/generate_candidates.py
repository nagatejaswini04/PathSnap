
"""
PathSnap - Candidate Road Generation

For every GPS observation:
1. Find nearby road arcs using a KD-tree of arc midpoints.
2. Calculate exact point-to-segment distance.
3. Rank candidate arcs by distance.
4. Save candidate features.

This is an initial candidate-generation experiment.
It does not generate training labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

TRIP_ID = "00000001"

DATA_DIR = ROOT / "data" / "train" / TRIP_ID

OUTPUT_DIR = ROOT / "data" / "processed" / "candidates"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 20
EARTH_RADIUS = 6371000


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    track = pd.read_csv(
        DATA_DIR / f"{TRIP_ID}.track",
        sep=r"\s+",
        header=None
    )

    nodes = pd.read_csv(
        DATA_DIR / f"{TRIP_ID}.nodes",
        sep=r"\s+",
        header=None
    )

    arcs = pd.read_csv(
        DATA_DIR / f"{TRIP_ID}.arcs",
        sep=r"\s+",
        header=None
    )

    track.columns = [
        "longitude", "latitude", "timestamp"
    ]

    nodes.columns = [
        "longitude", "latitude"
    ]

    arcs.columns = [
        "from_node_id", "to_node_id"
    ]

    return track, nodes, arcs


# =========================================================
# PROJECT COORDINATES INTO LOCAL METRES
# =========================================================

def project_coordinates(longitude, latitude, reference_latitude):

    longitude = np.asarray(longitude)
    latitude = np.asarray(latitude)

    x = (
        EARTH_RADIUS
        * np.radians(longitude)
        * np.cos(np.radians(reference_latitude))
    )

    y = EARTH_RADIUS * np.radians(latitude)

    return np.column_stack((x, y))


# =========================================================
# POINT TO SEGMENT DISTANCE
# =========================================================

def point_to_segment(point, start, end):

    segment = end - start

    length_squared = np.dot(segment, segment)

    if length_squared == 0:
        closest = start
        fraction = 0.0

    else:

        fraction = np.dot(
            point - start,
            segment
        ) / length_squared

        fraction = np.clip(fraction, 0.0, 1.0)

        closest = start + fraction * segment

    distance = np.linalg.norm(point - closest)

    return distance, fraction


# =========================================================
# GENERATE CANDIDATES
# =========================================================

def generate_candidates(track, nodes, arcs):

    reference_latitude = track["latitude"].mean()

    node_xy = project_coordinates(
        nodes["longitude"],
        nodes["latitude"],
        reference_latitude
    )

    gps_xy = project_coordinates(
        track["longitude"],
        track["latitude"],
        reference_latitude
    )

    from_ids = arcs["from_node_id"].to_numpy(dtype=int)
    to_ids = arcs["to_node_id"].to_numpy(dtype=int)

    start_points = node_xy[from_ids]
    end_points = node_xy[to_ids]

    midpoints = (start_points + end_points) / 2

    # Build spatial index over road-arc midpoints.
    tree = cKDTree(midpoints)

    results = []

    print("\nGenerating road candidates...")

    for gps_index, gps_point in enumerate(gps_xy):

        # Find the TOP_K nearest arc midpoints.
        k = min(TOP_K, len(midpoints))

        _, candidate_ids = tree.query(
            gps_point,
            k=k
        )

        candidate_ids = np.atleast_1d(candidate_ids)

        observation_candidates = []

        for arc_id in candidate_ids:

            start = start_points[arc_id]
            end = end_points[arc_id]

            distance, fraction = point_to_segment(
                gps_point,
                start,
                end
            )

            observation_candidates.append({
                "trip_id": TRIP_ID,
                "gps_index": gps_index,
                "timestamp": track.iloc[gps_index]["timestamp"],
                "arc_id": int(arc_id),
                "distance_to_arc_m": float(distance),
                "projection_fraction": float(fraction),
                "from_node_id": int(from_ids[arc_id]),
                "to_node_id": int(to_ids[arc_id])
            })

        # Rank by actual point-to-segment distance.
        observation_candidates.sort(
            key=lambda row: row["distance_to_arc_m"]
        )

        for rank, candidate in enumerate(
            observation_candidates,
            start=1
        ):

            candidate["candidate_rank"] = rank

            results.append(candidate)

        if (gps_index + 1) % 100 == 0:
            print(
                f"Processed {gps_index + 1}/"
                f"{len(track)} GPS observations"
            )

    return pd.DataFrame(results)


# =========================================================
# MAIN
# =========================================================

def main():

    print("PATHSNAP CANDIDATE GENERATION")
    print("Trip:", TRIP_ID)

    track, nodes, arcs = load_data()

    print("\nInput sizes:")
    print("GPS observations:", len(track))
    print("Road nodes:", len(nodes))
    print("Road arcs:", len(arcs))

    candidates = generate_candidates(
        track,
        nodes,
        arcs
    )

    output_file = (
        OUTPUT_DIR / f"{TRIP_ID}_candidates.csv"
    )

    candidates.to_csv(
        output_file,
        index=False
    )

    print("\n========== RESULTS ==========")

    print("Total candidate rows:", len(candidates))

    print(
        "GPS observations covered:",
        candidates["gps_index"].nunique()
    )

    print("\nDistance statistics:")

    print(
        candidates.groupby("candidate_rank")[
            "distance_to_arc_m"
        ].describe()
    )

    print("\nFirst candidate records:")

    print(candidates.head(10).to_string(index=False))

    print("\nSaved:", output_file)

    print("\nCANDIDATE GENERATION COMPLETED")


if __name__ == "__main__":
    main()
