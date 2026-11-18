
"""
PathSnap - Batch Road Candidate Generation

Generates the top 20 nearby road-segment candidates
for every GPS observation across all trips.

Uses:
    - Processed track.csv
    - Processed nodes.csv
    - Processed arcs.csv

Does NOT use reference route labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data" / "processed" / "trips"
OUTPUT_DIR = ROOT / "data" / "processed" / "candidates"

PARTITIONS = ["train", "student_test"]

TOP_K = 20
GPS_BATCH_SIZE = 5000
EARTH_RADIUS = 6371000


# --------------------------------------------------
# COORDINATE PROJECTION
# --------------------------------------------------

def project_coordinates(lon, lat, lon0, lat0):
    """
    Convert longitude/latitude to local metric coordinates.
    Suitable for local trip-level road networks.
    """

    x = (
        EARTH_RADIUS
        * np.radians(lon - lon0)
        * np.cos(np.radians(lat0))
    )

    y = EARTH_RADIUS * np.radians(lat - lat0)

    return x, y


# --------------------------------------------------
# POINT-TO-SEGMENT DISTANCE
# --------------------------------------------------

def calculate_segment_distances(
    px, py,
    ax, ay,
    bx, by
):
    """
    Calculate distance from one GPS point to
    multiple road segments.

    Returns:
        distances
        projection fractions
    """

    dx = bx - ax
    dy = by - ay

    denominator = dx**2 + dy**2
    denominator = np.where(denominator == 0, 1e-12, denominator)

    fraction = (
        (px - ax) * dx + (py - ay) * dy
    ) / denominator

    fraction = np.clip(fraction, 0, 1)

    nearest_x = ax + fraction * dx
    nearest_y = ay + fraction * dy

    distances = np.sqrt(
        (px - nearest_x)**2
        + (py - nearest_y)**2
    )

    return distances, fraction


# --------------------------------------------------
# PROCESS ONE TRIP
# --------------------------------------------------

def generate_trip_candidates(partition, trip_id):

    trip_dir = INPUT_DIR / partition / trip_id

    track = pd.read_csv(trip_dir / "track.csv")
    nodes = pd.read_csv(trip_dir / "nodes.csv")
    arcs = pd.read_csv(trip_dir / "arcs.csv")

    print(f"\nProcessing {partition}/{trip_id}")
    print(
        f"GPS: {len(track)}, "
        f"Nodes: {len(nodes)}, "
        f"Arcs: {len(arcs)}"
    )

    # Coordinate origin for local projection
    lon0 = float(track["longitude"].mean())
    lat0 = float(track["latitude"].mean())

    # Project road nodes
    node_x, node_y = project_coordinates(
        nodes["longitude"].to_numpy(),
        nodes["latitude"].to_numpy(),
        lon0,
        lat0
    )

    # Get start and end coordinates for every arc
    from_ids = arcs["from_node_id"].to_numpy(dtype=int)
    to_ids = arcs["to_node_id"].to_numpy(dtype=int)

    ax = node_x[from_ids]
    ay = node_y[from_ids]

    bx = node_x[to_ids]
    by = node_y[to_ids]

    # Calculate arc midpoints
    midpoint_x = (ax + bx) / 2
    midpoint_y = (ay + by) / 2

    midpoint_coordinates = np.column_stack(
        (midpoint_x, midpoint_y)
    )

    # Build spatial search tree
    tree = cKDTree(midpoint_coordinates)

    # Project GPS coordinates
    gps_x, gps_y = project_coordinates(
        track["longitude"].to_numpy(),
        track["latitude"].to_numpy(),
        lon0,
        lat0
    )

    # Candidate count cannot exceed available arcs
    k = min(TOP_K, len(arcs))

    all_results = []

    # Process GPS observations in batches
    for start in range(0, len(track), GPS_BATCH_SIZE):

        end = min(start + GPS_BATCH_SIZE, len(track))

        batch_x = gps_x[start:end]
        batch_y = gps_y[start:end]

        gps_coordinates = np.column_stack(
            (batch_x, batch_y)
        )

        # Find nearby arc midpoints
        _, candidate_indices = tree.query(
            gps_coordinates,
            k=k
        )

        candidate_indices = np.asarray(candidate_indices)

        if candidate_indices.ndim == 1:
            candidate_indices = candidate_indices.reshape(-1, 1)

        for local_index, arc_indices in enumerate(candidate_indices):

            gps_index = start + local_index

            px = batch_x[local_index]
            py = batch_y[local_index]

            distances, fractions = calculate_segment_distances(
                px,
                py,
                ax[arc_indices],
                ay[arc_indices],
                bx[arc_indices],
                by[arc_indices]
            )

            # Sort by actual point-to-segment distance
            order = np.argsort(distances)

            arc_indices = arc_indices[order]
            distances = distances[order]
            fractions = fractions[order]

            for rank, (arc_index, distance, fraction) in enumerate(
                zip(arc_indices, distances, fractions),
                start=1
            ):

                all_results.append({
                    "trip_id": trip_id,
                    "gps_index": gps_index,
                    "timestamp": track.iloc[gps_index]["timestamp"],
                    "arc_id": int(arc_index),
                    "distance_to_arc_m": float(distance),
                    "projection_fraction": float(fraction),
                    "from_node_id": int(from_ids[arc_index]),
                    "to_node_id": int(to_ids[arc_index]),
                    "candidate_rank": rank
                })

        print(
            f"  Processed {end}/{len(track)} GPS observations",
            end="\r"
        )

    # Convert results to DataFrame
    candidates = pd.DataFrame(all_results)

    # Save candidates
    output_folder = OUTPUT_DIR / partition
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / f"{trip_id}_candidates.csv"

    candidates.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")
    print(f"Candidate rows: {len(candidates)}")

    return {
        "partition": partition,
        "trip_id": trip_id,
        "gps_observations": len(track),
        "candidate_rows": len(candidates),
        "status": "success"
    }


# --------------------------------------------------
# PROCESS ALL TRIPS
# --------------------------------------------------

def main():

    summaries = []
    failures = []

    for partition in PARTITIONS:

        partition_dir = INPUT_DIR / partition

        trip_folders = sorted(
            folder for folder in partition_dir.iterdir()
            if folder.is_dir()
        )

        print(f"\n========== {partition.upper()} ==========")
        print(f"Total trips: {len(trip_folders)}")

        for trip_folder in trip_folders:

            try:

                result = generate_trip_candidates(
                    partition,
                    trip_folder.name
                )

                summaries.append(result)

            except Exception as error:

                print(f"\nFAILED: {trip_folder.name}: {error}")

                failures.append({
                    "partition": partition,
                    "trip_id": trip_folder.name,
                    "error": str(error)
                })

    # Save batch summary
    report_dir = ROOT / "reports" / "candidate_generation"
    report_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(summaries).to_csv(
        report_dir / "candidate_generation_summary.csv",
        index=False
    )

    pd.DataFrame(failures).to_csv(
        report_dir / "candidate_generation_failures.csv",
        index=False
    )

    print("\n========== FINAL RESULTS ==========")
    print("Successful trips:", len(summaries))
    print("Failed trips:", len(failures))

    if summaries:
        print(
            "Total candidate rows:",
            sum(item["candidate_rows"] for item in summaries)
        )

    print("===================================")


if __name__ == "__main__":
    main()
