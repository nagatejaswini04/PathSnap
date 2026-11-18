
"""
PathSnap - Candidate Generation Experiment

Compare:
    Baseline: TOP_K = 20
    Experiment: TOP_K = 50

Only processes the 17 trips with missing reference arcs.

Existing candidate files are NOT overwritten.
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

BASELINE_DIR = (
    ROOT / "data" / "processed" / "candidates" / "train"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed" / "candidates_k50" / "train"
)

REPORT_DIR = ROOT / "reports" / "candidate_generation"

TRIP_IDS = [
    "00000001",
    "00000010",
    "00000018",
    "00000026",
    "00000028",
    "00000029",
    "00000034",
    "00000043",
    "00000053",
    "00000054",
    "00000055",
    "00000060",
    "00000074",
    "00000078",
    "00000079",
    "00000093",
    "00000097",
]

TOP_K = 50
EARTH_RADIUS = 6371000


# --------------------------------------------------
# COORDINATE PROJECTION
# --------------------------------------------------

def project_coordinates(lon, lat, lon0, lat0):

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
    px, py, ax, ay, bx, by
):

    dx = bx - ax
    dy = by - ay

    denominator = dx**2 + dy**2
    denominator = np.where(
        denominator == 0,
        1e-12,
        denominator
    )

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
# GENERATE TOP 50 CANDIDATES
# --------------------------------------------------

def generate_candidates(trip_id):

    trip_dir = INPUT_DIR / "train" / trip_id

    track = pd.read_csv(trip_dir / "track.csv")
    nodes = pd.read_csv(trip_dir / "nodes.csv")
    arcs = pd.read_csv(trip_dir / "arcs.csv")

    print(f"\nProcessing trip {trip_id}")
    print("GPS observations:", len(track))
    print("Road arcs:", len(arcs))

    lon0 = track["longitude"].mean()
    lat0 = track["latitude"].mean()

    node_x, node_y = project_coordinates(
        nodes["longitude"].to_numpy(),
        nodes["latitude"].to_numpy(),
        lon0,
        lat0
    )

    from_ids = arcs["from_node_id"].to_numpy(dtype=int)
    to_ids = arcs["to_node_id"].to_numpy(dtype=int)

    ax = node_x[from_ids]
    ay = node_y[from_ids]

    bx = node_x[to_ids]
    by = node_y[to_ids]

    midpoint_x = (ax + bx) / 2
    midpoint_y = (ay + by) / 2

    tree = cKDTree(
        np.column_stack((midpoint_x, midpoint_y))
    )

    gps_x, gps_y = project_coordinates(
        track["longitude"].to_numpy(),
        track["latitude"].to_numpy(),
        lon0,
        lat0
    )

    k = min(TOP_K, len(arcs))

    results = []

    for gps_index in range(len(track)):

        px = gps_x[gps_index]
        py = gps_y[gps_index]

        _, indices = tree.query(
            [px, py],
            k=k
        )

        indices = np.atleast_1d(indices)

        distances, fractions = calculate_segment_distances(
            px,
            py,
            ax[indices],
            ay[indices],
            bx[indices],
            by[indices]
        )

        order = np.argsort(distances)

        for rank, position in enumerate(order, start=1):

            arc_index = int(indices[position])

            results.append({
                "trip_id": trip_id,
                "gps_index": gps_index,
                "timestamp": track.iloc[gps_index]["timestamp"],
                "arc_id": arc_index,
                "distance_to_arc_m": float(distances[position]),
                "projection_fraction": float(fractions[position]),
                "from_node_id": int(from_ids[arc_index]),
                "to_node_id": int(to_ids[arc_index]),
                "candidate_rank": rank
            })

        if (gps_index + 1) % 1000 == 0:
            print(
                f"Processed {gps_index + 1}/{len(track)}",
                end="\r"
            )

    candidates = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = (
        OUTPUT_DIR / f"{trip_id}_candidates.csv"
    )

    candidates.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")
    print("Candidate rows:", len(candidates))

    return candidates


# --------------------------------------------------
# CALCULATE REFERENCE-ARC COVERAGE
# --------------------------------------------------

def calculate_coverage(trip_id, candidate_df):

    trip_dir = ROOT / "data" / "train" / trip_id

    route = pd.read_csv(
        trip_dir / f"{trip_id}.route",
        sep=r"\s+",
        header=None
    )

    reference_arcs = set(
        route.iloc[:, 0].astype(int)
    )

    candidate_arcs = set(
        candidate_df["arc_id"].astype(int)
    )

    found = reference_arcs & candidate_arcs
    missing = reference_arcs - candidate_arcs

    coverage = (
        len(found) / len(reference_arcs)
        if reference_arcs else 0
    )

    return {
        "coverage": coverage,
        "found": len(found),
        "missing": len(missing),
        "missing_ids": ",".join(
            map(str, sorted(missing))
        )
    }


# --------------------------------------------------
# RUN EXPERIMENT
# --------------------------------------------------

def main():

    comparison = []

    for trip_id in TRIP_IDS:

        print("\n" + "=" * 50)

        # Existing TOP_K=20 candidates
        baseline_file = (
            BASELINE_DIR / f"{trip_id}_candidates.csv"
        )

        baseline = pd.read_csv(baseline_file)

        baseline_result = calculate_coverage(
            trip_id,
            baseline
        )

        # Generate TOP_K=50 candidates
        experiment = generate_candidates(trip_id)

        experiment_result = calculate_coverage(
            trip_id,
            experiment
        )

        comparison.append({
            "trip_id": trip_id,

            "top20_coverage":
                baseline_result["coverage"],

            "top20_missing":
                baseline_result["missing"],

            "top50_coverage":
                experiment_result["coverage"],

            "top50_missing":
                experiment_result["missing"],

            "top50_missing_arc_ids":
                experiment_result["missing_ids"]
        })

        print(
            f"TOP 20 coverage: "
            f"{baseline_result['coverage']:.2%}"
        )

        print(
            f"TOP 50 coverage: "
            f"{experiment_result['coverage']:.2%}"
        )

    # Save comparison
    comparison_df = pd.DataFrame(comparison)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_file = (
        REPORT_DIR / "top20_vs_top50_comparison.csv"
    )

    comparison_df.to_csv(report_file, index=False)

    # Final summary
    print("\n========== EXPERIMENT RESULTS ==========")

    print(
        "Mean TOP 20 coverage:",
        f"{comparison_df['top20_coverage'].mean():.2%}"
    )

    print(
        "Mean TOP 50 coverage:",
        f"{comparison_df['top50_coverage'].mean():.2%}"
    )

    print(
        "TOP 20 total missing arcs:",
        comparison_df["top20_missing"].sum()
    )

    print(
        "TOP 50 total missing arcs:",
        comparison_df["top50_missing"].sum()
    )

    print(
        "Trips with 100% TOP 50 coverage:",
        (comparison_df["top50_missing"] == 0).sum()
    )

    print("\nComparison saved:")
    print(report_file)

    print("========================================")


if __name__ == "__main__":
    main()
