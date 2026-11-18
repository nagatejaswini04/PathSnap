
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = ROOT / "data" / "train"
CANDIDATE_DIR = ROOT / "data" / "processed" / "candidates" / "train"

OUTPUT_DIR = ROOT / "data" / "processed" / "training" / "train"
REPORT_DIR = ROOT / "reports" / "candidate_generation"

MAX_ALIGNMENT_DISTANCE_M = 100.0
MAX_ROUTE_ADVANCE_PER_GPS = 10


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_raw(path):
    return pd.read_csv(
        path,
        sep=r"[\s,]+",
        header=None,
        engine="python",
        comment="#"
    )


def haversine_m(lat1, lon1, lat2, lon2):
    radius = 6371000.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * radius * np.arcsin(np.sqrt(a))


def project_coordinates(lon, lat, center_lat):
    radius = 6371000.0
    center = np.radians(center_lat)

    x = radius * np.radians(np.asarray(lon)) * np.cos(center)
    y = radius * np.radians(np.asarray(lat))

    return np.column_stack((x, y))


# ============================================================
# DISTANCE FROM GPS TO REFERENCE ROUTE
# ============================================================

def calculate_route_distances(gps_xy, starts, ends):

    n_gps = len(gps_xy)
    n_route = len(starts)

    distances = np.empty(
        (n_gps, n_route),
        dtype=np.float32
    )

    for j in range(n_route):

        vector = ends[j] - starts[j]
        length_squared = np.dot(vector, vector)

        relative = gps_xy - starts[j]

        if length_squared == 0:
            projection = np.zeros(n_gps)
        else:
            projection = (relative @ vector) / length_squared

        projection = np.clip(projection, 0, 1)

        closest = starts[j] + projection[:, None] * vector

        distances[:, j] = np.linalg.norm(
            gps_xy - closest,
            axis=1
        )

    return distances


# ============================================================
# CHRONOLOGICAL ALIGNMENT
# ============================================================

def align_gps_to_route(distances):

    n_gps, n_route = distances.shape

    dp = np.full(
        (n_gps, n_route),
        np.inf,
        dtype=np.float32
    )

    back = np.full(
        (n_gps, n_route),
        -1,
        dtype=np.int32
    )

    dp[0, 0] = distances[0, 0]

    for i in range(1, n_gps):

        previous = dp[i - 1]

        best_cost = np.full(n_route, np.inf)
        best_previous = np.full(n_route, -1, dtype=np.int32)

        for shift in range(MAX_ROUTE_ADVANCE_PER_GPS + 1):

            candidate = np.full(n_route, np.inf)

            if shift == 0:
                candidate[:] = previous
            else:
                candidate[shift:] = previous[:-shift]

            better = candidate < best_cost

            best_cost[better] = candidate[better]

            best_previous[better] = (
                np.arange(n_route)[better] - shift
            )

        dp[i] = best_cost + distances[i]
        back[i] = best_previous

    position = int(np.argmin(dp[-1]))

    aligned = np.full(n_gps, -1, dtype=np.int32)
    aligned[-1] = position

    for i in range(n_gps - 1, 0, -1):

        previous_position = back[i, aligned[i]]

        if previous_position < 0:
            raise RuntimeError("Alignment backtracking failed")

        aligned[i - 1] = previous_position

    matched_distances = distances[
        np.arange(n_gps),
        aligned
    ]

    return aligned, matched_distances


# ============================================================
# PROCESS ONE TRIP
# ============================================================

def process_trip(trip_dir):

    trip_id = trip_dir.name

    print(f"\nProcessing trip {trip_id}...")

    candidate_file = (
        CANDIDATE_DIR / f"{trip_id}_candidates.csv"
    )

    if not candidate_file.exists():
        raise FileNotFoundError(candidate_file)

    track = load_raw(trip_dir / f"{trip_id}.track")
    nodes = load_raw(trip_dir / f"{trip_id}.nodes")
    arcs = load_raw(trip_dir / f"{trip_id}.arcs")
    route = load_raw(trip_dir / f"{trip_id}.route")

    candidates = pd.read_csv(candidate_file)

    gps_lon = track.iloc[:, 0].to_numpy(dtype=float)
    gps_lat = track.iloc[:, 1].to_numpy(dtype=float)
    timestamps = track.iloc[:, 2].to_numpy(dtype=float)

    node_lon = nodes.iloc[:, 0].to_numpy(dtype=float)
    node_lat = nodes.iloc[:, 1].to_numpy(dtype=float)

    arc_data = arcs.iloc[:, :2].to_numpy(dtype=int)
    route_arc_ids = route.iloc[:, 0].to_numpy(dtype=int)

    if len(gps_lon) == 0 or len(route_arc_ids) == 0:
        raise ValueError("Empty GPS track or reference route")

    if (
        np.any(route_arc_ids < 0)
        or np.any(route_arc_ids >= len(arc_data))
    ):
        raise ValueError("Invalid reference-route arc ID")

    route_arcs = arc_data[route_arc_ids]

    if (
        np.any(route_arcs < 0)
        or np.any(route_arcs >= len(nodes))
    ):
        raise ValueError("Invalid node index")

    # Project coordinates.
    center_lat = float(np.mean(gps_lat))

    gps_xy = project_coordinates(
        gps_lon, gps_lat, center_lat
    )

    starts = project_coordinates(
        node_lon[route_arcs[:, 0]],
        node_lat[route_arcs[:, 0]],
        center_lat
    )

    ends = project_coordinates(
        node_lon[route_arcs[:, 1]],
        node_lat[route_arcs[:, 1]],
        center_lat
    )

    # Calculate alignment.
    distances = calculate_route_distances(
        gps_xy, starts, ends
    )

    positions, alignment_distances = align_gps_to_route(
        distances
    )

    aligned_arc_ids = route_arc_ids[positions]

    diagnostic = pd.DataFrame({
        "trip_id": trip_id,
        "gps_index": np.arange(len(gps_xy)),
        "aligned_route_position": positions,
        "aligned_arc_id": aligned_arc_ids,
        "alignment_distance_m": alignment_distances
    })

    diagnostic["alignment_accepted"] = (
        diagnostic["alignment_distance_m"]
        <= MAX_ALIGNMENT_DISTANCE_M
    )

    # Candidate labels.
    candidates["trip_id"] = trip_id
    candidates["gps_index"] = candidates["gps_index"].astype(int)
    candidates["arc_id"] = candidates["arc_id"].astype(int)

    lookup = diagnostic.set_index("gps_index")

    for column in [
        "aligned_route_position",
        "aligned_arc_id",
        "alignment_distance_m",
        "alignment_accepted"
    ]:
        candidates[column] = candidates["gps_index"].map(
            lookup[column]
        )

    candidates["label"] = (
        candidates["arc_id"] == candidates["aligned_arc_id"]
    ).astype(int)

    positive_by_gps = candidates.groupby("gps_index")["label"].max()

    valid_gps = positive_by_gps[
        positive_by_gps == 1
    ].index

    candidates = candidates[
        candidates["gps_index"].isin(valid_gps)
        & candidates["alignment_accepted"].astype(bool)
    ].copy()

    # GPS speed and heading.
    speed = np.zeros(len(gps_xy))
    heading = np.zeros(len(gps_xy))

    if len(gps_xy) > 1:

        displacement = haversine_m(
            gps_lat[:-1],
            gps_lon[:-1],
            gps_lat[1:],
            gps_lon[1:]
        )

        time_delta = np.diff(timestamps)
        valid_time = time_delta > 0

        speed[1:][valid_time] = (
            displacement[valid_time] / time_delta[valid_time]
        )

        lat1 = np.radians(gps_lat[:-1])
        lat2 = np.radians(gps_lat[1:])
        dlon = np.radians(gps_lon[1:] - gps_lon[:-1])

        y = np.sin(dlon) * np.cos(lat2)

        x = (
            np.cos(lat1) * np.sin(lat2)
            - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        )

        heading[1:] = (
            np.degrees(np.arctan2(y, x)) + 360
        ) % 360

    candidates["gps_speed_mps"] = candidates["gps_index"].map(
        dict(enumerate(speed))
    )

    candidates["gps_heading_deg"] = candidates["gps_index"].map(
        dict(enumerate(heading))
    )

    # Candidate road length and bearing.
    candidate_arcs = arc_data[
        candidates["arc_id"].to_numpy()
    ]

    start_lon = node_lon[candidate_arcs[:, 0]]
    start_lat = node_lat[candidate_arcs[:, 0]]

    end_lon = node_lon[candidate_arcs[:, 1]]
    end_lat = node_lat[candidate_arcs[:, 1]]

    candidates["road_length_m"] = haversine_m(
        start_lat, start_lon, end_lat, end_lon
    )

    lat1 = np.radians(start_lat)
    lat2 = np.radians(end_lat)
    dlon = np.radians(end_lon - start_lon)

    y = np.sin(dlon) * np.cos(lat2)

    x = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    road_bearing = (
        np.degrees(np.arctan2(y, x)) + 360
    ) % 360

    candidates["road_bearing_deg"] = road_bearing

    angle = np.abs(
        candidates["gps_heading_deg"].to_numpy()
        - road_bearing
    )

    candidates["heading_difference_deg"] = np.minimum(
        angle, 360 - angle
    )

    summary = {
        "trip_id": trip_id,
        "gps_observations": len(gps_xy),
        "route_arcs": len(route_arc_ids),
        "accepted_alignments": int(
            diagnostic["alignment_accepted"].sum()
        ),
        "gps_with_positive_candidate": len(valid_gps),
        "candidate_rows": len(candidates),
        "positive_labels": int(candidates["label"].sum()),
        "negative_labels": int((candidates["label"] == 0).sum()),
        "mean_alignment_distance_m": float(
            diagnostic["alignment_distance_m"].mean()
        ),
        "max_alignment_distance_m": float(
            diagnostic["alignment_distance_m"].max()
        )
    }

    return candidates, diagnostic, summary


# ============================================================
# MAIN - ALL TRAINING TRIPS
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    trip_dirs = sorted(
        [p for p in TRAIN_DIR.iterdir() if p.is_dir()]
    )

    print("=" * 65)
    print("PATHSNAP - BATCH TRAINING LABEL CONSTRUCTION")
    print("=" * 65)
    print("Training trips found:", len(trip_dirs))

    all_candidates = []
    all_diagnostics = []
    summaries = []
    failures = []

    for trip_dir in trip_dirs:

        try:
            candidates, diagnostic, summary = process_trip(trip_dir)

            all_candidates.append(candidates)
            all_diagnostics.append(diagnostic)
            summaries.append(summary)

            print(
                f"Success: {trip_dir.name} | "
                f"Rows: {len(candidates)} | "
                f"Positive: {summary['positive_labels']}"
            )

        except Exception as error:

            failures.append({
                "trip_id": trip_dir.name,
                "error": str(error)
            })

            print(f"FAILED: {trip_dir.name} | {error}")

    # Save combined datasets.
    if all_candidates:

        combined = pd.concat(
            all_candidates,
            ignore_index=True
        )

        combined.to_csv(
            OUTPUT_DIR / "all_train_labeled_candidates.csv",
            index=False
        )

    if all_diagnostics:

        diagnostics = pd.concat(
            all_diagnostics,
            ignore_index=True
        )

        diagnostics.to_csv(
            REPORT_DIR / "all_train_alignment_validation.csv",
            index=False
        )

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        REPORT_DIR / "training_label_summary.csv",
        index=False
    )

    pd.DataFrame(failures).to_csv(
        REPORT_DIR / "training_label_failures.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 65)

    print("Total trips:", len(trip_dirs))
    print("Successful trips:", len(summaries))
    print("Failed trips:", len(failures))

    if all_candidates:

        print("Total candidate rows:", len(combined))
        print("Positive labels:", int(combined["label"].sum()))
        print("Negative labels:", int((combined["label"] == 0).sum()))

    print("\nCombined dataset:")
    print(OUTPUT_DIR / "all_train_labeled_candidates.csv")

    print("\nTrip summary:")
    print(REPORT_DIR / "training_label_summary.csv")

    print("\nFailure report:")
    print(REPORT_DIR / "training_label_failures.csv")


if __name__ == "__main__":
    main()
