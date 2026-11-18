
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRIP_DIR = ROOT / "data" / "processed" / "trips" / "student_test"
CANDIDATE_DIR = ROOT / "data" / "processed" / "candidates" / "student_test"

OUTPUT_DIR = ROOT / "data" / "processed" / "training" / "student_test"
REPORT_DIR = ROOT / "reports" / "student_test"

MAX_ALIGNMENT_DISTANCE_M = 100.0
MAX_ROUTE_ADVANCE_PER_GPS = 10

EARTH_RADIUS = 6371000.0


# ============================================================
# GEOMETRY FUNCTIONS
# ============================================================

def haversine_m(lat1, lon1, lat2, lon2):

    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))

    dlat = lat2 - lat1
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )


def project_coordinates(lon, lat, center_lat):

    center = np.radians(center_lat)

    x = (
        EARTH_RADIUS
        * np.radians(np.asarray(lon, dtype=float))
        * np.cos(center)
    )

    y = EARTH_RADIUS * np.radians(
        np.asarray(lat, dtype=float)
    )

    return np.column_stack((x, y))


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
# CHRONOLOGICAL GPS-TO-ROUTE ALIGNMENT
# Same alignment method as training script
# ============================================================

def align_gps_to_route(distances):

    n_gps, n_route = distances.shape

    if n_gps == 0 or n_route == 0:
        raise ValueError("Empty GPS track or reference route")

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
        best_previous = np.full(
            n_route, -1, dtype=np.int32
        )

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
# GPS SPEED AND HEADING
# Same calculations as training script
# ============================================================

def calculate_gps_features(track):

    gps_lon = track["longitude"].to_numpy(dtype=float)
    gps_lat = track["latitude"].to_numpy(dtype=float)
    timestamps = track["timestamp"].to_numpy(dtype=float)

    n = len(track)

    speed = np.zeros(n)
    heading = np.zeros(n)

    if n > 1:

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

    return speed, heading


# ============================================================
# PROCESS ONE STUDENT-TEST TRIP
# ============================================================

def process_trip(trip_dir):

    trip_id = trip_dir.name

    print(f"\nProcessing trip {trip_id}...")

    candidate_file = (
        CANDIDATE_DIR / f"{trip_id}_candidates.csv"
    )

    if not candidate_file.exists():
        raise FileNotFoundError(candidate_file)

    # Load processed data
    track = pd.read_csv(trip_dir / "track.csv")
    nodes = pd.read_csv(trip_dir / "nodes.csv")
    arcs = pd.read_csv(trip_dir / "arcs.csv")
    route = pd.read_csv(trip_dir / "route.csv")

    candidates = pd.read_csv(candidate_file)

    # Sort data chronologically
    track = track.sort_values("gps_index").reset_index(drop=True)

    route = route.sort_values(
        "route_position"
    ).reset_index(drop=True)

    # Build ID-to-coordinate mappings
    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    arcs = arcs.sort_values("arc_id").reset_index(drop=True)

    node_ids = nodes["node_id"].to_numpy(dtype=int)
    node_lookup = {
        node_id: i for i, node_id in enumerate(node_ids)
    }

    arc_ids = arcs["arc_id"].to_numpy(dtype=int)
    arc_lookup = {
        arc_id: i for i, arc_id in enumerate(arc_ids)
    }

    # Validate IDs
    route_arc_ids = route["arc_id"].to_numpy(dtype=int)

    if not set(route_arc_ids).issubset(arc_lookup):
        raise ValueError("Reference route contains unknown arc IDs")

    if not set(candidates["arc_id"].astype(int)).issubset(arc_lookup):
        raise ValueError("Candidates contain unknown arc IDs")

    # GPS and road coordinates
    gps_lon = track["longitude"].to_numpy(dtype=float)
    gps_lat = track["latitude"].to_numpy(dtype=float)

    center_lat = float(np.mean(gps_lat))

    gps_xy = project_coordinates(
        gps_lon, gps_lat, center_lat
    )

    node_lon = nodes["longitude"].to_numpy(dtype=float)
    node_lat = nodes["latitude"].to_numpy(dtype=float)

    arc_from = arcs["from_node_id"].to_numpy(dtype=int)
    arc_to = arcs["to_node_id"].to_numpy(dtype=int)

    route_indices = np.array(
        [arc_lookup[a] for a in route_arc_ids],
        dtype=int
    )

    route_from = arc_from[route_indices]
    route_to = arc_to[route_indices]

    route_start = np.array(
        [node_lookup[n] for n in route_from],
        dtype=int
    )

    route_end = np.array(
        [node_lookup[n] for n in route_to],
        dtype=int
    )

    starts = project_coordinates(
        node_lon[route_start],
        node_lat[route_start],
        center_lat
    )

    ends = project_coordinates(
        node_lon[route_end],
        node_lat[route_end],
        center_lat
    )

    # Align GPS observations with reference route
    distances = calculate_route_distances(
        gps_xy, starts, ends
    )

    positions, alignment_distances = align_gps_to_route(
        distances
    )

    aligned_arc_ids = route_arc_ids[positions]

    diagnostic = pd.DataFrame({
        "trip_id": trip_id,
        "gps_index": track["gps_index"].to_numpy(),
        "aligned_route_position": positions,
        "aligned_arc_id": aligned_arc_ids,
        "alignment_distance_m": alignment_distances
    })

    diagnostic["alignment_accepted"] = (
        diagnostic["alignment_distance_m"]
        <= MAX_ALIGNMENT_DISTANCE_M
    )

    # Add alignment information to candidates
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

    if candidates["aligned_arc_id"].isna().any():
        raise ValueError("Some candidates have invalid GPS indices")

    # Provisional labels
    candidates["label"] = (
        candidates["arc_id"]
        == candidates["aligned_arc_id"]
    ).astype(int)

    positive_by_gps = candidates.groupby(
        "gps_index"
    )["label"].max()

    valid_gps = positive_by_gps[
        positive_by_gps == 1
    ].index

    candidates = candidates[
        candidates["gps_index"].isin(valid_gps)
        & candidates["alignment_accepted"].astype(bool)
    ].copy()

    # Calculate GPS features
    speed, heading = calculate_gps_features(track)

    speed_lookup = dict(
        zip(track["gps_index"].astype(int), speed)
    )

    heading_lookup = dict(
        zip(track["gps_index"].astype(int), heading)
    )

    candidates["gps_speed_mps"] = (
        candidates["gps_index"].map(speed_lookup)
    )

    candidates["gps_heading_deg"] = (
        candidates["gps_index"].map(heading_lookup)
    )

    # Calculate road features using arc IDs
    candidate_arc_indices = np.array([
        arc_lookup[a] for a in candidates["arc_id"]
    ])

    candidate_from = arc_from[candidate_arc_indices]
    candidate_to = arc_to[candidate_arc_indices]

    start_indices = np.array([
        node_lookup[n] for n in candidate_from
    ])

    end_indices = np.array([
        node_lookup[n] for n in candidate_to
    ])

    start_lon = node_lon[start_indices]
    start_lat = node_lat[start_indices]

    end_lon = node_lon[end_indices]
    end_lat = node_lat[end_indices]

    candidates["road_length_m"] = haversine_m(
        start_lat,
        start_lon,
        end_lat,
        end_lon
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

    # Final feature validation
    features = [
        "distance_to_arc_m",
        "projection_fraction",
        "candidate_rank",
        "gps_speed_mps",
        "gps_heading_deg",
        "road_length_m",
        "road_bearing_deg",
        "heading_difference_deg"
    ]

    if candidates[features].isna().any().any():
        raise ValueError("Missing values found in model features")

    # Summary
    summary = {
        "trip_id": trip_id,
        "gps_observations": len(track),
        "reference_route_arcs": len(route),
        "accepted_alignments": int(
            diagnostic["alignment_accepted"].sum()
        ),
        "gps_with_positive_candidate": len(valid_gps),
        "candidate_rows": len(candidates),
        "positive_labels": int(candidates["label"].sum()),
        "negative_labels": int(
            (candidates["label"] == 0).sum()
        ),
        "mean_alignment_distance_m": float(
            diagnostic["alignment_distance_m"].mean()
        ),
        "max_alignment_distance_m": float(
            diagnostic["alignment_distance_m"].max()
        )
    }

    return candidates, diagnostic, summary


# ============================================================
# MAIN - PROCESS ALL 10 STUDENT-TEST TRIPS
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    trip_dirs = sorted(
        [p for p in TRIP_DIR.iterdir() if p.is_dir()]
    )

    print("=" * 65)
    print("PATHSNAP - STUDENT-TEST DATA PREPARATION")
    print("=" * 65)

    print("Student-test trips found:", len(trip_dirs))

    all_candidates = []
    all_diagnostics = []
    summaries = []
    failures = []

    for trip_dir in trip_dirs:

        try:

            candidates, diagnostic, summary = process_trip(
                trip_dir
            )

            all_candidates.append(candidates)
            all_diagnostics.append(diagnostic)
            summaries.append(summary)

            print(
                f"SUCCESS: {trip_dir.name} | "
                f"Rows: {len(candidates)} | "
                f"Positive: {summary['positive_labels']}"
            )

        except Exception as error:

            failures.append({
                "trip_id": trip_dir.name,
                "error": str(error)
            })

            print(
                f"FAILED: {trip_dir.name} | {error}"
            )

    # Save successful trips
    if all_candidates:

        combined = pd.concat(
            all_candidates,
            ignore_index=True
        )

        combined.to_csv(
            OUTPUT_DIR / "all_student_test_labeled_candidates.csv",
            index=False
        )

        print("\nCombined candidate rows:", len(combined))
        print("Positive labels:", int(combined["label"].sum()))
        print("Negative labels:", int((combined["label"] == 0).sum()))

    if all_diagnostics:

        pd.concat(
            all_diagnostics,
            ignore_index=True
        ).to_csv(
            REPORT_DIR / "student_test_alignment_validation.csv",
            index=False
        )

    pd.DataFrame(summaries).to_csv(
        REPORT_DIR / "student_test_label_summary.csv",
        index=False
    )

    pd.DataFrame(failures).to_csv(
        REPORT_DIR / "student_test_label_failures.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("STUDENT-TEST PREPARATION SUMMARY")
    print("=" * 65)

    print("Total trips:", len(trip_dirs))
    print("Successful trips:", len(summaries))
    print("Failed trips:", len(failures))

    print("\nOutput dataset:")
    print(
        OUTPUT_DIR / "all_student_test_labeled_candidates.csv"
    )

    print("\nReports:")
    print(REPORT_DIR)

    print("\nSTUDENT-TEST PREPARATION FINISHED")


if __name__ == "__main__":
    main()
