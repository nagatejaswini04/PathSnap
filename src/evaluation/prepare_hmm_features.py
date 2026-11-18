
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRIP_DIR = ROOT / "data" / "processed" / "trips" / "student_test"
CANDIDATE_DIR = ROOT / "data" / "processed" / "candidates" / "student_test"

OUTPUT_DIR = ROOT / "data" / "processed" / "hmm" / "student_test"

EARTH_RADIUS = 6371000.0

FEATURES = [
    "distance_to_arc_m",
    "projection_fraction",
    "candidate_rank",
    "gps_speed_mps",
    "gps_heading_deg",
    "road_length_m",
    "road_bearing_deg",
    "heading_difference_deg",
]


# ============================================================
# GEOMETRY
# ============================================================

def haversine_m(lat1, lon1, lat2, lon2):

    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))

    dlat = lat2 - lat1
    dlon = np.radians(np.asarray(lon2, dtype=float) -
                       np.asarray(lon1, dtype=float))

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )


# ============================================================
# GPS FEATURES
# ============================================================

def calculate_gps_features(track):

    lon = track["longitude"].to_numpy(dtype=float)
    lat = track["latitude"].to_numpy(dtype=float)
    timestamps = track["timestamp"].to_numpy(dtype=float)

    n = len(track)

    speed = np.zeros(n)
    heading = np.zeros(n)

    if n > 1:

        displacement = haversine_m(
            lat[:-1], lon[:-1],
            lat[1:], lon[1:]
        )

        time_delta = np.diff(timestamps)

        valid = time_delta > 0

        speed[1:] = np.divide(
            displacement,
            time_delta,
            out=np.zeros(n - 1),
            where=valid
        )

        lat1 = np.radians(lat[:-1])
        lat2 = np.radians(lat[1:])
        dlon = np.radians(lon[1:] - lon[:-1])

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
# PROCESS ONE TRIP
# ============================================================

def process_trip(trip_dir):

    trip_id = trip_dir.name

    print(f"\nProcessing trip: {trip_id}")

    candidate_file = (
        CANDIDATE_DIR / f"{trip_id}_candidates.csv"
    )

    if not candidate_file.exists():
        raise FileNotFoundError(candidate_file)

    track = pd.read_csv(trip_dir / "track.csv")
    nodes = pd.read_csv(trip_dir / "nodes.csv")
    arcs = pd.read_csv(trip_dir / "arcs.csv")
    candidates = pd.read_csv(candidate_file)

    # Sort chronologically
    track = track.sort_values("gps_index").reset_index(drop=True)

    candidates["gps_index"] = candidates["gps_index"].astype(int)
    candidates["arc_id"] = candidates["arc_id"].astype(int)

    # Validate candidate GPS indices
    valid_gps_indices = set(track["gps_index"].astype(int))

    if not set(candidates["gps_index"]).issubset(valid_gps_indices):
        raise ValueError("Candidate file contains unknown GPS indices")

    # Calculate GPS features
    speed, heading = calculate_gps_features(track)

    gps_features = pd.DataFrame({
        "gps_index": track["gps_index"].astype(int),
        "gps_speed_mps": speed,
        "gps_heading_deg": heading,
    })

    candidates = candidates.merge(
        gps_features,
        on="gps_index",
        how="left",
        validate="many_to_one"
    )

    # Build node coordinate lookup
    nodes = nodes.drop_duplicates("node_id")

    node_lookup = nodes.set_index("node_id")[
        ["longitude", "latitude"]
    ]

    # Use road endpoints from the road network
    arc_endpoints = arcs[
        ["arc_id", "from_node_id", "to_node_id"]
    ].drop_duplicates("arc_id")

    candidates = candidates.drop(
        columns=["from_node_id", "to_node_id"],
        errors="ignore"
    )

    candidates = candidates.merge(
        arc_endpoints,
        on="arc_id",
        how="left",
        validate="many_to_one"
    )

    if candidates[
        ["from_node_id", "to_node_id"]
    ].isna().any().any():
        raise ValueError("Some candidate arcs have missing endpoints")

    # Retrieve endpoint coordinates
    start = candidates["from_node_id"].map(
        node_lookup["longitude"]
    )

    start_lat = candidates["from_node_id"].map(
        node_lookup["latitude"]
    )

    end = candidates["to_node_id"].map(
        node_lookup["longitude"]
    )

    end_lat = candidates["to_node_id"].map(
        node_lookup["latitude"]
    )

    if pd.concat(
        [start, start_lat, end, end_lat], axis=1
    ).isna().any().any():
        raise ValueError("Some road endpoints have missing coordinates")

    # Road length
    candidates["road_length_m"] = haversine_m(
        start_lat, start,
        end_lat, end
    )

    # Road bearing
    lat1 = np.radians(start_lat.to_numpy(dtype=float))
    lat2 = np.radians(end_lat.to_numpy(dtype=float))
    dlon = np.radians(
        end.to_numpy(dtype=float) - start.to_numpy(dtype=float)
    )

    y = np.sin(dlon) * np.cos(lat2)

    x = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    candidates["road_bearing_deg"] = (
        np.degrees(np.arctan2(y, x)) + 360
    ) % 360

    # Heading difference
    angle = np.abs(
        candidates["gps_heading_deg"].to_numpy()
        - candidates["road_bearing_deg"].to_numpy()
    )

    candidates["heading_difference_deg"] = np.minimum(
        angle, 360 - angle
    )

    # Validate model features
    if candidates[FEATURES].isna().any().any():
        raise ValueError("Missing values in model features")

    candidates = candidates.sort_values(
        ["gps_index", "candidate_rank"]
    ).reset_index(drop=True)

    print("GPS observations:", track["gps_index"].nunique())
    print("Candidate rows:", len(candidates))
    print("Unique candidate arcs:", candidates["arc_id"].nunique())

    return candidates


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trip_dirs = sorted(
        [p for p in TRIP_DIR.iterdir() if p.is_dir()]
    )

    all_candidates = []
    failures = []

    print("=" * 65)
    print("PATHSNAP - HMM FEATURE PREPARATION")
    print("=" * 65)

    for trip_dir in trip_dirs:

        try:

            candidates = process_trip(trip_dir)

            all_candidates.append(candidates)

            candidates.to_csv(
                OUTPUT_DIR / f"{trip_dir.name}_hmm_features.csv",
                index=False
            )

            print("SUCCESS:", trip_dir.name)

        except Exception as error:

            failures.append({
                "trip_id": trip_dir.name,
                "error": str(error)
            })

            print("FAILED:", trip_dir.name, "|", error)

    if all_candidates:

        combined = pd.concat(
            all_candidates,
            ignore_index=True
        )

        combined.to_csv(
            OUTPUT_DIR / "all_student_test_hmm_features.csv",
            index=False
        )

        print("\nTotal candidate rows:", len(combined))
        print("Total trips:", combined["trip_id"].nunique())
        print("Missing feature values:",
              int(combined[FEATURES].isna().sum().sum()))

    pd.DataFrame(failures).to_csv(
        OUTPUT_DIR / "hmm_feature_failures.csv",
        index=False
    )

    print("\nSuccessful trips:", len(all_candidates))
    print("Failed trips:", len(failures))
    print("Output directory:", OUTPUT_DIR)
    print("\nHMM FEATURE PREPARATION COMPLETED")


if __name__ == "__main__":
    main()
