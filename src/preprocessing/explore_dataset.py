
"""
PathSnap - Dataset Exploration
--------------------------------
Inspects:
1. Training and student-testing trips
2. File availability and file sizes
3. GPS observations
4. Road nodes and arcs
5. Reference routes
6. Missing and invalid values
7. Timestamp ordering and time gaps
8. Geographic boundaries
9. Road-network connectivity indicators
10. Route arc-ID validity

This script does NOT modify the original dataset.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

TRAIN_DIR = DATA_DIR / "train"
STUDENT_TEST_DIR = DATA_DIR / "student_test"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "dataset_exploration"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FILE CONFIGURATION
# ============================================================

FILE_EXTENSIONS = {
    "track": ".track",
    "nodes": ".nodes",
    "arcs": ".arcs",
    "route": ".route"
}


# ============================================================
# 3. DATA LOADING FUNCTION
# ============================================================

def load_file(file_path, expected_columns):

    if not file_path.exists():
        return None

    try:
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            engine="python"
        )

        if df.shape[1] != expected_columns:
            print(
                f"WARNING: {file_path.name} has "
                f"{df.shape[1]} columns; expected "
                f"{expected_columns}"
            )

        return df

    except Exception as error:

        print(f"ERROR reading {file_path}: {error}")

        return None


# ============================================================
# 4. INSPECT ONE TRIP
# ============================================================

def inspect_trip(trip_dir, partition):

    trip_id = trip_dir.name

    print("\n" + "=" * 65)
    print(f"PARTITION: {partition}")
    print(f"TRIP ID: {trip_id}")
    print("=" * 65)

    result = {
        "partition": partition,
        "trip_id": trip_id,
        "files_available": True,
        "track_rows": 0,
        "nodes_rows": 0,
        "arcs_rows": 0,
        "route_rows": 0,
        "track_missing_values": 0,
        "nodes_missing_values": 0,
        "arcs_missing_values": 0,
        "route_missing_values": 0,
        "track_invalid_values": 0,
        "nodes_invalid_values": 0,
        "arcs_invalid_values": 0,
        "route_invalid_values": 0,
        "timestamp_non_increasing": 0,
        "timestamp_gap_count": 0,
        "max_timestamp_gap_seconds": None,
        "latitude_min": None,
        "latitude_max": None,
        "longitude_min": None,
        "longitude_max": None,
        "isolated_nodes": None,
        "invalid_arc_endpoints": None,
        "invalid_route_arc_ids": None,
        "track_file_bytes": 0,
        "nodes_file_bytes": 0,
        "arcs_file_bytes": 0,
        "route_file_bytes": 0
    }

    # --------------------------------------------------------
    # Locate files
    # --------------------------------------------------------

    file_paths = {
        name: trip_dir / f"{trip_id}{extension}"
        for name, extension in FILE_EXTENSIONS.items()
    }

    expected_columns = {
        "track": 3,
        "nodes": 2,
        "arcs": 2,
        "route": 1
    }

    data = {}

    for name, file_path in file_paths.items():

        if not file_path.exists():

            print(f"MISSING FILE: {file_path.name}")

            result["files_available"] = False
            data[name] = None

            continue

        result[f"{name}_file_bytes"] = file_path.stat().st_size

        data[name] = load_file(
            file_path,
            expected_columns[name]
        )

    track = data["track"]
    nodes = data["nodes"]
    arcs = data["arcs"]
    route = data["route"]

    # --------------------------------------------------------
    # TRACK INSPECTION
    # --------------------------------------------------------

    if track is not None and track.shape[1] == 3:

        track.columns = [
            "longitude",
            "latitude",
            "timestamp"
        ]

        result["track_rows"] = len(track)

        result["track_missing_values"] = int(
            track.isnull().sum().sum()
        )

        track = track.apply(
            pd.to_numeric,
            errors="coerce"
        )

        result["track_invalid_values"] = int(
            track.isnull().sum().sum()
        )

        valid_track = track.dropna()

        if len(valid_track) > 0:

            longitude = valid_track["longitude"]
            latitude = valid_track["latitude"]
            timestamp = valid_track["timestamp"]

            result["longitude_min"] = float(longitude.min())
            result["longitude_max"] = float(longitude.max())

            result["latitude_min"] = float(latitude.min())
            result["latitude_max"] = float(latitude.max())

            # Basic geographical plausibility checks
            invalid_coordinates = (
                (longitude < -180) |
                (longitude > 180) |
                (latitude < -90) |
                (latitude > 90)
            )

            print(
                "Invalid GPS coordinate rows:",
                int(invalid_coordinates.sum())
            )

            # Timestamp ordering
            time_differences = timestamp.diff().dropna()

            result["timestamp_non_increasing"] = int(
                (time_differences <= 0).sum()
            )

            positive_gaps = time_differences[
                time_differences > 0
            ]

            result["timestamp_gap_count"] = int(
                (positive_gaps > 1).sum()
            )

            if len(positive_gaps) > 0:

                result["max_timestamp_gap_seconds"] = float(
                    positive_gaps.max()
                )

        print("\nGPS TRACK")
        print(track.head())
        print("Shape:", track.shape)

    # --------------------------------------------------------
    # NODES INSPECTION
    # --------------------------------------------------------

    if nodes is not None and nodes.shape[1] == 2:

        nodes.columns = [
            "longitude",
            "latitude"
        ]

        result["nodes_rows"] = len(nodes)

        result["nodes_missing_values"] = int(
            nodes.isnull().sum().sum()
        )

        nodes = nodes.apply(
            pd.to_numeric,
            errors="coerce"
        )

        result["nodes_invalid_values"] = int(
            nodes.isnull().sum().sum()
        )

        valid_nodes = nodes.dropna()

        if len(valid_nodes) > 0:

            invalid_node_coordinates = (
                (valid_nodes["longitude"] < -180) |
                (valid_nodes["longitude"] > 180) |
                (valid_nodes["latitude"] < -90) |
                (valid_nodes["latitude"] > 90)
            )

            print(
                "Invalid node coordinate rows:",
                int(invalid_node_coordinates.sum())
            )

        print("\nROAD NODES")
        print(nodes.head())
        print("Shape:", nodes.shape)

    # --------------------------------------------------------
    # ARCS INSPECTION
    # --------------------------------------------------------

    if arcs is not None and arcs.shape[1] == 2:

        arcs.columns = [
            "from_node_id",
            "to_node_id"
        ]

        result["arcs_rows"] = len(arcs)

        result["arcs_missing_values"] = int(
            arcs.isnull().sum().sum()
        )

        arcs = arcs.apply(
            pd.to_numeric,
            errors="coerce"
        )

        result["arcs_invalid_values"] = int(
            arcs.isnull().sum().sum()
        )

        print("\nROAD ARCS")
        print(arcs.head())
        print("Shape:", arcs.shape)

        # Check arc endpoints against node IDs
        if nodes is not None:

            valid_arcs = arcs.dropna()

            integer_arcs = valid_arcs[
                (valid_arcs % 1 == 0).all(axis=1)
            ]

            node_count = len(nodes)

            invalid_endpoints = (
                (integer_arcs["from_node_id"] < 0) |
                (integer_arcs["from_node_id"] >= node_count) |
                (integer_arcs["to_node_id"] < 0) |
                (integer_arcs["to_node_id"] >= node_count)
            )

            result["invalid_arc_endpoints"] = int(
                invalid_endpoints.sum()
            )

            # Count isolated nodes using valid arc endpoints
            if node_count > 0:

                connected_nodes = set(
                    integer_arcs.loc[
                        ~invalid_endpoints,
                        "from_node_id"
                    ].astype(int)
                )

                connected_nodes.update(
                    integer_arcs.loc[
                        ~invalid_endpoints,
                        "to_node_id"
                    ].astype(int)
                )

                result["isolated_nodes"] = int(
                    node_count - len(connected_nodes)
                )

    # --------------------------------------------------------
    # ROUTE INSPECTION
    # --------------------------------------------------------

    if route is not None and route.shape[1] == 1:

        route.columns = ["arc_id"]

        result["route_rows"] = len(route)

        result["route_missing_values"] = int(
            route.isnull().sum().sum()
        )

        route = route.apply(
            pd.to_numeric,
            errors="coerce"
        )

        result["route_invalid_values"] = int(
            route.isnull().sum().sum()
        )

        print("\nREFERENCE ROUTE")
        print(route.head())
        print("Shape:", route.shape)

        # Validate route arc IDs
        if arcs is not None:

            valid_route = route["arc_id"].dropna()

            integer_route = valid_route[
                valid_route % 1 == 0
            ].astype(int)

            invalid_route_ids = (
                (integer_route < 0) |
                (integer_route >= len(arcs))
            )

            result["invalid_route_arc_ids"] = int(
                invalid_route_ids.sum()
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\nTRIP SUMMARY")

    for key, value in result.items():

        print(f"{key}: {value}")

    return result


# ============================================================
# 5. INSPECT A DATASET PARTITION
# ============================================================

def inspect_partition(partition_dir, partition_name):

    print("\n\n")
    print("#" * 75)
    print(f"INSPECTING {partition_name.upper()}")
    print("#" * 75)

    if not partition_dir.exists():

        print(f"Partition not found: {partition_dir}")
        return []

    trip_dirs = sorted(
        path for path in partition_dir.iterdir()
        if path.is_dir()
    )

    print("Number of trips:", len(trip_dirs))

    results = []

    for trip_dir in trip_dirs:

        result = inspect_trip(
            trip_dir,
            partition_name
        )

        results.append(result)

    return results


# ============================================================
# 6. MAIN FUNCTION
# ============================================================

def main():

    print("\nPATHSNAP DATASET EXPLORATION")
    print("=" * 75)

    print("Project directory:", PROJECT_ROOT)
    print("Data directory:", DATA_DIR)

    # Inspect both partitions
    training_results = inspect_partition(
        TRAIN_DIR,
        "train"
    )

    student_results = inspect_partition(
        STUDENT_TEST_DIR,
        "student_test"
    )

    all_results = training_results + student_results

    if not all_results:

        print("No trips found. Check your data directory.")
        return

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    results_df = pd.DataFrame(all_results)

    output_file = OUTPUT_DIR / "trip_exploration_summary.csv"

    results_df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Save partition summary
    # --------------------------------------------------------

    summary = (
        results_df.groupby("partition")
        .agg(
            number_of_trips=("trip_id", "count"),
            total_gps_observations=("track_rows", "sum"),
            total_nodes=("nodes_rows", "sum"),
            total_arcs=("arcs_rows", "sum"),
            total_route_arcs=("route_rows", "sum"),
            trips_with_missing_files=(
                "files_available",
                lambda x: int((~x).sum())
            )
        )
        .reset_index()
    )

    summary_file = OUTPUT_DIR / "partition_summary.csv"

    summary.to_csv(
        summary_file,
        index=False
    )

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print("\n\n")
    print("=" * 75)
    print("FINAL DATASET SUMMARY")
    print("=" * 75)

    print(summary.to_string(index=False))

    print("\nTotal trips inspected:", len(results_df))

    print("\nReports saved:")
    print(output_file)
    print(summary_file)

    print("\nDATASET EXPLORATION COMPLETED")


if __name__ == "__main__":
    main()
