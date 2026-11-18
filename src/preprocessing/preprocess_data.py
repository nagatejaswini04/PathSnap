
"""
PathSnap - Data Preprocessing Pipeline

Processes:
    data/train/
    data/student_test/

For each trip:
    1. Load GPS observations (.track)
    2. Load road nodes (.nodes)
    3. Load road arcs (.arcs)
    4. Load reference route (.route), when available
    5. Validate values and coordinates
    6. Calculate GPS movement features
    7. Save processed CSV files

Original dataset files are never modified.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "processed" / "trips"

PARTITIONS = ["train", "student_test"]


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def read_dataset_file(file_path):
    """Read a whitespace-separated dataset file."""

    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    return pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        engine="python"
    )


def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate distance between two GPS coordinates in metres."""

    earth_radius = 6371000

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius * np.arcsin(np.sqrt(a))


def validate_coordinates(longitude, latitude, description):
    """Check longitude and latitude ranges."""

    invalid = (
        (longitude < -180)
        | (longitude > 180)
        | (latitude < -90)
        | (latitude > 90)
    )

    if invalid.any():
        raise ValueError(
            f"{description}: found {invalid.sum()} invalid coordinates."
        )


def convert_timestamp(timestamp_series):
    """
    Convert timestamps to seconds.

    Numeric timestamps are retained as numeric values.
    Date/time strings are converted to Unix timestamps.
    """

    numeric = pd.to_numeric(timestamp_series, errors="coerce")

    if numeric.notna().all():
        return numeric.astype(float)

    parsed = pd.to_datetime(
        timestamp_series,
        errors="coerce",
        utc=True
    )

    if parsed.isna().any():
        raise ValueError("Some timestamps could not be interpreted.")

    return parsed.astype("int64") / 1_000_000_000


# --------------------------------------------------
# 3. PREPROCESS ONE TRIP
# --------------------------------------------------

def preprocess_trip(partition, trip_folder):

    trip_id = trip_folder.name

    print(f"\nProcessing {partition}/{trip_id}")

    # File paths
    track_file = trip_folder / f"{trip_id}.track"
    nodes_file = trip_folder / f"{trip_id}.nodes"
    arcs_file = trip_folder / f"{trip_id}.arcs"
    route_file = trip_folder / f"{trip_id}.route"

    # Load raw files
    track_raw = read_dataset_file(track_file)
    nodes_raw = read_dataset_file(nodes_file)
    arcs_raw = read_dataset_file(arcs_file)

    # Check expected columns
    if track_raw.shape[1] < 3:
        raise ValueError(f"{trip_id}: track must have at least 3 columns.")

    if nodes_raw.shape[1] < 2:
        raise ValueError(f"{trip_id}: nodes must have at least 2 columns.")

    if arcs_raw.shape[1] < 2:
        raise ValueError(f"{trip_id}: arcs must have at least 2 columns.")

    # --------------------------------------------------
    # A. GPS TRACK PREPROCESSING
    # --------------------------------------------------

    track = track_raw.iloc[:, :3].copy()
    track.columns = ["longitude", "latitude", "timestamp"]

    for column in ["longitude", "latitude"]:
        track[column] = pd.to_numeric(
            track[column], errors="coerce"
        )

    if track[["longitude", "latitude"]].isna().any().any():
        raise ValueError(f"{trip_id}: invalid GPS coordinate values.")

    validate_coordinates(
        track["longitude"],
        track["latitude"],
        f"{trip_id} GPS track"
    )

    track["timestamp"] = convert_timestamp(track["timestamp"])

    # Preserve original observation order.
    track.insert(0, "gps_index", np.arange(len(track)))
    track.insert(0, "trip_id", trip_id)

    # Time difference between consecutive GPS observations
    track["time_delta_sec"] = track["timestamp"].diff()

    if (track["time_delta_sec"].dropna() <= 0).any():
        raise ValueError(
            f"{trip_id}: timestamps are not strictly increasing."
        )

    # Distance between consecutive GPS observations
    track["displacement_m"] = haversine_distance(
        track["longitude"].shift(1),
        track["latitude"].shift(1),
        track["longitude"],
        track["latitude"]
    )

    # Approximate GPS-derived speed
    track["gps_speed_mps"] = (
        track["displacement_m"] / track["time_delta_sec"]
    )

    # First observation has no previous observation
    track.loc[track["gps_index"] == 0, [
        "time_delta_sec",
        "displacement_m",
        "gps_speed_mps"
    ]] = np.nan

    # --------------------------------------------------
    # B. ROAD NODES PREPROCESSING
    # --------------------------------------------------

    nodes = nodes_raw.iloc[:, :2].copy()
    nodes.columns = ["longitude", "latitude"]

    nodes["longitude"] = pd.to_numeric(
        nodes["longitude"], errors="coerce"
    )

    nodes["latitude"] = pd.to_numeric(
        nodes["latitude"], errors="coerce"
    )

    if nodes.isna().any().any():
        raise ValueError(f"{trip_id}: invalid road node values.")

    validate_coordinates(
        nodes["longitude"],
        nodes["latitude"],
        f"{trip_id} road nodes"
    )

    # Preserve row-index-based node IDs
    nodes.insert(0, "node_id", np.arange(len(nodes)))
    nodes.insert(0, "trip_id", trip_id)

    # --------------------------------------------------
    # C. ROAD ARCS PREPROCESSING
    # --------------------------------------------------

    arcs = arcs_raw.iloc[:, :2].copy()
    arcs.columns = ["from_node_id", "to_node_id"]

    for column in ["from_node_id", "to_node_id"]:
        arcs[column] = pd.to_numeric(
            arcs[column], errors="coerce"
        )

    if arcs.isna().any().any():
        raise ValueError(f"{trip_id}: invalid road arc endpoints.")

    if not np.equal(
        arcs[["from_node_id", "to_node_id"]].to_numpy(),
        arcs[["from_node_id", "to_node_id"]].to_numpy().astype(int)
    ).all():
        raise ValueError(f"{trip_id}: non-integer node IDs found.")

    arcs["from_node_id"] = arcs["from_node_id"].astype(int)
    arcs["to_node_id"] = arcs["to_node_id"].astype(int)

    # Check that arc endpoints refer to existing nodes
    invalid_endpoints = (
        (arcs["from_node_id"] < 0)
        | (arcs["from_node_id"] >= len(nodes))
        | (arcs["to_node_id"] < 0)
        | (arcs["to_node_id"] >= len(nodes))
    )

    if invalid_endpoints.any():
        raise ValueError(
            f"{trip_id}: {invalid_endpoints.sum()} invalid arc endpoints."
        )

    # Preserve row-index-based arc IDs
    arcs.insert(0, "arc_id", np.arange(len(arcs)))
    arcs.insert(0, "trip_id", trip_id)

    # --------------------------------------------------
    # D. REFERENCE ROUTE PREPROCESSING
    # --------------------------------------------------

    route = None

    if route_file.exists():

        route_raw = read_dataset_file(route_file)

        route = route_raw.iloc[:, 0].copy()
        route = pd.to_numeric(route, errors="coerce")

        if route.isna().any():
            raise ValueError(f"{trip_id}: invalid route arc IDs.")

        if not np.equal(route.to_numpy(), route.to_numpy().astype(int)).all():
            raise ValueError(f"{trip_id}: non-integer route IDs found.")

        route = route.astype(int)

        invalid_route_ids = (
            (route < 0) | (route >= len(arcs))
        )

        if invalid_route_ids.any():
            raise ValueError(
                f"{trip_id}: invalid route arc IDs found."
            )

        route = pd.DataFrame({
            "trip_id": trip_id,
            "route_position": np.arange(len(route)),
            "arc_id": route
        })

    # --------------------------------------------------
    # E. SAVE PROCESSED FILES
    # --------------------------------------------------

    output_folder = OUTPUT_DIR / partition / trip_id
    output_folder.mkdir(parents=True, exist_ok=True)

    track.to_csv(output_folder / "track.csv", index=False)
    nodes.to_csv(output_folder / "nodes.csv", index=False)
    arcs.to_csv(output_folder / "arcs.csv", index=False)

    if route is not None:
        route.to_csv(output_folder / "route.csv", index=False)

    # --------------------------------------------------
    # F. RETURN TRIP SUMMARY
    # --------------------------------------------------

    return {
        "partition": partition,
        "trip_id": trip_id,
        "gps_observations": len(track),
        "road_nodes": len(nodes),
        "road_arcs": len(arcs),
        "route_arcs": len(route) if route is not None else 0,
        "max_time_gap_sec": track["time_delta_sec"].max(),
        "mean_displacement_m": track["displacement_m"].mean(),
        "mean_gps_speed_mps": track["gps_speed_mps"].mean(),
        "status": "success"
    }


# --------------------------------------------------
# 4. PROCESS ALL TRIPS
# --------------------------------------------------

def main():

    summaries = []
    failures = []

    for partition in PARTITIONS:

        partition_dir = DATA_DIR / partition

        if not partition_dir.exists():
            print(f"Skipping missing partition: {partition}")
            continue

        trip_folders = sorted(
            folder for folder in partition_dir.iterdir()
            if folder.is_dir()
        )

        print(f"\nFound {len(trip_folders)} trips in {partition}")

        for trip_folder in trip_folders:

            try:
                result = preprocess_trip(partition, trip_folder)
                summaries.append(result)

                print(
                    f"SUCCESS: {trip_folder.name} | "
                    f"GPS: {result['gps_observations']} | "
                    f"Nodes: {result['road_nodes']} | "
                    f"Arcs: {result['road_arcs']}"
                )

            except Exception as error:

                failures.append({
                    "partition": partition,
                    "trip_id": trip_folder.name,
                    "error": str(error)
                })

                print(f"FAILED: {trip_folder.name} | {error}")

    # Save overall reports
    report_dir = ROOT / "reports" / "preprocessing"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summaries)
    failure_df = pd.DataFrame(failures)

    summary_df.to_csv(
        report_dir / "preprocessing_summary.csv",
        index=False
    )

    failure_df.to_csv(
        report_dir / "preprocessing_failures.csv",
        index=False
    )

    # Final report
    print("\n========== PREPROCESSING COMPLETE ==========")
    print("Successfully processed:", len(summaries))
    print("Failed trips:", len(failures))

    if not summary_df.empty:
        print("\nPartition summary:")
        print(
            summary_df.groupby("partition")[
                ["gps_observations", "road_nodes", "road_arcs", "route_arcs"]
            ].sum()
        )

    if failures:
        print("\nCheck reports/preprocessing/preprocessing_failures.csv")

    print("\nProcessed data saved in:")
    print(OUTPUT_DIR)
    print("============================================")


if __name__ == "__main__":
    main()
