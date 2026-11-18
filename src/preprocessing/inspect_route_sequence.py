
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]

TRIP_ID = "00000028"

TRAIN_DIR = ROOT / "data" / "train"
REPORT_DIR = ROOT / "reports" / "candidate_generation"


def project(lon, lat, reference_lat):
    x = np.asarray(lon) * 111320 * np.cos(
        np.radians(reference_lat)
    )
    y = np.asarray(lat) * 110540
    return np.column_stack((x, y))


def point_segment_distance(point, start, end):
    segment = end - start
    denominator = np.dot(segment, segment)

    if denominator == 0:
        return float(np.linalg.norm(point - start))

    fraction = np.dot(point - start, segment) / denominator
    fraction = np.clip(fraction, 0, 1)

    closest = start + fraction * segment

    return float(np.linalg.norm(point - closest))


def main():

    trip_dir = TRAIN_DIR / TRIP_ID

    track = pd.read_csv(
        trip_dir / f"{TRIP_ID}.track",
        sep=r"\s+",
        header=None
    )

    nodes = pd.read_csv(
        trip_dir / f"{TRIP_ID}.nodes",
        sep=r"\s+",
        header=None
    )

    arcs = pd.read_csv(
        trip_dir / f"{TRIP_ID}.arcs",
        sep=r"\s+",
        header=None
    )

    route = pd.read_csv(
        trip_dir / f"{TRIP_ID}.route",
        sep=r"\s+",
        header=None
    ).iloc[:, 0].astype(int).to_numpy()

    missing_df = pd.read_csv(
        REPORT_DIR / "missing_candidate_details.csv"
    )

    missing_ids = set(
        missing_df.loc[
            missing_df["trip_id"].astype(str).str.zfill(8)
            == TRIP_ID,
            "missing_arc_id"
        ].astype(int)
    )

    reference_lat = track.iloc[:, 1].mean()

    gps_points = project(
        track.iloc[:, 0],
        track.iloc[:, 1],
        reference_lat
    )

    node_points = project(
        nodes.iloc[:, 0],
        nodes.iloc[:, 1],
        reference_lat
    )

    gps_tree = cKDTree(gps_points)

    records = []

    for route_position, arc_id in enumerate(route):

        if arc_id < 0 or arc_id >= len(arcs):
            continue

        from_node = int(arcs.iloc[arc_id, 0])
        to_node = int(arcs.iloc[arc_id, 1])

        start = node_points[from_node]
        end = node_points[to_node]

        midpoint = (start + end) / 2

        midpoint_distance, nearest_index = gps_tree.query(
            midpoint
        )

        exact_distance = point_segment_distance(
            gps_points[nearest_index],
            start,
            end
        )

        records.append({
            "route_position": route_position,
            "arc_id": arc_id,
            "is_missing_from_top50": arc_id in missing_ids,
            "nearest_gps_index": int(nearest_index),
            "midpoint_distance_m": float(midpoint_distance),
            "segment_distance_at_nearest_midpoint_gps_m": exact_distance,
            "arc_length_m": float(np.linalg.norm(end - start))
        })

    result = pd.DataFrame(records)

    output = REPORT_DIR / f"{TRIP_ID}_route_sequence_diagnostic.csv"
    result.to_csv(output, index=False)

    print("\n========== ROUTE SEQUENCE DIAGNOSTIC ==========")
    print("Trip:", TRIP_ID)
    print("GPS observations:", len(track))
    print("Reference route arcs:", len(route))
    print("Missing unique arcs:", len(missing_ids))

    print("\nMissing arcs and their route positions:")

    missing_positions = result[
        result["is_missing_from_top50"]
    ]["route_position"].tolist()

    for position in missing_positions:

        print(f"\nRoute position: {position}")

        start = max(0, position - 2)
        end = min(len(result), position + 3)

        print(
            result.iloc[start:end].to_string(index=False)
        )

    print("\nNearest GPS index progression:")
    print(
        result["nearest_gps_index"].describe()
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
