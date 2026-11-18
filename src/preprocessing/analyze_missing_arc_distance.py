
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = ROOT / "data" / "train"
REPORT_DIR = ROOT / "reports" / "candidate_generation"

COMPARISON_FILE = REPORT_DIR / "top20_vs_top50_comparison.csv"
OUTPUT_FILE = REPORT_DIR / "missing_arc_distance_analysis.csv"


def project_coordinates(lon, lat, reference_lat):
    """Convert longitude/latitude to approximate local metre coordinates."""
    x = np.asarray(lon) * 111320 * np.cos(np.radians(reference_lat))
    y = np.asarray(lat) * 110540
    return np.column_stack((x, y))


def point_to_segment_distances(points, start, end):
    """Calculate distances from many GPS points to one road segment."""

    segment = end - start
    denominator = np.dot(segment, segment)

    if denominator == 0:
        return np.linalg.norm(points - start, axis=1)

    relative = points - start

    projection = (
        np.sum(relative * segment, axis=1) / denominator
    )

    projection = np.clip(projection, 0, 1)

    closest_points = start + projection[:, None] * segment

    return np.linalg.norm(points - closest_points, axis=1)


def main():

    comparison = pd.read_csv(COMPARISON_FILE)

    problematic = comparison[
        comparison["top50_missing"] > 0
    ]

    results = []

    for _, row in problematic.iterrows():

        trip_id = str(row["trip_id"]).zfill(8)

        print(f"\nAnalyzing trip {trip_id}")

        trip_dir = TRAIN_DIR / trip_id

        track = pd.read_csv(
            trip_dir / f"{trip_id}.track",
            sep=r"\s+",
            header=None
        )

        nodes = pd.read_csv(
            trip_dir / f"{trip_id}.nodes",
            sep=r"\s+",
            header=None
        )

        arcs = pd.read_csv(
            trip_dir / f"{trip_id}.arcs",
            sep=r"\s+",
            header=None
        )

        missing_ids = [
            int(x)
            for x in str(row["top50_missing_arc_ids"]).split(",")
            if x.strip()
        ]

        reference_lat = track.iloc[:, 1].mean()

        gps_points = project_coordinates(
            track.iloc[:, 0].to_numpy(),
            track.iloc[:, 1].to_numpy(),
            reference_lat
        )

        node_points = project_coordinates(
            nodes.iloc[:, 0].to_numpy(),
            nodes.iloc[:, 1].to_numpy(),
            reference_lat
        )

        for arc_id in missing_ids:

            if arc_id < 0 or arc_id >= len(arcs):
                continue

            from_node = int(arcs.iloc[arc_id, 0])
            to_node = int(arcs.iloc[arc_id, 1])

            start = node_points[from_node]
            end = node_points[to_node]

            distances = point_to_segment_distances(
                gps_points, start, end
            )

            nearest_gps_index = int(np.argmin(distances))
            minimum_distance = float(distances[nearest_gps_index])

            results.append({
                "trip_id": trip_id,
                "missing_arc_id": arc_id,
                "nearest_gps_index": nearest_gps_index,
                "minimum_segment_distance_m": minimum_distance,
                "nearest_gps_longitude": track.iloc[
                    nearest_gps_index, 0
                ],
                "nearest_gps_latitude": track.iloc[
                    nearest_gps_index, 1
                ],
                "arc_length_m": float(np.linalg.norm(end - start))
            })

        print(f"Analyzed {len(missing_ids)} missing arcs")

    result_df = pd.DataFrame(results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\n========== DISTANCE ANALYSIS ==========")
    print("Total missing arcs analyzed:", len(result_df))

    if not result_df.empty:

        print("\nDistance summary:")
        print(
            result_df["minimum_segment_distance_m"].describe()
        )

        print("\nClosest missing arcs:")
        print(
            result_df.sort_values(
                "minimum_segment_distance_m"
            ).head(10).to_string(index=False)
        )

        print("\nFarthest missing arcs:")
        print(
            result_df.sort_values(
                "minimum_segment_distance_m",
                ascending=False
            ).head(10).to_string(index=False)
        )

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
