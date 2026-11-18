
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = ROOT / "data" / "train"

CANDIDATE_DIR = (
    ROOT / "data" / "processed" / "candidates" / "train"
)

REPORT_DIR = ROOT / "reports" / "candidate_generation"

COMPARISON_FILE = (
    REPORT_DIR / "top20_vs_top50_comparison.csv"
)

OUTPUT_FILE = (
    REPORT_DIR / "missing_candidate_details.csv"
)


def main():

    comparison = pd.read_csv(COMPARISON_FILE)

    # Select trips that still have missing arcs at TOP_K=50
    problematic = comparison[
        comparison["top50_missing"] > 0
    ]

    results = []

    for _, row in problematic.iterrows():

        trip_id = str(row["trip_id"]).zfill(8)

        print(f"\nInspecting trip {trip_id}")

        trip_dir = TRAIN_DIR / trip_id

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

        route = pd.read_csv(
            trip_dir / f"{trip_id}.route",
            sep=r"\s+",
            header=None
        )

        candidates20 = pd.read_csv(
            CANDIDATE_DIR / f"{trip_id}_candidates.csv"
        )

        candidates50 = pd.read_csv(
            ROOT / "data" / "processed" / "candidates_k50"
            / "train" / f"{trip_id}_candidates.csv"
        )

        missing_ids = [
            int(x)
            for x in str(row["top50_missing_arc_ids"]).split(",")
            if x.strip()
        ]

        for arc_id in missing_ids:

            if arc_id < 0 or arc_id >= len(arcs):
                print(f"Invalid arc ID: {arc_id}")
                continue

            from_node = int(arcs.iloc[arc_id, 0])
            to_node = int(arcs.iloc[arc_id, 1])

            results.append({
                "trip_id": trip_id,
                "missing_arc_id": arc_id,
                "from_node_id": from_node,
                "to_node_id": to_node,
                "from_longitude": nodes.iloc[from_node, 0],
                "from_latitude": nodes.iloc[from_node, 1],
                "to_longitude": nodes.iloc[to_node, 0],
                "to_latitude": nodes.iloc[to_node, 1],
                "route_occurrences": int(
                    (route.iloc[:, 0] == arc_id).sum()
                ),
                "present_in_top20": (
                    arc_id in set(candidates20["arc_id"])
                ),
                "present_in_top50": (
                    arc_id in set(candidates50["arc_id"])
                )
            })

    result_df = pd.DataFrame(results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\n========== INSPECTION COMPLETE ==========")
    print("Missing arc records inspected:", len(result_df))
    print("Saved report:", OUTPUT_FILE)

    if not result_df.empty:
        print("\nMissing arcs by trip:")
        print(result_df.groupby("trip_id").size())

        print("\nFirst 10 records:")
        print(result_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
