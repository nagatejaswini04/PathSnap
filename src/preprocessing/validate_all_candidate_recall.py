
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = ROOT / "data" / "train"
CANDIDATE_DIR = ROOT / "data" / "processed" / "candidates" / "train"

REPORT_DIR = ROOT / "reports" / "candidate_generation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def validate_trip(trip_folder):

    trip_id = trip_folder.name

    route_file = trip_folder / f"{trip_id}.route"
    candidate_file = CANDIDATE_DIR / f"{trip_id}_candidates.csv"

    route = pd.read_csv(
        route_file, sep=r"\s+", header=None
    ).iloc[:, 0].astype(int)

    candidates = pd.read_csv(candidate_file)

    reference_arcs = set(route)
    candidate_arcs = set(candidates["arc_id"].astype(int))

    found = reference_arcs & candidate_arcs
    missing = reference_arcs - candidate_arcs

    recall = (
        len(found) / len(reference_arcs)
        if reference_arcs else 0
    )

    return {
        "trip_id": trip_id,
        "reference_route_arcs": len(reference_arcs),
        "candidate_unique_arcs": len(candidate_arcs),
        "found_reference_arcs": len(found),
        "missing_reference_arcs": len(missing),
        "unique_arc_coverage": recall,
        "missing_arc_ids": ",".join(map(str, sorted(missing)))
    }


def main():

    results = []

    trip_folders = sorted(
        folder for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
    )

    for folder in trip_folders:

        try:
            result = validate_trip(folder)
            results.append(result)

            print(
                f"{result['trip_id']}: "
                f"Coverage = {result['unique_arc_coverage']:.2%}, "
                f"Missing = {result['missing_reference_arcs']}"
            )

        except Exception as error:
            print(f"FAILED {folder.name}: {error}")

    df = pd.DataFrame(results)

    df.to_csv(
        REPORT_DIR / "candidate_recall_by_trip.csv",
        index=False
    )

    print("\n========== OVERALL RESULTS ==========")
    print("Trips validated:", len(df))

    if not df.empty:
        print(
            "Mean trip-level unique-arc coverage:",
            f"{df['unique_arc_coverage'].mean():.2%}"
        )

        print(
            "Trips with 100% coverage:",
            (df["missing_reference_arcs"] == 0).sum()
        )

        print(
            "Total missing unique route arcs:",
            df["missing_reference_arcs"].sum()
        )

    print("\nReport saved:")
    print(REPORT_DIR / "candidate_recall_by_trip.csv")


if __name__ == "__main__":
    main()
