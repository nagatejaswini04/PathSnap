
"""
PathSnap - Candidate Recall Validation

Checks whether reference-route arcs appear in the
generated candidate lists for each GPS observation.

This is a preliminary route-level coverage check.
It does not establish exact GPS-to-route alignment.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TRIP_ID = "00000001"

DATA_DIR = ROOT / "data" / "train" / TRIP_ID

CANDIDATE_FILE = (
    ROOT / "data" / "processed" / "candidates"
    / f"{TRIP_ID}_candidates.csv"
)


def main():

    # Load reference route and candidate results.
    route = pd.read_csv(
        DATA_DIR / f"{TRIP_ID}.route",
        sep=r"\s+",
        header=None
    )

    candidates = pd.read_csv(CANDIDATE_FILE)

    # Extract reference arc IDs.
    reference_arc_ids = set(
        route.iloc[:, 0].astype(int)
    )

    # Extract generated candidate arc IDs.
    candidate_arc_ids = set(
        candidates["arc_id"].astype(int)
    )

    # Compare unique arc IDs.
    found_arcs = reference_arc_ids & candidate_arc_ids
    missing_arcs = reference_arc_ids - candidate_arc_ids

    recall = (
        len(found_arcs) / len(reference_arc_ids)
        if reference_arc_ids else 0
    )

    print("\n========== CANDIDATE COVERAGE ==========")

    print("Reference route arc occurrences:", len(route))
    print("Unique reference route arcs:", len(reference_arc_ids))
    print("Unique candidate arcs:", len(candidate_arc_ids))
    print("Reference arcs found:", len(found_arcs))
    print("Reference arcs missing:", len(missing_arcs))

    print(f"\nUnique-arc coverage: {recall:.2%}")

    if missing_arcs:
        print("\nFirst 20 missing reference arc IDs:")
        print(sorted(missing_arcs)[:20])
    else:
        print("\nAll unique reference route arcs appear in candidates.")

    print("\n========================================")


if __name__ == "__main__":
    main()
