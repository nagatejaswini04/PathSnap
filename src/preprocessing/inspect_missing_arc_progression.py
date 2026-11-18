
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "candidate_generation"

TRIP_ID = "00000028"

sequence_file = REPORT / f"{TRIP_ID}_route_sequence_diagnostic.csv"
distance_file = REPORT / "missing_arc_distance_analysis.csv"

sequence = pd.read_csv(sequence_file)
distances = pd.read_csv(distance_file)

# Get missing route arcs.
missing = sequence[
    sequence["is_missing_from_top50"].astype(bool)
].copy()

# Get distance information for this trip.
trip_distances = distances[
    distances["trip_id"].astype(str).str.zfill(8) == TRIP_ID
][[
    "missing_arc_id",
    "nearest_gps_index",
    "minimum_segment_distance_m"
]].copy()

# Rename columns before merging to avoid duplicate names.
trip_distances = trip_distances.rename(columns={
    "nearest_gps_index": "nearest_gps_any_index",
    "minimum_segment_distance_m": "minimum_distance_any_gps_m"
})

# Merge using the correct arc ID columns.
merged = missing.merge(
    trip_distances,
    left_on="arc_id",
    right_on="missing_arc_id",
    how="left"
)

print("\n========== MISSING ARC PROGRESSION ==========")
print("Trip:", TRIP_ID)
print("Missing route arcs:", len(merged))

for _, row in merged.iterrows():

    position = int(row["route_position"])

    print("\n" + "=" * 60)
    print("Missing arc ID:", int(row["arc_id"]))
    print("Route position:", position)

    print(
        "Nearest GPS index (route midpoint diagnostic):",
        int(row["nearest_gps_index"])
    )

    print(
        "Nearest GPS index (whole-trip distance analysis):",
        int(row["nearest_gps_any_index"])
    )

    print(
        "Minimum distance to any GPS:",
        round(row["minimum_distance_any_gps_m"], 2),
        "m"
    )

    start = max(0, position - 2)
    end = min(len(sequence), position + 3)

    print("\nNeighbouring route arcs:")

    print(
        sequence.iloc[start:end][[
            "route_position",
            "arc_id",
            "is_missing_from_top50",
            "nearest_gps_index",
            "segment_distance_at_nearest_midpoint_gps_m"
        ]].to_string(index=False)
    )

print("\nDiagnostic complete.")
