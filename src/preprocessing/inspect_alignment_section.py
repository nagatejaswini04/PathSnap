
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

TRIP_ID = "00000028"
DATA_DIR = ROOT / "data" / "train" / TRIP_ID
OUTPUT_DIR = ROOT / "reports" / "candidate_generation"

track_file = DATA_DIR / f"{TRIP_ID}.track"
nodes_file = DATA_DIR / f"{TRIP_ID}.nodes"
arcs_file = DATA_DIR / f"{TRIP_ID}.arcs"
route_file = DATA_DIR / f"{TRIP_ID}.route"

def load_file(path):
    return pd.read_csv(
        path,
        sep=r"[\s,]+",
        header=None,
        engine="python",
        comment="#"
    )

print("Loading trip data...")

track = load_file(track_file)
nodes = load_file(nodes_file)
arcs = load_file(arcs_file)
route = load_file(route_file)

# GPS observations around the mismatch
gps_start = 1285
gps_end = 1311

gps_section = track.iloc[gps_start:gps_end].copy()
gps_section.columns = ["longitude", "latitude", "timestamp"] + [
    f"extra_{i}" for i in range(max(0, track.shape[1] - 3))
]

gps_section.insert(
    0, "gps_index", range(gps_start, gps_start + len(gps_section))
)

# Reference route positions around the mismatch
route_start = 180
route_end = 206

route_arc_ids = route.iloc[:, 0].astype(int).tolist()
node_coordinates = nodes.iloc[:, :2].astype(float).values
arc_data = arcs.iloc[:, :2].astype(int).values

route_records = []

for position in range(route_start, min(route_end + 1, len(route_arc_ids))):
    arc_id = route_arc_ids[position]

    from_node, to_node = arc_data[arc_id]

    route_records.append({
        "route_position": position,
        "arc_id": arc_id,
        "from_node_id": from_node,
        "to_node_id": to_node,
        "from_longitude": node_coordinates[from_node, 0],
        "from_latitude": node_coordinates[from_node, 1],
        "to_longitude": node_coordinates[to_node, 0],
        "to_latitude": node_coordinates[to_node, 1],
    })

route_section = pd.DataFrame(route_records)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

gps_output = OUTPUT_DIR / f"{TRIP_ID}_gps_alignment_section.csv"
route_output = OUTPUT_DIR / f"{TRIP_ID}_route_alignment_section.csv"

gps_section.to_csv(gps_output, index=False)
route_section.to_csv(route_output, index=False)

print("\nGPS SECTION")
print(gps_section.to_string(index=False))

print("\nREFERENCE ROUTE SECTION")
print(route_section.to_string(index=False))

print("\nSaved files:")
print(gps_output)
print(route_output)

print("\nAlignment section inspection completed.")
