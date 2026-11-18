
from pathlib import Path

import pandas as pd
import folium


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRIP_ID = "00000028"

DATA_DIR = ROOT / "data" / "train" / TRIP_ID
REPORT_DIR = ROOT / "reports" / "candidate_generation"
OUTPUT_DIR = ROOT / "visualizations"

TRACK_FILE = DATA_DIR / f"{TRIP_ID}.track"
NODES_FILE = DATA_DIR / f"{TRIP_ID}.nodes"
ARCS_FILE = DATA_DIR / f"{TRIP_ID}.arcs"
ROUTE_FILE = DATA_DIR / f"{TRIP_ID}.route"

MISSING_FILE = REPORT_DIR / "missing_candidate_details.csv"

OUTPUT_FILE = OUTPUT_DIR / "00000028_missing_route_section.html"

# Focused region
GPS_START = 1250
GPS_END = 1350

ROUTE_START = 175
ROUTE_END = 210


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path):
    return pd.read_csv(
        path,
        sep=r"[\s,]+",
        header=None,
        engine="python",
        comment="#"
    )


print("Loading trip data...")

track = load_data(TRACK_FILE)
nodes = load_data(NODES_FILE)
arcs = load_data(ARCS_FILE)
route = load_data(ROUTE_FILE)

# GPS coordinates in Folium order: latitude, longitude
gps_coordinates = list(zip(
    track.iloc[:, 1].astype(float),
    track.iloc[:, 0].astype(float)
))

# Road nodes: longitude, latitude
node_coordinates = nodes.iloc[:, :2].astype(float).values

# Road arcs: from_node_id, to_node_id
arc_data = arcs.iloc[:, :2].astype(int).values

# Chronological reference route
route_arc_ids = route.iloc[:, 0].astype(int).tolist()


# ============================================================
# LOAD MISSING ARC IDs
# ============================================================

missing = pd.read_csv(MISSING_FILE)

missing_trip = missing[
    missing["trip_id"].astype(str).str.zfill(8) == TRIP_ID
]

missing_arc_ids = set(missing_trip["missing_arc_id"].astype(int))

print("Missing arcs for this trip:", len(missing_arc_ids))


# ============================================================
# BUILD FOCUSED GPS SECTION
# ============================================================

gps_start = max(0, GPS_START)
gps_end = min(len(gps_coordinates), GPS_END)

gps_section = gps_coordinates[gps_start:gps_end]


# ============================================================
# BUILD REFERENCE ROUTE SECTION
# ============================================================

route_section = []

for position in range(ROUTE_START, min(ROUTE_END + 1, len(route_arc_ids))):

    arc_id = route_arc_ids[position]

    if arc_id < 0 or arc_id >= len(arc_data):
        continue

    from_node, to_node = arc_data[arc_id]

    if (
        from_node < 0 or to_node < 0
        or from_node >= len(node_coordinates)
        or to_node >= len(node_coordinates)
    ):
        continue

    start = (
        node_coordinates[from_node][1],
        node_coordinates[from_node][0]
    )

    end = (
        node_coordinates[to_node][1],
        node_coordinates[to_node][0]
    )

    if not route_section:
        route_section.append(start)

    route_section.append(end)


# ============================================================
# CREATE MAP
# ============================================================

all_coordinates = gps_section + route_section

if not all_coordinates:
    raise ValueError("No coordinates available for focused map.")

center = [
    sum(point[0] for point in all_coordinates) / len(all_coordinates),
    sum(point[1] for point in all_coordinates) / len(all_coordinates)
]

map_view = folium.Map(
    location=center,
    zoom_start=13,
    tiles="OpenStreetMap"
)


# ============================================================
# GPS OBSERVATIONS
# ============================================================

folium.PolyLine(
    gps_section,
    color="blue",
    weight=4,
    opacity=0.9,
    dash_array="6, 6",
    tooltip="GPS trajectory"
).add_to(map_view)

# Add GPS observation markers
for index in range(gps_start, gps_end):

    lat, lon = gps_coordinates[index]

    folium.CircleMarker(
        location=[lat, lon],
        radius=3,
        color="blue",
        fill=True,
        fill_opacity=0.8,
        tooltip=f"GPS observation {index}"
    ).add_to(map_view)


# ============================================================
# REFERENCE ROUTE ARCS
# ============================================================

for position in range(
    ROUTE_START,
    min(ROUTE_END + 1, len(route_arc_ids))
):

    arc_id = route_arc_ids[position]

    if arc_id < 0 or arc_id >= len(arc_data):
        continue

    from_node, to_node = arc_data[arc_id]

    if (
        from_node < 0 or to_node < 0
        or from_node >= len(node_coordinates)
        or to_node >= len(node_coordinates)
    ):
        continue

    start = [
        node_coordinates[from_node][1],
        node_coordinates[from_node][0]
    ]

    end = [
        node_coordinates[to_node][1],
        node_coordinates[to_node][0]
    ]

    is_missing = arc_id in missing_arc_ids

    color = "orange" if is_missing else "red"
    weight = 7 if is_missing else 4

    folium.PolyLine(
        [start, end],
        color=color,
        weight=weight,
        opacity=0.9,
        tooltip=(
            f"Route position: {position}, Arc ID: {arc_id}, "
            f"{'MISSING FROM TOP-50' if is_missing else 'Found in TOP-50'}"
        )
    ).add_to(map_view)

    # Mark the midpoint of each missing arc
    if is_missing:
        midpoint = [
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2
        ]

        folium.CircleMarker(
            location=midpoint,
            radius=6,
            color="orange",
            fill=True,
            fill_color="orange",
            fill_opacity=1,
            tooltip=f"MISSING ARC {arc_id}, position {position}"
        ).add_to(map_view)


# ============================================================
# LEGEND
# ============================================================

legend = """
<div style="
position: fixed;
bottom: 35px;
left: 35px;
z-index: 9999;
background: white;
padding: 12px;
border: 2px solid grey;
border-radius: 5px;
font-size: 13px;
">
<b>Missing Route Investigation</b><br><br>
<span style="color:blue;">━━ ━ ━</span> GPS trajectory<br>
<span style="color:red;">━━━━━━━━</span> Reference arcs found in Top-50<br>
<span style="color:orange;">━━━━━━━━</span> Missing reference arcs
</div>
"""

map_view.get_root().html.add_child(folium.Element(legend))


# ============================================================
# FIT MAP AND SAVE
# ============================================================

map_view.fit_bounds(all_coordinates)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

map_view.save(str(OUTPUT_FILE))

print("\nFocused map generated successfully!")
print("GPS indices:", gps_start, "to", gps_end - 1)
print("Route positions:", ROUTE_START, "to", ROUTE_END)
print("Missing arcs:", len(missing_arc_ids))
print("Saved at:", OUTPUT_FILE)
