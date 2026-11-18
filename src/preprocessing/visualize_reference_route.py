
from pathlib import Path

import pandas as pd
import folium


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRIP_ID = "00000028"

DATA_DIR = PROJECT_ROOT / "data" / "train" / TRIP_ID
OUTPUT_DIR = PROJECT_ROOT / "visualizations"

TRACK_FILE = DATA_DIR / f"{TRIP_ID}.track"
NODES_FILE = DATA_DIR / f"{TRIP_ID}.nodes"
ARCS_FILE = DATA_DIR / f"{TRIP_ID}.arcs"
ROUTE_FILE = DATA_DIR / f"{TRIP_ID}.route"

OUTPUT_FILE = OUTPUT_DIR / f"{TRIP_ID}_route_comparison.html"


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_file(file_path):
    """
    Load a whitespace-separated or comma-separated dataset.
    Files are expected to have no header.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(
        file_path,
        sep=r"[\s,]+",
        header=None,
        engine="python",
        comment="#"
    )


def load_trip_data():

    print("\nLoading trip data...")

    track = load_file(TRACK_FILE)
    nodes = load_file(NODES_FILE)
    arcs = load_file(ARCS_FILE)
    route = load_file(ROUTE_FILE)

    if track.shape[1] < 2:
        raise ValueError("Track file must contain longitude and latitude.")

    if nodes.shape[1] < 2:
        raise ValueError("Nodes file must contain longitude and latitude.")

    if arcs.shape[1] < 2:
        raise ValueError("Arcs file must contain from_node_id and to_node_id.")

    # Track: longitude, latitude, timestamp
    track_coordinates = list(
        zip(
            track.iloc[:, 1].astype(float),
            track.iloc[:, 0].astype(float)
        )
    )

    # Nodes: longitude, latitude
    node_coordinates = nodes.iloc[:, :2].astype(float).values

    # Arcs: from_node_id, to_node_id
    arc_data = arcs.iloc[:, :2].astype(int).values

    # Route: chronological sequence of arc IDs
    route_arc_ids = route.iloc[:, 0].astype(int).tolist()

    print(f"GPS observations: {len(track_coordinates)}")
    print(f"Road network nodes: {len(node_coordinates)}")
    print(f"Road network arcs: {len(arc_data)}")
    print(f"Reference route arcs: {len(route_arc_ids)}")

    return track_coordinates, node_coordinates, arc_data, route_arc_ids


# ============================================================
# 3. RECONSTRUCT REFERENCE ROUTE COORDINATES
# ============================================================

def build_reference_route(node_coordinates, arc_data, route_arc_ids):

    print("\nReconstructing reference route...")

    route_coordinates = []

    for position, arc_id in enumerate(route_arc_ids):

        if arc_id < 0 or arc_id >= len(arc_data):
            raise ValueError(
                f"Invalid arc ID {arc_id} at route position {position}"
            )

        from_node_id, to_node_id = arc_data[arc_id]

        if (
            from_node_id < 0
            or to_node_id < 0
            or from_node_id >= len(node_coordinates)
            or to_node_id >= len(node_coordinates)
        ):
            raise ValueError(
                f"Invalid node index in arc {arc_id}"
            )

        # Convert node coordinates from (longitude, latitude)
        # to Folium's (latitude, longitude).

        start_node = (
            node_coordinates[from_node_id][1],
            node_coordinates[from_node_id][0]
        )

        end_node = (
            node_coordinates[to_node_id][1],
            node_coordinates[to_node_id][0]
        )

        # Add the first arc's starting node only once.
        if position == 0:
            route_coordinates.append(start_node)

        # Add the ending node of every arc.
        route_coordinates.append(end_node)

    print(f"Reference route coordinates: {len(route_coordinates)}")

    return route_coordinates


# ============================================================
# 4. CREATE MAP
# ============================================================

def create_route_map(gps_coordinates, route_coordinates):

    print("\nCreating Folium map...")

    # Calculate an initial center from both trajectories.
    all_coordinates = gps_coordinates + route_coordinates

    center_latitude = sum(p[0] for p in all_coordinates) / len(all_coordinates)
    center_longitude = sum(p[1] for p in all_coordinates) / len(all_coordinates)

    map_view = folium.Map(
        location=[center_latitude, center_longitude],
        zoom_start=8,
        tiles="OpenStreetMap"
    )

    # --------------------------------------------------------
    # 4A. Reference route - RED SOLID LINE
    # --------------------------------------------------------

    folium.PolyLine(
        locations=route_coordinates,
        color="red",
        weight=4,
        opacity=0.85,
        tooltip="Reference Route"
    ).add_to(map_view)

    # --------------------------------------------------------
    # 4B. GPS trajectory - BLUE DASHED LINE
    # Added after the red line so it remains visible.
    # --------------------------------------------------------

    folium.PolyLine(
        locations=gps_coordinates,
        color="blue",
        weight=4,
        opacity=0.9,
        dash_array="8, 8",
        tooltip="GPS Trajectory"
    ).add_to(map_view)

    # --------------------------------------------------------
    # 4C. Start and end markers
    # --------------------------------------------------------

    folium.Marker(
        location=gps_coordinates[0],
        tooltip="GPS Start",
        popup="Start of GPS trajectory",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(map_view)

    folium.Marker(
        location=gps_coordinates[-1],
        tooltip="GPS End",
        popup="End of GPS trajectory",
        icon=folium.Icon(color="black", icon="stop")
    ).add_to(map_view)

    # --------------------------------------------------------
    # 4D. Legend
    # --------------------------------------------------------

    legend_html = """
    <div style="
        position: fixed;
        bottom: 35px;
        left: 35px;
        width: 220px;
        background-color: white;
        border: 2px solid grey;
        z-index: 9999;
        padding: 10px;
        font-size: 14px;
        border-radius: 5px;
    ">
        <b>PathSnap - Route Comparison</b><br><br>

        <span style="
            color: blue;
            font-size: 20px;
            font-weight: bold;
        ">━━ ━ ━</span>
        GPS Trajectory<br>

        <span style="
            color: red;
            font-size: 20px;
            font-weight: bold;
        ">━━━━━━━━</span>
        Reference Route<br>
    </div>
    """

    map_view.get_root().html.add_child(
        folium.Element(legend_html)
    )

    # --------------------------------------------------------
    # 4E. Automatically fit map to both trajectories
    # --------------------------------------------------------

    map_view.fit_bounds(all_coordinates)

    return map_view


# ============================================================
# 5. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 60)
    print("PATHSNAP - GPS AND REFERENCE ROUTE VISUALIZATION")
    print("=" * 60)

    # Load data
    gps_coordinates, node_coordinates, arc_data, route_arc_ids = (
        load_trip_data()
    )

    if not gps_coordinates or not route_arc_ids:
        raise ValueError("GPS trajectory or reference route is empty.")

    # Reconstruct route
    route_coordinates = build_reference_route(
        node_coordinates,
        arc_data,
        route_arc_ids
    )

    # Create map
    map_view = create_route_map(
        gps_coordinates,
        route_coordinates
    )

    # Save HTML
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    map_view.save(str(OUTPUT_FILE))

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nMap saved at:\n{OUTPUT_FILE}")

    print(f"\nGPS observations: {len(gps_coordinates)}")
    print(f"Reference route arcs: {len(route_arc_ids)}")
    print(f"Reference route coordinates: {len(route_coordinates)}")

    print("\nOpen the HTML file in Google Chrome to view the map.")


if __name__ == "__main__":
    main()
