
from pathlib import Path

import pandas as pd
import folium


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRIP_ID = "00000000"

TRIP_DIR = (
    ROOT
    / "data"
    / "processed"
    / "trips"
    / "student_test"
    / TRIP_ID
)

PREDICTION_FILE = (
    ROOT
    / "reports"
    / "hmm_viterbi"
    / f"{TRIP_ID}_reconstructed_route.csv"
)

OUTPUT_DIR = (
    ROOT
    / "visualizations"
    / "hmm_viterbi"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    print("\nLoading trip data...")

    track = pd.read_csv(
        TRIP_DIR / "track.csv"
    )

    nodes = pd.read_csv(
        TRIP_DIR / "nodes.csv"
    )

    arcs = pd.read_csv(
        TRIP_DIR / "arcs.csv"
    )

    reference = pd.read_csv(
        TRIP_DIR / "route.csv"
    )

    predicted = pd.read_csv(
        PREDICTION_FILE
    )

    track = track.sort_values(
        "gps_index"
    ).reset_index(drop=True)

    reference = reference.sort_values(
        "route_position"
    ).reset_index(drop=True)

    predicted = predicted.sort_values(
        "gps_index"
    ).reset_index(drop=True)

    return track, nodes, arcs, reference, predicted


# ============================================================
# 3. CONVERT ROAD ARCS INTO MAP COORDINATES
# ============================================================

def get_arc_coordinates(arc_ids, arcs, nodes):

    node_lookup = (
        nodes.drop_duplicates("node_id")
        .set_index("node_id")
    )

    arc_lookup = (
        arcs.drop_duplicates("arc_id")
        .set_index("arc_id")
    )

    coordinates = []

    missing = 0

    for arc_id in arc_ids:

        if arc_id not in arc_lookup.index:
            missing += 1
            continue

        arc = arc_lookup.loc[arc_id]

        from_node = int(
            arc["from_node_id"]
        )

        to_node = int(
            arc["to_node_id"]
        )

        if (
            from_node not in node_lookup.index
            or to_node not in node_lookup.index
        ):
            missing += 1
            continue

        start = node_lookup.loc[from_node]
        end = node_lookup.loc[to_node]

        segment = [
            (
                float(start["latitude"]),
                float(start["longitude"])
            ),
            (
                float(end["latitude"]),
                float(end["longitude"])
            )
        ]

        coordinates.append(segment)

    return coordinates, missing


# ============================================================
# 4. ADD ROUTE LAYER
# ============================================================

def add_route_layer(
    route_map,
    coordinates,
    layer_name,
    color,
    weight=4,
    opacity=0.8
):

    layer = folium.FeatureGroup(
        name=layer_name,
        show=True
    )

    for segment in coordinates:

        folium.PolyLine(
            locations=segment,
            color=color,
            weight=weight,
            opacity=opacity,
            tooltip=layer_name
        ).add_to(layer)

    layer.add_to(route_map)

    return layer


# ============================================================
# 5. MAIN VISUALIZATION
# ============================================================

def main():

    print("=" * 65)
    print("PATHSNAP - FOLIUM ROUTE VISUALIZATION")
    print("=" * 65)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    track, nodes, arcs, reference, predicted = load_data()

    print("Trip ID:", TRIP_ID)

    print("GPS observations:", len(track))

    print("Road nodes:", len(nodes))

    print("Road arcs:", len(arcs))

    print("Reference route arcs:", len(reference))

    print("Predicted route arcs:", len(predicted))

    # --------------------------------------------------------
    # GPS coordinates
    # --------------------------------------------------------

    gps_coordinates = list(zip(
        track["latitude"].astype(float),
        track["longitude"].astype(float)
    ))

    if not gps_coordinates:
        raise ValueError("GPS track is empty")

    # --------------------------------------------------------
    # Reference route coordinates
    # --------------------------------------------------------

    reference_ids = (
        reference["arc_id"]
        .astype(int)
        .tolist()
    )

    reference_coordinates, missing_reference = (
        get_arc_coordinates(
            reference_ids,
            arcs,
            nodes
        )
    )

    # --------------------------------------------------------
    # Predicted route coordinates
    # --------------------------------------------------------

    predicted_ids = (
        predicted["predicted_arc_id"]
        .astype(int)
        .tolist()
    )

    predicted_coordinates, missing_predicted = (
        get_arc_coordinates(
            predicted_ids,
            arcs,
            nodes
        )
    )

    # --------------------------------------------------------
    # Calculate map center
    # --------------------------------------------------------

    center = [
        float(track["latitude"].mean()),
        float(track["longitude"].mean())
    ]

    # --------------------------------------------------------
    # Create Folium map
    # CartoDB Positron instead of OpenStreetMap
    # --------------------------------------------------------

    route_map = folium.Map(
        location=center,
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True
    )

    # --------------------------------------------------------
    # Layer 1: Original GPS trajectory
    # --------------------------------------------------------

    gps_layer = folium.FeatureGroup(
        name="Original GPS trajectory",
        show=True
    )

    folium.PolyLine(
        locations=gps_coordinates,
        color="blue",
        weight=3,
        opacity=0.85,
        tooltip="Original GPS trajectory"
    ).add_to(gps_layer)

    # GPS start marker
    folium.Marker(
        location=gps_coordinates[0],
        tooltip="GPS Start",
        popup="Starting GPS observation",
        icon=folium.Icon(
            color="blue",
            icon="play"
        )
    ).add_to(gps_layer)

    # GPS end marker
    folium.Marker(
        location=gps_coordinates[-1],
        tooltip="GPS End",
        popup="Final GPS observation",
        icon=folium.Icon(
            color="darkblue",
            icon="flag"
        )
    ).add_to(gps_layer)

    gps_layer.add_to(route_map)

    # --------------------------------------------------------
    # Layer 2: Reference route
    # --------------------------------------------------------

    add_route_layer(
        route_map=route_map,
        coordinates=reference_coordinates,
        layer_name="Reference route",
        color="green",
        weight=5,
        opacity=0.8
    )

    # --------------------------------------------------------
    # Layer 3: HMM + Viterbi reconstructed route
    # --------------------------------------------------------

    add_route_layer(
        route_map=route_map,
        coordinates=predicted_coordinates,
        layer_name="HMM + Viterbi reconstructed route",
        color="red",
        weight=4,
        opacity=0.85
    )

    # --------------------------------------------------------
    # Layer control
    # --------------------------------------------------------

    folium.LayerControl(
        position="topright",
        collapsed=False
    ).add_to(route_map)

    # --------------------------------------------------------
    # Fit map to GPS trajectory
    # --------------------------------------------------------

    latitudes = track["latitude"].astype(float)
    longitudes = track["longitude"].astype(float)

    route_map.fit_bounds([
        [latitudes.min(), longitudes.min()],
        [latitudes.max(), longitudes.max()]
    ])

    # --------------------------------------------------------
    # Save map
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / f"{TRIP_ID}_route_comparison.html"
    )

    route_map.save(
        str(output_file)
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("VISUALIZATION SUMMARY")
    print("=" * 65)

    print("GPS trajectory points:", len(gps_coordinates))

    print(
        "Reference route segments:",
        len(reference_coordinates)
    )

    print(
        "Predicted route segments:",
        len(predicted_coordinates)
    )

    print(
        "Missing reference arcs:",
        missing_reference
    )

    print(
        "Missing predicted arcs:",
        missing_predicted
    )

    print("\nMap saved to:")

    print(output_file)

    print("\nFOLIUM VISUALIZATION COMPLETED")


if __name__ == "__main__":
    main()
