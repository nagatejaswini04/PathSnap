
"""
PathSnap - Road Network and GPS Analysis

Investigates one training trip without modifying raw data.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


# ==========================================================
# 1. PATHS AND SETTINGS
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data" / "train"

TRIP_ID = "00000001"

TRIP_DIR = TRAIN_DIR / TRIP_ID

OUTPUT_DIR = ROOT / "reports" / "network_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. LOAD TRIP FILES
# ==========================================================

def load_trip():

    track = pd.read_csv(
        TRIP_DIR / f"{TRIP_ID}.track",
        sep=r"\s+",
        header=None
    )

    nodes = pd.read_csv(
        TRIP_DIR / f"{TRIP_ID}.nodes",
        sep=r"\s+",
        header=None
    )

    arcs = pd.read_csv(
        TRIP_DIR / f"{TRIP_ID}.arcs",
        sep=r"\s+",
        header=None
    )

    route = pd.read_csv(
        TRIP_DIR / f"{TRIP_ID}.route",
        sep=r"\s+",
        header=None
    )

    track.columns = ["longitude", "latitude", "timestamp"]
    nodes.columns = ["longitude", "latitude"]
    arcs.columns = ["from_node_id", "to_node_id"]
    route.columns = ["arc_id"]

    return track, nodes, arcs, route


# ==========================================================
# 3. HAVERSINE DISTANCE
# ==========================================================

def haversine(lon1, lat1, lon2, lat2):

    earth_radius = 6371000

    lon1, lat1, lon2, lat2 = np.radians(
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius * np.arcsin(
        np.sqrt(a)
    )


# ==========================================================
# 4. BASIC DATASET INFORMATION
# ==========================================================

def inspect_sizes(track, nodes, arcs, route):

    print("\n========== DATASET SIZES ==========")

    print("GPS observations:", len(track))
    print("Road nodes:", len(nodes))
    print("Road arcs:", len(arcs))
    print("Reference route arcs:", len(route))

    print("\nGPS sample:")
    print(track.head())

    print("\nNode sample:")
    print(nodes.head())

    print("\nArc sample:")
    print(arcs.head())

    print("\nRoute sample:")
    print(route.head())


# ==========================================================
# 5. VERIFY ROAD NETWORK REFERENCES
# ==========================================================

def validate_network(nodes, arcs, route):

    print("\n========== NETWORK VALIDATION ==========")

    # Node IDs refer to zero-based node row positions.
    node_ids = set(range(len(nodes)))

    valid_arc_endpoints = (
        arcs["from_node_id"].isin(node_ids)
        & arcs["to_node_id"].isin(node_ids)
    )

    print(
        "Invalid arc endpoints:",
        int((~valid_arc_endpoints).sum())
    )

    # Arc IDs refer to zero-based arc row positions.
    arc_ids = set(range(len(arcs)))

    valid_route_arcs = route["arc_id"].isin(arc_ids)

    print(
        "Invalid route arc IDs:",
        int((~valid_route_arcs).sum())
    )

    return valid_arc_endpoints, valid_route_arcs


# ==========================================================
# 6. ROAD SEGMENT LENGTHS
# ==========================================================

def calculate_road_lengths(nodes, arcs):

    print("\n========== ROAD SEGMENT LENGTHS ==========")

    from_ids = arcs["from_node_id"].to_numpy(dtype=int)
    to_ids = arcs["to_node_id"].to_numpy(dtype=int)

    coordinates = nodes[
        ["longitude", "latitude"]
    ].to_numpy()

    start = coordinates[from_ids]
    end = coordinates[to_ids]

    lengths = haversine(
        start[:, 0],
        start[:, 1],
        end[:, 0],
        end[:, 1]
    )

    arcs_result = arcs.copy()
    arcs_result["arc_id"] = np.arange(len(arcs))
    arcs_result["length_meters"] = lengths

    print(arcs_result["length_meters"].describe())

    return arcs_result


# ==========================================================
# 7. ROAD NETWORK CONNECTIVITY
# ==========================================================

def analyze_connectivity(nodes, arcs):

    print("\n========== NETWORK CONNECTIVITY ==========")

    number_of_nodes = len(nodes)

    from_ids = arcs["from_node_id"].to_numpy(dtype=int)
    to_ids = arcs["to_node_id"].to_numpy(dtype=int)

    # Treat the road network as undirected only for
    # identifying weakly connected components.
    row = np.concatenate([from_ids, to_ids])
    col = np.concatenate([to_ids, from_ids])

    adjacency = coo_matrix(
        (
            np.ones(len(row)),
            (row, col)
        ),
        shape=(number_of_nodes, number_of_nodes)
    ).tocsr()

    component_count, labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True
    )

    component_sizes = pd.Series(labels).value_counts()

    print("Connected components:", component_count)
    print("Largest component size:", component_sizes.max())
    print("Smallest component size:", component_sizes.min())

    return labels


# ==========================================================
# 8. REFERENCE ROUTE CONTINUITY
# ==========================================================

def analyze_route(arcs, route):

    print("\n========== REFERENCE ROUTE CONTINUITY ==========")

    route_ids = route["arc_id"].to_numpy(dtype=int)

    from_ids = arcs["from_node_id"].to_numpy(dtype=int)
    to_ids = arcs["to_node_id"].to_numpy(dtype=int)

    route_starts = from_ids[route_ids]
    route_ends = to_ids[route_ids]

    # End of one arc should equal start of next arc.
    connected = (
        route_ends[:-1] == route_starts[1:]
    )

    connected_count = int(connected.sum())
    disconnected_count = int((~connected).sum())

    total_transitions = len(connected)

    continuity = (
        connected_count / total_transitions
        if total_transitions > 0 else 0
    )

    print("Route transitions:", total_transitions)
    print("Connected transitions:", connected_count)
    print("Disconnected transitions:", disconnected_count)
    print(f"Continuity ratio: {continuity:.4f}")

    return connected


# ==========================================================
# 9. GPS TIMESTAMP AND MOVEMENT ANALYSIS
# ==========================================================

def analyze_gps(track):

    print("\n========== GPS ANALYSIS ==========")

    gps = track.copy()

    gps["time_gap_seconds"] = (
        gps["timestamp"].diff()
    )

    gps["gps_distance_meters"] = haversine(
        gps["longitude"].shift(),
        gps["latitude"].shift(),
        gps["longitude"],
        gps["latitude"]
    )

    gps["gps_speed_mps"] = (
        gps["gps_distance_meters"]
        / gps["time_gap_seconds"]
    )

    print("\nTimestamp gap statistics:")
    print(gps["time_gap_seconds"].describe())

    print("\nGPS movement distance statistics:")
    print(gps["gps_distance_meters"].describe())

    print("\nGPS speed statistics (m/s):")
    print(gps["gps_speed_mps"].replace(
        [np.inf, -np.inf], np.nan
    ).describe())

    print(
        "\nGPS intervals greater than 1 second:",
        int((gps["time_gap_seconds"] > 1).sum())
    )

    return gps


# ==========================================================
# 10. NEAREST ROAD NODE ANALYSIS
# ==========================================================

def nearest_road_nodes(track, nodes):

    print("\n========== NEAREST ROAD NODE ANALYSIS ==========")

    # Convert longitude/latitude into approximate
    # local metric coordinates using a spherical projection.
    mean_latitude = np.radians(
        track["latitude"].mean()
    )

    earth_radius = 6371000

    def project(df):

        x = (
            earth_radius
            * np.radians(df["longitude"])
            * np.cos(mean_latitude)
        )

        y = earth_radius * np.radians(df["latitude"])

        return np.column_stack([x, y])

    node_xy = project(nodes)
    gps_xy = project(track)

    tree = cKDTree(node_xy)

    distances, indices = tree.query(gps_xy, k=1)

    result = track.copy()

    result["nearest_node_id"] = indices
    result["nearest_node_distance_m"] = distances

    print("\nNearest-node distance statistics:")
    print(pd.Series(distances).describe())

    print("\nGPS points with nearest nodes:")
    print(result.head(10))

    return result


# ==========================================================
# 11. SAVE ANALYSIS OUTPUTS
# ==========================================================

def save_outputs(arcs_result, gps_result, nearest_result):

    arcs_result.to_csv(
        OUTPUT_DIR / "road_arc_lengths.csv",
        index=False
    )

    gps_result.to_csv(
        OUTPUT_DIR / "gps_movement_analysis.csv",
        index=False
    )

    nearest_result.to_csv(
        OUTPUT_DIR / "gps_nearest_nodes.csv",
        index=False
    )

    print("\n========== SAVED FILES ==========")

    print(OUTPUT_DIR / "road_arc_lengths.csv")
    print(OUTPUT_DIR / "gps_movement_analysis.csv")
    print(OUTPUT_DIR / "gps_nearest_nodes.csv")


# ==========================================================
# 12. MAIN
# ==========================================================

def main():

    print("\nPATHSNAP NETWORK ANALYSIS")
    print("Trip:", TRIP_ID)

    track, nodes, arcs, route = load_trip()

    inspect_sizes(track, nodes, arcs, route)

    validate_network(nodes, arcs, route)

    arcs_result = calculate_road_lengths(
        nodes, arcs
    )

    analyze_connectivity(nodes, arcs)

    analyze_route(arcs, route)

    gps_result = analyze_gps(track)

    nearest_result = nearest_road_nodes(
        track, nodes
    )

    save_outputs(
        arcs_result,
        gps_result,
        nearest_result
    )

    print("\nNETWORK ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
