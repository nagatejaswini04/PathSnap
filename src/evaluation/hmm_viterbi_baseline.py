
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

# Accept trip ID from command line
TRIP_ID = sys.argv[1] if len(sys.argv) > 1 else "00000000"

TRIP_DIR = (
    ROOT / "data" / "processed" / "trips"
    / "student_test" / TRIP_ID
)

FEATURE_DIR = (
    ROOT / "data" / "processed" / "hmm"
    / "student_test"
)

MODEL_PATH = ROOT / "models" / "histgradientboosting.joblib"

OUTPUT_DIR = ROOT / "reports" / "hmm_viterbi"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "distance_to_arc_m",
    "projection_fraction",
    "candidate_rank",
    "gps_speed_mps",
    "gps_heading_deg",
    "road_length_m",
    "road_bearing_deg",
    "heading_difference_deg",
]

MAX_NETWORK_DISTANCE_M = 1500.0
MIN_SIGMA_M = 25.0


# ============================================================
# 2. HAVERSINE DISTANCE
# ============================================================

def haversine_m(lat1, lon1, lat2, lon2):

    earth_radius = 6371000.0

    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))

    dlat = lat2 - lat1

    dlon = np.radians(
        np.asarray(lon2, dtype=float)
        - np.asarray(lon1, dtype=float)
    )

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )


# ============================================================
# 3. BUILD DIRECTED ROAD GRAPH
# ============================================================

def build_road_graph(nodes, arcs):

    nodes = nodes.drop_duplicates(
        "node_id"
    ).reset_index(drop=True)

    node_ids = nodes["node_id"].to_numpy(dtype=int)

    node_lookup = {
        node_id: index
        for index, node_id in enumerate(node_ids)
    }

    arcs = arcs.drop_duplicates("arc_id").copy()

    from_indices = arcs["from_node_id"].map(node_lookup)
    to_indices = arcs["to_node_id"].map(node_lookup)

    if from_indices.isna().any() or to_indices.isna().any():
        raise ValueError("Road network contains unknown node IDs")

    coordinates = nodes.set_index("node_id")

    start_lon = arcs["from_node_id"].map(
        coordinates["longitude"]
    )

    start_lat = arcs["from_node_id"].map(
        coordinates["latitude"]
    )

    end_lon = arcs["to_node_id"].map(
        coordinates["longitude"]
    )

    end_lat = arcs["to_node_id"].map(
        coordinates["latitude"]
    )

    lengths = haversine_m(
        start_lat,
        start_lon,
        end_lat,
        end_lon
    )

    graph = coo_matrix(
        (
            lengths,
            (
                from_indices.to_numpy(dtype=int),
                to_indices.to_numpy(dtype=int)
            )
        ),
        shape=(len(nodes), len(nodes))
    ).tocsr()

    graph.sum_duplicates()

    return graph, node_lookup


# ============================================================
# 4. CALCULATE EMISSION PROBABILITIES
# ============================================================

def calculate_emissions(model, candidates):

    candidates = candidates.copy()

    probabilities = model.predict_proba(
        candidates[FEATURES]
    )

    positive_index = list(model.classes_).index(1)

    candidates["positive_probability"] = (
        probabilities[:, positive_index]
    )

    emissions = []
    gps_groups = []

    for _, group in candidates.groupby(
        "gps_index", sort=True
    ):

        group = group.reset_index(drop=True)

        gps_groups.append(group)

        values = group[
            "positive_probability"
        ].to_numpy(dtype=float)

        total = values.sum()

        if total <= 0:
            values = np.ones(len(values)) / len(values)
        else:
            values = values / total

        emissions.append(
            np.log(np.clip(values, 1e-12, 1.0))
        )

    return candidates, gps_groups, emissions


# ============================================================
# 5. CALCULATE TRANSITION SCORES
# ============================================================

def calculate_transition(
    previous,
    current,
    shortest_paths,
    source_lookup,
    node_lookup,
    gps_displacement
):

    previous_arc = previous["arc_id"].to_numpy(dtype=int)
    current_arc = current["arc_id"].to_numpy(dtype=int)

    previous_to = previous["to_node_id"].to_numpy(dtype=int)
    current_from = current["from_node_id"].to_numpy(dtype=int)

    previous_fraction = previous[
        "projection_fraction"
    ].to_numpy(dtype=float)

    current_fraction = current[
        "projection_fraction"
    ].to_numpy(dtype=float)

    previous_length = previous[
        "road_length_m"
    ].to_numpy(dtype=float)

    current_length = current[
        "road_length_m"
    ].to_numpy(dtype=float)

    previous_rows = np.array([
        source_lookup[node]
        for node in previous_to
    ])

    current_columns = np.array([
        node_lookup[node]
        for node in current_from
    ])

    network_distance = shortest_paths[
        previous_rows
    ][:, current_columns]

    remaining_distance = (
        (1.0 - previous_fraction) * previous_length
    )

    next_distance = (
        current_fraction * current_length
    )

    route_distance = (
        remaining_distance[:, None]
        + network_distance
        + next_distance[None, :]
    )

    valid = (
        np.isfinite(network_distance)
        & (network_distance <= MAX_NETWORK_DISTANCE_M)
    )

    # Handle consecutive observations on the same road arc
    same_arc = (
        previous_arc[:, None]
        == current_arc[None, :]
    )

    same_arc_distance = (
        current_fraction[None, :]
        - previous_fraction[:, None]
    ) * previous_length[:, None]

    same_arc_valid = (
        same_arc
        & (same_arc_distance >= -5.0)
    )

    same_arc_distance = np.maximum(
        same_arc_distance, 0.0
    )

    route_distance[same_arc_valid] = (
        same_arc_distance[same_arc_valid]
    )

    valid[same_arc_valid] = True

    sigma = max(
        MIN_SIGMA_M,
        gps_displacement * 0.5 + 15.0
    )

    transition = -0.5 * (
        (route_distance - gps_displacement) / sigma
    ) ** 2

    maximum_distance = max(
        1000.0,
        gps_displacement * 5.0 + 100.0
    )

    valid = (
        valid
        & (route_distance <= maximum_distance)
    )

    transition[~valid] = -np.inf

    return transition


# ============================================================
# 6. VITERBI DECODING
# ============================================================

def viterbi_decode(emissions, transition_matrices):

    n_steps = len(emissions)

    delta = [None] * n_steps
    backpointer = [None] * n_steps

    segment_starts = [0]

    delta[0] = emissions[0].copy()

    backpointer[0] = np.full(
        len(emissions[0]), -1, dtype=int
    )

    restart_count = 0

    # Forward pass
    for t in range(1, n_steps):

        transition = transition_matrices[t - 1]

        scores = (
            delta[t - 1][:, None]
            + transition
        )

        best_previous = np.argmax(scores, axis=0)

        best_scores = np.max(scores, axis=0)

        new_delta = best_scores + emissions[t]

        if not np.isfinite(new_delta).any():

            restart_count += 1
            segment_starts.append(t)

            print(
                f"WARNING: No feasible transition at step {t}. "
                "Starting a new segment."
            )

            delta[t] = emissions[t].copy()

            backpointer[t] = np.full(
                len(emissions[t]), -1, dtype=int
            )

        else:

            delta[t] = new_delta
            backpointer[t] = best_previous

    # Backtracking
    segment_starts.append(n_steps)

    states = np.zeros(n_steps, dtype=int)

    for segment_number in range(
        len(segment_starts) - 1
    ):

        start = segment_starts[segment_number]
        end = segment_starts[segment_number + 1]

        states[end - 1] = int(
            np.argmax(delta[end - 1])
        )

        for t in range(end - 1, start, -1):

            states[t - 1] = backpointer[t][
                states[t]
            ]

    return states, restart_count


# ============================================================
# 7. ROUTE CONTINUITY
# ============================================================

def calculate_continuity(route):

    if len(route) < 2:
        return 1.0

    previous_arc = route[
        "predicted_arc_id"
    ].to_numpy()[:-1]

    next_arc = route[
        "predicted_arc_id"
    ].to_numpy()[1:]

    previous_to = route[
        "to_node_id"
    ].to_numpy()[:-1]

    next_from = route[
        "from_node_id"
    ].to_numpy()[1:]

    connected = (
        (previous_arc == next_arc)
        | (previous_to == next_from)
    )

    return float(connected.mean())


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PATHSNAP - HMM AND VITERBI ROUTE RECONSTRUCTION")
    print("=" * 65)

    print("Trip ID:", TRIP_ID)

    # Validate trip files
    if not TRIP_DIR.exists():
        raise FileNotFoundError(
            f"Trip directory not found: {TRIP_DIR}"
        )

    feature_file = (
        FEATURE_DIR / f"{TRIP_ID}_hmm_features.csv"
    )

    if not feature_file.exists():
        raise FileNotFoundError(
            f"HMM feature file not found: {feature_file}"
        )

    # Load files
    candidates = pd.read_csv(feature_file)

    track = pd.read_csv(TRIP_DIR / "track.csv")
    nodes = pd.read_csv(TRIP_DIR / "nodes.csv")
    arcs = pd.read_csv(TRIP_DIR / "arcs.csv")

    model = joblib.load(MODEL_PATH)

    print("\nLoaded model and trip data.")

    if model.n_features_in_ != len(FEATURES):
        raise ValueError("Model feature count mismatch")

    candidates = candidates.sort_values(
        ["gps_index", "candidate_rank"]
    ).reset_index(drop=True)

    track = track.sort_values(
        "gps_index"
    ).reset_index(drop=True)

    print("GPS observations:", len(track))
    print("Candidate rows:", len(candidates))

    # Validate observations
    candidate_gps = sorted(
        candidates["gps_index"].unique()
    )

    track_gps = sorted(
        track["gps_index"].unique()
    )

    if candidate_gps != track_gps:
        raise ValueError(
            "Candidate GPS observations do not match track"
        )

    # Build directed road graph
    print("\nBuilding directed road graph...")

    graph, node_lookup = build_road_graph(
        nodes, arcs
    )

    source_nodes = sorted(
        candidates["to_node_id"].unique().astype(int)
    )

    source_indices = np.array([
        node_lookup[node]
        for node in source_nodes
    ])

    source_lookup = {
        node: index
        for index, node in enumerate(source_nodes)
    }

    print("Unique source nodes:", len(source_nodes))

    # Shortest paths
    print("\nCalculating road-network shortest paths...")

    shortest_paths = dijkstra(
        graph,
        directed=True,
        indices=source_indices,
        limit=MAX_NETWORK_DISTANCE_M
    )

    print("Shortest-path matrix:", shortest_paths.shape)

    # Emissions
    print("\nCalculating emission probabilities...")

    candidates, gps_groups, emissions = calculate_emissions(
        model, candidates
    )

    print("Emission groups:", len(emissions))

    # Transitions
    print("\nCalculating transition probabilities...")

    transition_matrices = []

    for t in range(1, len(gps_groups)):

        previous = gps_groups[t - 1]
        current = gps_groups[t]

        gps_displacement = float(
            haversine_m(
                track.loc[t - 1, "latitude"],
                track.loc[t - 1, "longitude"],
                track.loc[t, "latitude"],
                track.loc[t, "longitude"]
            )
        )

        transition = calculate_transition(
            previous,
            current,
            shortest_paths,
            source_lookup,
            node_lookup,
            gps_displacement
        )

        transition_matrices.append(transition)

        if t % 500 == 0:
            print("Processed transitions:", t)

    # Viterbi
    print("\nRunning Viterbi decoding...")

    selected_states, restart_count = viterbi_decode(
        emissions,
        transition_matrices
    )

    # Reconstruct predictions
    results = []

    for t, state in enumerate(selected_states):

        selected = gps_groups[t].iloc[state]

        results.append({
            "trip_id": TRIP_ID,
            "gps_index": int(selected["gps_index"]),
            "predicted_arc_id": int(selected["arc_id"]),
            "positive_probability": float(
                selected["positive_probability"]
            ),
            "from_node_id": int(selected["from_node_id"]),
            "to_node_id": int(selected["to_node_id"]),
            "candidate_rank": int(selected["candidate_rank"]),
        })

    result = pd.DataFrame(results)

    # Compress consecutive duplicate arcs
    route_sequence = result[
        result["predicted_arc_id"].ne(
            result["predicted_arc_id"].shift()
        )
    ].copy()

    route_sequence = route_sequence.reset_index(drop=True)

    continuity = calculate_continuity(route_sequence)

    # Save outputs
    prediction_path = (
        OUTPUT_DIR / f"{TRIP_ID}_viterbi_predictions.csv"
    )

    route_path = (
        OUTPUT_DIR / f"{TRIP_ID}_reconstructed_route.csv"
    )

    result.to_csv(prediction_path, index=False)

    route_sequence.to_csv(route_path, index=False)

    # Summary
    print("\n" + "=" * 65)
    print("VITERBI RECONSTRUCTION SUMMARY")
    print("=" * 65)

    print("Trip:", TRIP_ID)
    print("GPS observations:", len(result))
    print("Selected candidate states:", len(result))
    print("Reconstructed route arcs:", len(route_sequence))
    print("Viterbi segment restarts:", restart_count)
    print("Direct route continuity:", round(continuity, 4))

    print("\nPrediction file:")
    print(prediction_path)

    print("\nReconstructed route:")
    print(route_path)

    print("\nROUTE RECONSTRUCTION COMPLETED")


if __name__ == "__main__":
    main()
