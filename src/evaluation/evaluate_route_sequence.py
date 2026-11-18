
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TRIP_IDS = [
    "00000000",
    "00000007",
    "00000009",
    "00000040",
    "00000052",
    "00000056",
    "00000065",
    "00000076",
    "00000077",
    "00000091",
]

PREDICTION_DIR = ROOT / "reports/hmm_viterbi"

REFERENCE_DIR = (
    ROOT / "data/processed/trips/student_test"
)

OUTPUT_DIR = ROOT / "reports/hmm_viterbi"

OUTPUT_FILE = OUTPUT_DIR / "all_trips_route_metrics.csv"


def longest_common_subsequence(a, b):

    dp = [
        [0] * (len(b) + 1)
        for _ in range(len(a) + 1)
    ]

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):

            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1

            else:
                dp[i][j] = max(
                    dp[i - 1][j],
                    dp[i][j - 1]
                )

    return dp[-1][-1]


def evaluate_trip(trip_id):

    prediction_file = (
        PREDICTION_DIR
        / f"{trip_id}_reconstructed_route.csv"
    )

    reference_file = (
        REFERENCE_DIR
        / trip_id
        / "route.csv"
    )

    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Prediction file missing: {prediction_file}"
        )

    if not reference_file.exists():
        raise FileNotFoundError(
            f"Reference file missing: {reference_file}"
        )

    predicted = pd.read_csv(prediction_file)
    reference = pd.read_csv(reference_file)

    predicted_sequence = (
        predicted["predicted_arc_id"]
        .dropna()
        .astype(int)
        .tolist()
    )

    reference_sequence = (
        reference.sort_values("route_position")["arc_id"]
        .dropna()
        .astype(int)
        .tolist()
    )

    lcs_length = longest_common_subsequence(
        predicted_sequence,
        reference_sequence
    )

    precision = (
        lcs_length / len(predicted_sequence)
        if predicted_sequence else 0
    )

    recall = (
        lcs_length / len(reference_sequence)
        if reference_sequence else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    return {
        "trip_id": trip_id,
        "predicted_arcs": len(predicted_sequence),
        "reference_arcs": len(reference_sequence),
        "lcs_length": lcs_length,
        "sequence_precision": precision,
        "sequence_recall": recall,
        "sequence_f1": f1,
    }


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []

    print("=" * 65)
    print("PATHSNAP - ALL TRIPS ROUTE SEQUENCE EVALUATION")
    print("=" * 65)

    for trip_id in TRIP_IDS:

        try:
            metrics = evaluate_trip(trip_id)
            results.append(metrics)

            print(
                f"{trip_id} | "
                f"Precision: {metrics['sequence_precision']:.4f} | "
                f"Recall: {metrics['sequence_recall']:.4f} | "
                f"F1: {metrics['sequence_f1']:.4f}"
            )

        except Exception as error:

            failures.append({
                "trip_id": trip_id,
                "error": str(error)
            })

            print(f"{trip_id} | FAILED: {error}")

    if results:

        df = pd.DataFrame(results)

        df.to_csv(OUTPUT_FILE, index=False)

        print("\n" + "=" * 65)
        print("OVERALL ROUTE SEQUENCE RESULTS")
        print("=" * 65)

        print(
            "Mean Sequence Precision:",
            round(df["sequence_precision"].mean(), 4)
        )

        print(
            "Mean Sequence Recall:",
            round(df["sequence_recall"].mean(), 4)
        )

        print(
            "Mean Sequence F1:",
            round(df["sequence_f1"].mean(), 4)
        )

        print("\nResults saved to:")
        print(OUTPUT_FILE)

    if failures:

        failure_file = (
            OUTPUT_DIR / "route_sequence_failures.csv"
        )

        pd.DataFrame(failures).to_csv(
            failure_file,
            index=False
        )

        print("\nFailed trips:", len(failures))
        print("Failure report:", failure_file)

    print("\nCompleted:", len(results), "/", len(TRIP_IDS))


if __name__ == "__main__":
    main()
