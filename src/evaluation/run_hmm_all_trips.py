
import subprocess
import sys
from pathlib import Path

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

SCRIPT = ROOT / "src" / "evaluation" / "hmm_viterbi.py"

def main():

    results = []

    print("=" * 65)
    print("PATHSNAP - BATCH HMM ROUTE RECONSTRUCTION")
    print("=" * 65)

    for trip_id in TRIP_IDS:

        print(f"\n{'=' * 65}")
        print("PROCESSING TRIP:", trip_id)
        print("=" * 65)

        process = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                trip_id
            ],
            cwd=ROOT,
            capture_output=True,
            text=True
        )

        print(process.stdout)

        if process.returncode == 0:

            results.append({
                "trip_id": trip_id,
                "status": "SUCCESS",
                "error": ""
            })

        else:

            print("ERROR:")
            print(process.stderr)

            results.append({
                "trip_id": trip_id,
                "status": "FAILED",
                "error": process.stderr[-1000:]
            })

    report_dir = ROOT / "reports" / "hmm_viterbi"
    report_dir.mkdir(parents=True, exist_ok=True)

    report = Path(
        report_dir / "batch_execution_summary.csv"
    )

    import pandas as pd

    pd.DataFrame(results).to_csv(
        report,
        index=False
    )

    print("\n" + "=" * 65)
    print("BATCH EXECUTION SUMMARY")
    print("=" * 65)

    print(pd.DataFrame(results).to_string(index=False))

    print("\nSummary saved:", report)


if __name__ == "__main__":
    main()
