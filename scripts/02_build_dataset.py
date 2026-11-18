"""Build candidate-level datasets for train and student_test.

data/processed/<part>_cands.parquet : one row per (observation, candidate arc)
data/processed/<part>_obs.parquet   : one row per observation (label + reliability)
results/candidates/candidate_stats.csv
"""
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from pathsnap.features import build_rows
from pathsnap.io import list_trips, load_trip
from pathsnap.network import TripContext

RAW, OUT = "data/raw", "data/processed"
LABEL_OK_M = 50.0  # observations farther than this from their aligned route arc are unreliable
os.makedirs(OUT, exist_ok=True); os.makedirs("results/candidates", exist_ok=True)


def work(args):
    part, tid = args
    t0 = time.time()
    ctx = TripContext(load_trip(f"{RAW}/{part}", tid))
    lab, pos, dl = ctx.point_labels()
    cands = ctx.candidates()
    df = build_rows(ctx, cands, lab)
    ok = dl <= LABEL_OK_M
    df["label_ok"] = ok[df.obs.values]
    in_c = np.zeros(len(lab), bool)
    in_c[df.obs[df.y == 1].values] = True
    obs = pd.DataFrame({"trip_id": tid, "obs": np.arange(len(lab)), "t": ctx.t, "lon": ctx.track[:, 0],
                        "lat": ctx.track[:, 1], "ref_arc": lab, "ref_pos": pos, "ref_dist": dl,
                        "label_ok": ok, "n_cand": [len(c) for c in cands], "ref_in_cands": in_c})
    return df, obs, time.time() - t0


if __name__ == "__main__":
    stats = []
    for part in ["train", "student_test"]:
        jobs = [(part, t) for t in list_trips(f"{RAW}/{part}")]
        with Pool(2) as p:
            res = p.map(work, jobs)
        cands = pd.concat([r[0] for r in res], ignore_index=True)
        obs = pd.concat([r[1] for r in res], ignore_index=True)
        cands.to_parquet(f"{OUT}/{part}_cands.parquet")
        obs.to_parquet(f"{OUT}/{part}_obs.parquet")
        o = obs[obs.label_ok]
        stats.append(dict(partition=part, trips=len(jobs), observations=len(obs),
                          reliable_obs=int(obs.label_ok.sum()), reliable_frac=obs.label_ok.mean(),
                          candidate_rows=len(cands), mean_cands=obs.n_cand.mean(),
                          obs_without_cands=int((obs.n_cand == 0).sum()),
                          candidate_miss_rate_reliable=1 - o.ref_in_cands.mean(),
                          candidate_miss_rate_all=1 - obs.ref_in_cands.mean(),
                          positive_rate=cands.y.mean(), build_seconds=sum(r[2] for r in res)))
        print(stats[-1])
    pd.DataFrame(stats).to_csv("results/candidates/candidate_stats.csv", index=False)
