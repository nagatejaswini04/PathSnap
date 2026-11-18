"""Dataset exploration (train + student_test partitions only).

Outputs -> results/eda/*.csv, reports/figures/eda_*.png, maps/eda_*.html
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.geo import haversine
from pathsnap.io import clean_track, list_trips, load_trip
from pathsnap.network import TripContext

RAW = "data/raw"
OUT = "results/eda"; FIG = "reports/figures"; MAPS = "maps"
for d in (OUT, FIG, MAPS):
    os.makedirs(d, exist_ok=True)

rows, gaps_all, dists_all = [], [], {}
for part in ["train", "student_test"]:
    for tid in list_trips(f"{RAW}/{part}"):
        p = f"{RAW}/{part}/{tid}/{tid}"
        sizes = {ext: os.path.getsize(p + "." + ext) if os.path.exists(p + "." + ext) else -1
                 for ext in ["track", "nodes", "arcs", "route"]}
        trip = load_trip(f"{RAW}/{part}", tid)
        tr = trip.track
        nan_rows = int((~np.isfinite(tr)).any(1).sum())
        _, cinfo = clean_track(tr)
        dt = np.diff(tr[:, 2])
        step = haversine(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
        V, E = len(trip.nodes), len(trip.arcs)
        A = coo_matrix((np.ones(E), (trip.arcs[:, 0], trip.arcs[:, 1])), shape=(V, V))
        n_weak, lab_w = connected_components(A, directed=True, connection="weak")
        n_strong, lab_s = connected_components(A, directed=True, connection="strong")
        deg = np.bincount(trip.arcs.ravel(), minlength=V)
        r = trip.route
        route_conn = np.mean(trip.arcs[r[:-1], 1] == trip.arcs[r[1:], 0]) if len(r) > 1 else np.nan
        ctx = TripContext(trip)
        _, _, dl = ctx.point_labels()
        dists_all[(part, tid)] = dl
        gaps_all.append(pd.DataFrame({"part": part, "trip": tid, "dt": dt}))
        rows.append(dict(
            partition=part, trip=tid, **{f"bytes_{k}": v for k, v in sizes.items()},
            n_obs=len(tr), nan_rows=nan_rows, invalid_rows=cinfo["n_invalid"],
            out_of_order=int((dt < 0).sum()), dup_time=int((dt == 0).sum()),
            duration_s=float(tr[-1, 2] - tr[0, 2]), t_start=float(tr[0, 2]),
            dt_median=float(np.median(dt)), frac_dt_1s=float(np.mean(np.isclose(dt, 1))),
            dt_max=float(dt.max()), n_gaps_gt5s=int((dt > 5).sum()), n_gaps_gt30s=int((dt > 30).sum()),
            max_step_m=float(step.max()), length_km=float(step.sum() / 1000),
            mean_speed_kmh=float(step.sum() / max(1, tr[-1, 2] - tr[0, 2]) * 3.6),
            lon_min=tr[:, 0].min(), lon_max=tr[:, 0].max(), lat_min=tr[:, 1].min(), lat_max=tr[:, 1].max(),
            lon_c=float(np.median(tr[:, 0])), lat_c=float(np.median(tr[:, 1])),
            n_nodes=V, n_arcs=E, n_active_arcs=len(ctx.active),
            weak_components=n_weak, largest_weak_frac=np.bincount(lab_w).max() / V,
            strong_components=n_strong, isolated_nodes=int((deg == 0).sum()),
            twin_frac=float(ctx.has_twin.mean()),
            route_len=len(r), route_unique=len(np.unique(r)), route_connected_frac=route_conn,
            route_arc_oob=int((r >= E).sum()),
            label_dist_median=float(np.median(dl)), label_dist_p95=float(np.percentile(dl, 95)),
            frac_label_gt50m=float(np.mean(dl > 50)),
        ))
        print(part, tid, len(tr), V, E)

inv = pd.DataFrame(rows)
inv.to_csv(f"{OUT}/inventory.csv", index=False)
gaps = pd.concat(gaps_all)

summ = inv.groupby("partition").agg(
    trips=("trip", "count"), obs_total=("n_obs", "sum"), obs_mean=("n_obs", "mean"),
    obs_min=("n_obs", "min"), obs_max=("n_obs", "max"), nan_rows=("nan_rows", "sum"),
    invalid_rows=("invalid_rows", "sum"), out_of_order=("out_of_order", "sum"), dup_time=("dup_time", "sum"),
    frac_dt_1s=("frac_dt_1s", "mean"), trips_with_gap_gt30s=("n_gaps_gt30s", lambda s: int((s > 0).sum())),
    max_gap_s=("dt_max", "max"), length_km_mean=("length_km", "mean"),
    nodes_mean=("n_nodes", "mean"), arcs_mean=("n_arcs", "mean"), route_len_mean=("route_len", "mean"),
    route_connected=("route_connected_frac", "mean"), label_dist_median=("label_dist_median", "median"),
).T
summ.to_csv(f"{OUT}/summary_by_partition.csv")
print(summ)

anom = inv[(inv.n_gaps_gt30s > 0) | (inv.max_step_m > 200) | (inv.frac_label_gt50m > 0.02) |
           (inv.route_connected_frac < 1)][["partition", "trip", "n_obs", "dt_max", "n_gaps_gt30s",
                                             "max_step_m", "frac_label_gt50m", "route_connected_frac"]]
anom.to_csv(f"{OUT}/anomalies.csv", index=False)

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
for part, c in [("train", "#2a6fdb"), ("student_test", "#e07b00")]:
    s = inv[inv.partition == part]
    ax[0, 0].hist(s.n_obs, bins=25, alpha=.6, color=c, label=part)
    ax[0, 1].hist(s.length_km, bins=25, alpha=.6, color=c, label=part)
    ax[0, 2].hist(np.log10(s.n_arcs), bins=25, alpha=.6, color=c, label=part)
    ax[1, 2].hist(s.mean_speed_kmh, bins=25, alpha=.6, color=c, label=part)
ax[0, 0].set(title="GPS observations per trip", xlabel="observations")
ax[0, 1].set(title="Trajectory length", xlabel="km")
ax[0, 2].set(title="Road-network size", xlabel="log10(#arcs)")
ax[1, 2].set(title="Mean speed per trip", xlabel="km/h")
g = gaps.dt.clip(upper=20)
ax[1, 0].hist(g, bins=np.arange(0, 21.5, 0.5), color="#555")
ax[1, 0].set_yscale("log"); ax[1, 0].set(title="Sampling interval (clipped at 20 s, both partitions)", xlabel="seconds")
alld = np.concatenate([v for (p, _), v in dists_all.items() if p == "train"])
ax[1, 1].hist(np.clip(alld, 0, 100), bins=50, color="#2a9d8f")
ax[1, 1].set_yscale("log"); ax[1, 1].set(title="Distance GPS point -> derived reference arc (train)", xlabel="metres (clipped 100)")
for a in ax.ravel():
    a.legend(fontsize=8) if a.get_legend_handles_labels()[0] else None
plt.tight_layout(); plt.savefig(f"{FIG}/eda_distributions.png", dpi=130); plt.close()

# geographic distribution of trips
plt.figure(figsize=(11, 5.2))
for part, c in [("train", "#2a6fdb"), ("student_test", "#e07b00")]:
    s = inv[inv.partition == part]
    plt.scatter(s.lon_c, s.lat_c, s=18, c=c, label=part, alpha=.8)
plt.xlim(-180, 180); plt.ylim(-60, 75); plt.grid(alpha=.3)
plt.xlabel("longitude"); plt.ylabel("latitude"); plt.title("Trip locations (median GPS coordinate)"); plt.legend()
plt.tight_layout(); plt.savefig(f"{FIG}/eda_trip_locations.png", dpi=130); plt.close()

# missing / quality bar chart
q = inv.groupby("partition")[["nan_rows", "invalid_rows", "out_of_order", "dup_time", "n_gaps_gt5s", "n_gaps_gt30s"]].sum()
q.T.plot.bar(figsize=(9, 4), color=["#2a6fdb", "#e07b00"])
plt.title("Missing / invalid values and timestamp issues (counts)"); plt.ylabel("count"); plt.xticks(rotation=20)
plt.tight_layout(); plt.savefig(f"{FIG}/eda_missing_quality.png", dpi=130); plt.close()

# trajectories + network for a few trips
sel = [("train", "00000048"), ("train", "00000002"), ("student_test", "00000052"), ("student_test", "00000065")]
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for a, (part, tid) in zip(axs, sel):
    trip = load_trip(f"{RAW}/{part}", tid); ctx = TripContext(trip)
    act = ctx.active
    for e in act[:20000]:
        a.plot([ctx.ax[e], ctx.bx[e]], [ctx.ay[e], ctx.by[e]], c="#bbb", lw=.5)
    r = trip.route
    for e in r:
        a.plot([ctx.ax[e], ctx.bx[e]], [ctx.ay[e], ctx.by[e]], c="#2a9d8f", lw=2.5)
    a.scatter(ctx.px, ctx.py, s=2, c="#1f4fd1", zorder=3)
    a.set_aspect("equal"); a.set_title(f"{part}/{tid}"); a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
plt.suptitle("GPS (blue), reference route (green), road network near trajectory (grey)")
plt.tight_layout(); plt.savefig(f"{FIG}/eda_trajectories.png", dpi=120); plt.close()

# network connectivity table plot
plt.figure(figsize=(7, 4))
plt.scatter(inv.n_arcs, inv.weak_components, c=(inv.partition == "train").map({True: "#2a6fdb", False: "#e07b00"}), s=14)
plt.xscale("log"); plt.xlabel("#arcs"); plt.ylabel("#weakly connected components"); plt.title("Road-network connectivity per trip")
plt.tight_layout(); plt.savefig(f"{FIG}/eda_connectivity.png", dpi=130); plt.close()

# --------------------------------------------------------------- folium map
part, tid = "train", "00000048"
trip = load_trip(f"{RAW}/{part}", tid); ctx = TripContext(trip)
m = folium.Map(location=[np.median(trip.track[:, 1]), np.median(trip.track[:, 0])], zoom_start=15, tiles="OpenStreetMap")
fg_net = folium.FeatureGroup("Road network (near trip)")
for e in ctx.active:
    a, b = trip.arcs[e]
    folium.PolyLine([(trip.nodes[a, 1], trip.nodes[a, 0]), (trip.nodes[b, 1], trip.nodes[b, 0])], color="#999", weight=1).add_to(fg_net)
fg_gps = folium.FeatureGroup("Raw GPS")
for lon, lat, t in trip.track[::2]:
    folium.CircleMarker((lat, lon), radius=2, color="#1f4fd1", fill=True, tooltip=f"t={t:.0f}").add_to(fg_gps)
fg_rt = folium.FeatureGroup("Reference route")
for e in trip.route:
    a, b = trip.arcs[e]
    folium.PolyLine([(trip.nodes[a, 1], trip.nodes[a, 0]), (trip.nodes[b, 1], trip.nodes[b, 0])], color="#16a34a", weight=5, opacity=.8).add_to(fg_rt)
for fg in (fg_net, fg_rt, fg_gps):
    fg.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds([[trip.track[:, 1].min(), trip.track[:, 0].min()], [trip.track[:, 1].max(), trip.track[:, 0].max()]])
m.save(f"{MAPS}/eda_train_00000048.html")
print("done")
