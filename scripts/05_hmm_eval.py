"""HMM / Viterbi experiments.

1. Out-of-fold (GroupKFold by trip) probabilities of the tuned classifier for all
   80 training trips -> decoder parameters are tuned on these (no in-sample bias,
   evaluator data never touched).
2. Grid over emission type / beta / emission weight / sigma; pick best mean
   reliable-observation Top-1 on training trips.
3. Student-test comparison: classifier-only vs HMM(classifier emission) vs
   HMM(Gaussian-distance emission baseline), per trip and overall.
4. Folium maps for every student-test trip.
"""
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import folium
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.features import FEATURES
from pathsnap.hmm import reconstruct_route, viterbi
from pathsnap.io import list_trips, load_trip
from pathsnap.metrics import continuity, route_set_metrics, traversable_continuity
from pathsnap.models import SEED
from pathsnap.network import TripContext

OUT = "results/hmm"; MAPS = "maps"
os.makedirs(OUT, exist_ok=True); os.makedirs(MAPS, exist_ok=True)
best = json.load(open("results/bayes_opt/best.json"))["best_params"]


def make():
    return lgb.LGBMClassifier(**best, subsample_freq=1, random_state=SEED, n_jobs=2, verbose=-1)


# ------------------------------------------------ 1. OOF probabilities (train)
tr_all = pd.read_parquet("data/processed/train_cands.parquet")
fit_rows = tr_all[tr_all.label_ok & (tr_all.obs % 3 == 0)]
trips = np.array(sorted(tr_all.trip_id.unique()))
oof = np.zeros(len(tr_all))
for k, (a, b) in enumerate(GroupKFold(5).split(trips, groups=trips)):
    fit_mask = fit_rows.trip_id.isin(trips[a])
    m = make().fit(fit_rows.loc[fit_mask, FEATURES].values, fit_rows.loc[fit_mask, "y"].values)
    sel = tr_all.trip_id.isin(trips[b]).values
    oof[sel] = m.predict_proba(tr_all.loc[sel, FEATURES].values)[:, 1]
    print("oof fold", k)
tr_all["p"] = oof

te_all = pd.read_parquet("data/processed/student_test_cands.parquet")
final = joblib.load("models/final/classifier.joblib")["model"]
te_all["p"] = final.predict_proba(te_all[FEATURES].values)[:, 1]


def prepare(part, df):
    """Per trip: context, candidate lists, probs, derived labels."""
    out = {}
    for tid, g in df.groupby("trip_id"):
        ctx = TripContext(load_trip(f"data/raw/{part}", tid))
        lab, _, dl = ctx.point_labels()
        N = len(lab)
        g = g.sort_values(["obs"], kind="stable")
        sizes = np.bincount(g.obs.values, minlength=N)
        splits = np.cumsum(sizes)[:-1]
        cands = np.split(g.arc.values, splits); probs = np.split(g.p.values, splits)
        out[tid] = (ctx, cands, probs, lab, dl <= 50)
    return out


def evaluate(data, method, params=None, with_route=False):
    rows = []
    for tid, (ctx, cands, probs, lab, ok) in data.items():
        t0 = time.time()
        clf = np.array([c[int(np.argmax(q))] if len(c) else -1 for c, q in zip(cands, probs)])
        if method == "classifier":
            pred, br = clf, 0
        else:
            pred, br = viterbi(ctx, cands, probs, params, emission=method)
        dt = time.time() - t0
        r = dict(trip=tid, n_obs=len(lab), n_scored=int(ok.sum()), correct=int((pred[ok] == lab[ok]).sum()),
                 top1=float(np.mean(pred[ok] == lab[ok])) if ok.any() else np.nan, breaks=br, decode_s=dt)
        if with_route:
            r["cont_adjacent"] = continuity(pred, ctx.arcs)
            r["cont_traversable"] = traversable_continuity(ctx, pred)
            route = reconstruct_route(ctx, pred) if method != "classifier" else \
                [int(a) for k, a in enumerate(pred) if k == 0 or a != pred[k - 1]]
            r.update(route_set_metrics(np.array(route, dtype=np.int64), ctx.route, ctx.alen))
            r["exact_route_match"] = float(list(dict.fromkeys(route)) == list(dict.fromkeys(ctx.route.tolist())))
            r["_pred"], r["_route"] = pred, route
        rows.append(r)
    return rows


train_data = prepare("train", tr_all)
# ------------------------------------------------ 2. decoder grid (train OOF)
grid = []
for beta, w in itertools.product([3, 5, 10, 20, 40], [0.5, 1.0, 2.0]):
    grid.append(("clf", dict(beta=beta, w_emit=w)))
for beta, s in itertools.product([5, 10, 20], [5, 10, 20]):
    grid.append(("gauss", dict(beta=beta, sigma=s)))
base = evaluate(train_data, "classifier")
gres = [dict(emission="classifier-only", beta=np.nan, w_emit=np.nan, sigma=np.nan,
             train_top1_micro=sum(r["correct"] for r in base) / sum(r["n_scored"] for r in base),
             train_top1_trip_mean=np.nanmean([r["top1"] for r in base]))]
for em, prm in grid:
    t0 = time.time()
    rr = evaluate(train_data, em, prm)
    gres.append(dict(emission=em, beta=prm["beta"], w_emit=prm.get("w_emit", np.nan), sigma=prm.get("sigma", np.nan),
                     train_top1_micro=sum(r["correct"] for r in rr) / sum(r["n_scored"] for r in rr),
                     train_top1_trip_mean=np.nanmean([r["top1"] for r in rr]), seconds=time.time() - t0))
    print(gres[-1])
G = pd.DataFrame(gres); G.to_csv(f"{OUT}/decoder_grid_train_oof.csv", index=False)
bc = G[G.emission == "clf"].sort_values("train_top1_micro", ascending=False).iloc[0]
bg = G[G.emission == "gauss"].sort_values("train_top1_micro", ascending=False).iloc[0]
clf_params = dict(beta=float(bc.beta), w_emit=float(bc.w_emit))
gauss_params = dict(beta=float(bg.beta), sigma=float(bg.sigma))
json.dump({"emission": "clf", "params": clf_params, "selected_on": "train OOF top-1 (80 trips)"},
          open("models/final/hmm_config.json", "w"), indent=2)

# ------------------------------------------------ 3. student-test comparison
test_data = prepare("student_test", te_all)
methods = {"Classifier only": ("classifier", None), "HMM + classifier emission": ("clf", clf_params),
           "HMM + Gaussian emission (baseline)": ("gauss", gauss_params)}
per_trip, overall, preds = [], [], {}
for name, (em, prm) in methods.items():
    rr = evaluate(test_data, em, prm, with_route=True)
    preds[name] = {r["trip"]: (r.pop("_pred"), r.pop("_route")) for r in rr}
    for r in rr:
        per_trip.append(dict(method=name, **r))
    d = pd.DataFrame(rr)
    overall.append(dict(method=name, top1_micro=d.correct.sum() / d.n_scored.sum(), top1_trip_mean=d.top1.mean(),
                        cont_adjacent=d.cont_adjacent.mean(), cont_traversable=d.cont_traversable.mean(),
                        route_precision=d.route_precision.mean(), route_recall=d.route_recall.mean(),
                        route_f1=d.route_f1.mean(), exact_route_match=d.exact_route_match.sum(),
                        breaks=d.breaks.sum(), decode_ms_per_obs=1000 * d.decode_s.sum() / d.n_obs.sum()))
    print(overall[-1])
pd.DataFrame(per_trip).to_csv(f"{OUT}/student_test_per_trip.csv", index=False)
pd.DataFrame(overall).to_csv(f"{OUT}/student_test_overall.csv", index=False)

# ------------------------------------------------ 4. Folium maps
def arc_line(ctx, e):
    a, b = ctx.arcs[e]
    return [(ctx.nodes_ll[a, 1], ctx.nodes_ll[a, 0]), (ctx.nodes_ll[b, 1], ctx.nodes_ll[b, 0])]


for tid, (ctx, cands, probs, lab, ok) in test_data.items():
    tr = ctx.track
    m = folium.Map(location=[np.median(tr[:, 1]), np.median(tr[:, 0])], zoom_start=15, tiles="OpenStreetMap")
    layers = [("Road network (near trip)", ctx.active[:15000], "#9ca3af", 1, .6, True),
              ("Reference route", ctx.route, "#16a34a", 8, .55, True),
              ("Classifier-only (per-point arcs)", np.unique(preds["Classifier only"][tid][0]), "#f97316", 3, .9, False),
              ("Predicted route (HMM)", preds["HMM + classifier emission"][tid][1], "#dc2626", 4, .9, True)]
    for nm, arcs_, col, wgt, op, show in layers:
        fg = folium.FeatureGroup(nm, show=show)
        for e in arcs_:
            folium.PolyLine(arc_line(ctx, e), color=col, weight=wgt, opacity=op).add_to(fg)
        fg.add_to(m)
    fg = folium.FeatureGroup("Raw GPS observations")
    step = max(1, len(tr) // 1500)
    for i in range(0, len(tr), step):
        folium.CircleMarker((tr[i, 1], tr[i, 0]), radius=2, color="#1d4ed8", fill=True, weight=1,
                            tooltip=f"obs {i}, t={tr[i, 2]:.0f}s").add_to(fg)
    fg.add_to(m)
    folium.Marker((tr[0, 1], tr[0, 0]), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((tr[-1, 1], tr[-1, 0]), tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = ('<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px 12px;'
              'border:1px solid #999;font:13px sans-serif">'
              f'<b>PathSnap - student_test {tid}</b><br>'
              '<span style="color:#1d4ed8">&#9679;</span> Raw GPS<br>'
              '<span style="color:#16a34a">&#9644;</span> Reference route<br>'
              '<span style="color:#dc2626">&#9644;</span> Predicted route (HMM)<br>'
              '<span style="color:#f97316">&#9644;</span> Classifier-only arcs<br>'
              '<span style="color:#9ca3af">&#9644;</span> Road network</div>')
    m.get_root().html.add_child(folium.Element(legend))
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    m.save(f"{MAPS}/student_test_{tid}.html")
print("done")
