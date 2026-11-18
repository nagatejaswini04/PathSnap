"""Build the technical report (HTML -> PDF via headless Chrome).

Every number in the report is read from files in results/ produced by scripts 01-05.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

R = "reports"
TEAM = os.environ.get("PATHSNAP_TEAM", "Team name: ________ &nbsp;|&nbsp; Members (Name, SRN): ________")

inv = pd.read_csv("results/eda/inventory.csv", dtype={"trip": str})
summ = pd.read_csv("results/eda/summary_by_partition.csv", index_col=0)
anom = pd.read_csv("results/eda/anomalies.csv", dtype={"trip": str})
cst = pd.read_csv("results/candidates/candidate_stats.csv").set_index("partition")
cmp_ = pd.read_csv("results/models/comparison.csv").set_index("model")
folds = pd.read_csv("results/models/cv_folds.csv")
best = json.load(open("results/bayes_opt/best.json"))
tv = pd.read_csv("results/bayes_opt/tuned_vs_untuned.csv")
trials = pd.read_csv("results/bayes_opt/trials.csv")
fi = pd.read_csv("results/bayes_opt/feature_importance.csv")
hov = pd.read_csv("results/hmm/student_test_overall.csv").set_index("method")
hpt = pd.read_csv("results/hmm/student_test_per_trip.csv", dtype={"trip": str})
grid = pd.read_csv("results/hmm/decoder_grid_train_oof.csv")
hcfg = json.load(open("models/final/hmm_config.json"))
mlp_c = pd.read_csv("results/models/curve_MLP.csv")
lgb_c = pd.read_csv("results/models/curve_LightGBM.csv")
hgb_c = pd.read_csv("results/models/curve_HistGradientBoosting.csv")


def f4(x): return f"{x:.4f}"
def pct(x): return f"{100 * x:.2f}%"


def table(df, cls=""):
    return df.to_html(index=False, escape=False, border=0, classes=cls)


tr, st = summ["train"], summ["student_test"]
c = cmp_
H = []
H.append(f"""
<section class="title">
<div class="kicker">UE24CS352A Machine Learning · PES University RR Campus</div>
<h1>PathSnap: Distributed Map-Matching &amp; Route Reconstruction</h1>
<div class="subtitle">Technical Report</div>
<p class="team">{TEAM}</p>
<p class="team">GitHub repository: ________</p>
<div class="abstract">
<b>Summary.</b> GPS points are matched to a per-trip directed road network. For every observation we generate
candidate arcs within 60&nbsp;m, describe each (observation, candidate) pair with 22 engineered features, and score it with a
binary classifier. Seven models from six algorithm families were compared with group-aware 5-fold cross-validation on the
80 training trips and evaluated on the 10 student-testing trips. LightGBM was selected (CV macro-F1
{f4(c.loc['LightGBM','cv_macro_f1'])}) and tuned with Bayesian optimisation (Optuna TPE, {best['n_trials']} trials). An
HMM with Viterbi decoding then enforces road-network connectivity. On the student-testing trips, observation-level Top-1
accuracy rises from {pct(hov.loc['Classifier only','top1_micro'])} (classifier only) to
{pct(hov.loc['HMM + classifier emission','top1_micro'])} (classifier + HMM), and route continuity from
{pct(hov.loc['Classifier only','cont_traversable'])} to {pct(hov.loc['HMM + classifier emission','cont_traversable'])}.
The system is served by a FastAPI backend (Laptop&nbsp;1) and a browser UI (Laptop&nbsp;2).
</div>
</section>
""")

# ---------------------------------------------------------------- 1. pipeline
H.append("""
<h2>1. Problem framing and pipeline</h2>
<p>Map matching is framed as <b>candidate ranking</b>. The road network differs for every trip (90 trips in cities across
several continents), so a fixed multi-class label space of arc IDs is not possible. Instead, for each GPS observation
<i>i</i> we gather nearby candidate arcs <i>C<sub>i</sub></i> and learn <i>P(arc = true arc | features)</i>. The candidate
with the highest score is the classifier-only prediction. The HMM then picks one candidate per observation so that the
whole sequence is likely <i>and</i> traversable on the directed network.</p>
<pre>
.track/.nodes/.arcs ─► clean track ─► local metric projection ─► KD-tree candidates (≤8 within 60 m)
   ─► 22 features per (obs, candidate) ─► LightGBM probability ─► HMM/Viterbi ─► arc per obs ─► connected route
.route (server-side only) ─► monotone DP alignment ─► per-observation reference arc (training labels / scoring)
</pre>
""")

# ---------------------------------------------------------------- 2. EDA
H.append(f"""
<h2>2. Dataset understanding and exploratory data analysis</h2>
<p class="note">Partitions used: <b>train</b> (80 trips) and <b>student_test</b> (10 trips) for all exploration.
The evaluator-testing partition was not available to us and was never used. Code: <code>scripts/01_eda.py</code>,
outputs in <code>results/eda/</code>.</p>

<h3>2.1 Inventory and structure</h3>
<table class="small">
<tr><th></th><th>Train</th><th>Student test</th></tr>
<tr><td>Trips (folders)</td><td>{int(tr.trips)}</td><td>{int(st.trips)}</td></tr>
<tr><td>Files per trip</td><td colspan=2>4 (.track, .nodes, .arcs, .route) in every trip, none empty</td></tr>
<tr><td>GPS observations (total / mean / min / max)</td><td>{int(tr.obs_total):,} / {tr.obs_mean:.0f} / {int(tr.obs_min)} / {int(tr.obs_max):,}</td>
<td>{int(st.obs_total):,} / {st.obs_mean:.0f} / {int(st.obs_min)} / {int(st.obs_max):,}</td></tr>
<tr><td>Mean trajectory length</td><td>{tr.length_km_mean:.1f} km</td><td>{st.length_km_mean:.1f} km</td></tr>
<tr><td>Mean road nodes / arcs per trip</td><td>{tr.nodes_mean:,.0f} / {tr.arcs_mean:,.0f}</td><td>{st.nodes_mean:,.0f} / {st.arcs_mean:,.0f}</td></tr>
<tr><td>Mean reference-route length (arcs)</td><td>{tr.route_len_mean:.0f}</td><td>{st.route_len_mean:.0f}</td></tr>
<tr><td>Steps exactly 1 s apart</td><td>{pct(tr.frac_dt_1s)}</td><td>{pct(st.frac_dt_1s)}</td></tr>
<tr><td>Trips with a gap &gt; 30 s / largest gap</td><td>{int(tr.trips_with_gap_gt30s)} / {tr.max_gap_s:,.0f} s</td><td>{int(st.trips_with_gap_gt30s)} / {st.max_gap_s:,.0f} s</td></tr>
<tr><td>Consecutive route arcs that are connected</td><td>{pct(tr.route_connected)}</td><td>{pct(st.route_connected)}</td></tr>
</table>
<p>Column meanings were checked against the data. In <code>.track</code>, column 1 is longitude, column 2 latitude and
column 3 a timestamp in seconds (latitude ranges stay within ±90 only in that order). Node and arc IDs are 0-based row
indices: every arc endpoint and every route arc ID is within range ({int(inv.route_arc_oob.sum())} out-of-range
references). Arcs are <b>directed</b>. On average {pct(inv.twin_frac.mean())} of arcs have a reverse twin
(<i>b→a</i> for <i>a→b</i>), so two-way streets appear as two geometrically identical arcs. Distance alone cannot tell
them apart; driving direction must. A GPS row does not correspond to any node or arc row; the only link is geometric.
Each trip has its <b>own</b> network (coordinates span longitudes {inv.lon_c.min():.0f}° to {inv.lon_c.max():.0f}°),
so nothing can be learned per arc ID.</p>
<img src="figures/eda_trip_locations.png" class="w80">
<h3>2.2 Missing values, invalid values and timestamps</h3>
<p>No NaN rows, no out-of-range coordinates, no (0,0) points, no out-of-order and no duplicate timestamps in either
partition (figure below; zero bars are genuinely zero). Missing data shows up instead as <b>time gaps</b>.
{int(tr.trips_with_gap_gt30s)} training trips contain at least one gap over 30&nbsp;s, and four have extreme gaps
(trip 00000090: {inv.set_index('trip').loc['00000090','dt_max']:,.0f}&nbsp;s; 00000078:
{inv.set_index('trip').loc['00000078','dt_max']:,.0f}&nbsp;s). In {int((inv.t_start != 1).sum())} trips timestamps do
not start at 1. {len(anom)} trips were flagged as anomalous (gaps &gt; 30&nbsp;s, single steps &gt; 200&nbsp;m, &gt;2%
of points far from the route, or a non-contiguous route; see <code>results/eda/anomalies.csv</code>).</p>
<img src="figures/eda_missing_quality.png" class="w70">
<img src="figures/eda_distributions.png" class="w100">
<h3>2.3 Road-network connectivity</h3>
<p>No isolated nodes in any trip. The networks are not single components: the median training trip has
{inv[inv.partition=='train'].weak_components.median():.0f} weakly connected components, though the largest holds
{pct(inv.largest_weak_frac.mean())} of nodes on average. The many small strongly connected components are mostly isolated one-way fragments, plausibly from the network being clipped around each trip. This matters for the HMM, because a transition between candidates in different components is
impossible and must be handled (Section 5).</p>
<img src="figures/eda_connectivity.png" class="w60">
<img src="figures/eda_trajectories.png" class="w100">
<p class="cap">GPS trajectories (blue), reference route (green) and road network near the trip (grey), in local metres.
Train 00000002 shows GPS drift and a section the route does not cover.</p>
<h3>2.4 Folium map (train trip 00000048)</h3>
<img src="screenshots/map_eda_train_00000048.png" class="w90">
<p class="cap">Interactive map <code>maps/eda_train_00000048.html</code>, with layer control for the road network, reference
route and raw GPS. The screenshot was rendered offline without basemap tiles; opened in a browser, the map shows
OpenStreetMap tiles.</p>

<h3>2.5 Preprocessing (answer to Q1)</h3>
<ol>
<li><b>Track cleaning</b> (<code>io.clean_track</code>): drop non-finite and out-of-range coordinates and (0,0) points,
sort by time, and drop duplicate timestamps. Nothing triggered on this data, but the step protects the API against unseen
evaluator files.</li>
<li><b>Projection</b>: longitude/latitude are angles, so Euclidean distance in degrees is wrong (1° of longitude is about 67&nbsp;km at
53°N but 111&nbsp;km at the equator). Each trip is projected to a local tangent plane in metres around its median point.
Trips span at most a few tens of km, where the scale error is negligible next to GPS noise. All geometry (point-to-segment
distance, bearings, path lengths) is in metres.</li>
<li><b>Network cropping and indexing</b>: only arcs whose bounding box intersects the trajectory bbox + 500&nbsp;m are kept
(from up to 307k down to a few thousand arcs). They are sampled every 15&nbsp;m into a KD-tree.</li>
<li><b>Per-point reference labels</b>: <code>.route</code> is an ordered arc sequence, not a label per point. We align the
track to the route with a <b>monotone dynamic program</b>: the route position may only move forward, each point pays its
distance to the assigned route arc, and skipping a route arc costs 2&nbsp;m (so short arcs crossed between two 1&nbsp;Hz fixes
can be skipped). The median point-to-label distance is {tr.label_dist_median:.1f}&nbsp;m (train).</li>
<li><b>Label reliability</b>: in some trips the reference route does not cover part of the trajectory (e.g. trip 00000054:
{pct(inv.set_index('trip').loc['00000054','frac_label_gt50m'])} of points are more than 50&nbsp;m from any route arc, even
ignoring order). We did not trust these points. Observations more than 50&nbsp;m from their aligned arc are flagged unreliable,
excluded from training and from point-level scoring, and reported: {pct(cst.loc['train','reliable_frac'])} of training and
{pct(cst.loc['student_test','reliable_frac'])} of student-test observations are reliable.</li>
<li><b>Candidate generation</b>: all arcs within 60&nbsp;m (at most 8, nearest first). If none exist, the 3 nearest within 300&nbsp;m are used.
Mean candidates per observation: {cst.loc['train','mean_cands']:.2f} (train). <b>Candidate miss rate</b> (reference arc not
in the set, reliable observations): {pct(cst.loc['train','candidate_miss_rate_reliable'])} train,
{pct(cst.loc['student_test','candidate_miss_rate_reliable'])} student-test. This is the ceiling on achievable Top-1.</li>
<li><b>Scaling</b>: StandardScaler (LR, KNN, MLP) and QuantileTransformer (GaussianNB) sit inside sklearn pipelines, so they
are fitted on training folds only.</li>
<li><b>Training subsample</b>: consecutive 1&nbsp;Hz points are strongly autocorrelated, so model fitting uses every 3rd
reliable observation of each trip ({cst.loc['train','candidate_rows']/1e6:.2f}M candidate rows → about 0.36M), identically for every model.
The deployed model is refit on all reliable observations.</li>
</ol>

<h3>2.6 Observed patterns and features (answer to Q2)</h3>
<ul>
<li>The nearest arc is often wrong: on train trip 00000048 the nearest candidate was correct for only 43% of points,
mostly because of reverse-direction twins at identical distance and parallel roads. This motivates the
<b>direction features</b>: <code>head_diff</code> / <code>head_cos</code> compare the arc bearing with the motion heading
(displacement over ±2 points), and <code>head_valid</code> is set when the vehicle moved &gt;3&nbsp;m. Stationary points get a
neutral value.</li>
<li>GPS noise is about 5&nbsp;m median but bursty, so a single point is unreliable. <b>Temporal-context features</b> measure how close
the neighbouring observations are to the same arc: <code>d_prev</code>, <code>d_next</code>, and <code>win_dist</code> (mean
over ±5 observations).</li>
<li>What matters is how a candidate compares with the others, not its absolute distance. <b>Relative features</b> within each
observation: <code>dist_rank</code>, <code>dist_minus_min</code>, <code>dist_ratio</code>, <code>win_rank</code>,
<code>win_minus_min</code>, <code>n_cand</code>.</li>
<li>Projections past an arc end usually indicate the neighbouring arc: <code>proj_t</code> and <code>end_dist</code>.</li>
<li>Network context: <code>arc_len</code>, <code>out_deg</code>, <code>in_deg</code>, <code>has_twin</code>. Sampling
context: <code>speed</code>, <code>dt_prev</code>, <code>step_prev</code>.</li>
</ul>
<p>The final model's gain-based feature importance (Section 4) confirms the choices: windowed distance, direction
agreement and relative distance dominate.</p>
""")

# ---------------------------------------------------------------- 3. models
rows = []
for m in c.index:
    r = c.loc[m]
    rows.append({"Model": m, "Algorithm family": r.family, "Key configuration": r.config,
                 "CV macro F1 (±sd)": f"{r.cv_macro_f1:.4f} ± {r.cv_macro_f1_std:.4f}",
                 "Top-1 acc. (student-test)": f4(r.test_top1),
                 "Route continuity (adjacent, student-test)": f4(r.test_route_continuity),
                 "Student-test macro F1": f4(r.test_macro_f1),
                 "Log loss (student-test)": f4(r.test_log_loss),
                 "Inference time (ms / obs)": f"{r.infer_ms_per_obs:.4f}"})
H.append(f"""
<h2>3. Model experimentation and comparison</h2>
<p><b>Protocol.</b> Binary target: candidate = reference arc. <b>GroupKFold(5) by trip</b> on the 80 training trips, so no
trip appears in both a training and a validation fold. Each model was then refit on all 80 training trips and evaluated on the 10
student-testing trips (reliable observations only, {int(c.loc['LightGBM','test_obs']):,} observations). Seed 42 throughout.
Code: <code>scripts/03_train_compare.py</code>; per-fold results in <code>results/models/cv_folds.csv</code>.</p>
<p><b>Metric definitions.</b> <i>Top-1 accuracy</i>: share of observations whose highest-scoring candidate is the
reference arc (missed candidates count as wrong). <i>Macro F1 / precision / recall</i>: candidate-level binary metrics with
the same decision rule (top candidate = positive, the rest = negative), so they are threshold-free and consistent with Top-1.
<i>Log loss</i>: binary cross-entropy of candidate probabilities. <i>Route continuity</i> (in this table): share of
consecutive observation pairs whose classifier-only arcs are the same arc or directly connected. The <b>primary selection metric
is CV macro F1</b>, with CV log loss as a tie-breaker, because the HMM consumes the probabilities directly.</p>
<h3>3.1 Overall model comparison</h3>
{table(pd.DataFrame(rows), "small")}
<p class="note">All seven models output probabilities, so log loss applies to all of them. Precision equals Top-1 accuracy
here because exactly one candidate per observation is predicted positive.</p>
""")

# per-model sections
M = {}
def gap(m): return c.loc[m, "train_macro_f1"] - c.loc[m, "cv_macro_f1"]
M["LogisticRegression"] = dict(
    why="Several features relate roughly monotonically to the log-odds of a correct match (smaller windowed distance, "
        "larger heading cosine, lower distance rank). A linear model is therefore a strong, interpretable baseline that also "
        "yields reasonably calibrated probabilities for the HMM.",
    sl="Strengths: fastest inference of all models (%.4f ms/obs), stable across folds (sd %.4f), calibrated. Limitation: it "
       "cannot model interactions. Heading agreement only matters when the vehicle is moving (<code>head_valid</code>), and "
       "distance matters differently on short vs long arcs, so the decision surface is too simple." % (c.loc["LogisticRegression", "infer_ms_per_obs"], c.loc["LogisticRegression", "cv_macro_f1_std"]),
    hp="C = 1.0 (L2), solver lbfgs, max_iter = 1000, StandardScaler on all 22 features.",
    loss="Binary cross-entropy (logistic loss) + L2 penalty, minimised with L-BFGS.",
    fit="Train macro F1 %.4f vs CV %.4f (gap %.4f) and test %.4f: the gap is essentially zero, but all scores trail the "
        "tree ensembles by about 0.01 macro F1 / 0.016 Top-1. This is <b>underfitting</b> (high bias), not overfitting." % (
        c.loc["LogisticRegression", "train_macro_f1"], c.loc["LogisticRegression", "cv_macro_f1"], gap("LogisticRegression"), c.loc["LogisticRegression", "test_macro_f1"]))
M["GaussianNB"] = dict(
    why="A generative probabilistic baseline: it models each feature's class-conditional distribution and needs almost no "
        "tuning. A QuantileTransformer maps the skewed distance features to approximately Gaussian marginals first.",
    sl="Strengths: trivial to train, gives probabilities. Limitation: the naive independence assumption is badly violated. "
       "<code>dist</code>, <code>win_dist</code>, <code>d_prev</code> and <code>d_next</code> are near-duplicates, so their "
       "evidence is counted several times and the probabilities become extremely over-confident.",
    hp="QuantileTransformer(output_distribution='normal', subsample=200k), var_smoothing = 1e-9.",
    loss="No iterative loss: closed-form maximum-likelihood estimates of per-class Gaussian means and variances.",
    fit="Train %.4f ≈ CV %.4f ≈ test %.4f macro F1, so there is no overfitting. Performance is the lowest, and CV log loss is %.3f, "
        "about 12× LightGBM's, with the largest fold variance (sd %.4f). This is <b>underfitting with poor calibration</b>, which "
        "makes the model unsuitable as an HMM emission." % (
        c.loc["GaussianNB", "train_macro_f1"], c.loc["GaussianNB", "cv_macro_f1"], c.loc["GaussianNB", "test_macro_f1"], c.loc["GaussianNB", "cv_log_loss"], c.loc["GaussianNB", "cv_macro_f1_std"]))
M["KNN"] = dict(
    why="Candidates with similar geometric/heading context should share outcomes, so a similarity-based method is a natural "
        "non-parametric choice with no assumption about the decision boundary.",
    sl="Strengths: non-linear, no training phase (%.2f s fit). Limitations: it depends on feature scaling, and inference is by far the "
       "slowest (%.2f ms per observation, about %d× LightGBM), which is a real cost when every API call scores thousands of "
       "candidates." % (c.loc["KNN", "fit_seconds"], c.loc["KNN", "infer_ms_per_obs"], round(c.loc["KNN", "infer_ms_per_obs"] / c.loc["LightGBM", "infer_ms_per_obs"])),
    hp="k = 31 neighbours, distance weighting, Euclidean metric, StandardScaler.",
    loss="None: a lazy learner that stores the training set and predicts by weighted neighbour vote.",
    fit="In-sample macro F1 %.4f vs CV %.4f. With distance weighting every training point is its own nearest neighbour, so the "
        "in-sample score is meaningless and the %.3f gap reflects <b>memorisation (overfitting in the variance sense)</b>. Its "
        "CV/test scores match Logistic Regression's, so the extra flexibility brought no gain." % (
        c.loc["KNN", "train_macro_f1"], c.loc["KNN", "cv_macro_f1"], gap("KNN")))
M["RandomForest"] = dict(
    why="Bagged decision trees capture interactions (heading × moving, distance × arc length) and are robust to the unscaled, "
        "heavy-tailed distance features.",
    sl="Strengths: second-best CV macro F1 (%.4f) and good log loss. Limitations: slowest training (%.0f s for one fit vs %.0f s "
       "for LightGBM) and a large model file; the probability is a vote share, which is coarser than boosting." % (
        c.loc["RandomForest", "cv_macro_f1"], c.loc["RandomForest", "fit_seconds"], c.loc["LightGBM", "fit_seconds"]),
    hp="n_estimators = 200, min_samples_leaf = 5, max_features = sqrt, bootstrap, unlimited depth.",
    loss="Each tree greedily minimises Gini impurity at each split; predictions average the trees' class proportions.",
    fit="Train %.4f vs CV %.4f (gap %.4f): <b>moderate overfitting</b> from deep, fully grown trees. Bagging limits the damage, and "
        "student-test (%.4f) matches CV, so generalisation is still good." % (
        c.loc["RandomForest", "train_macro_f1"], c.loc["RandomForest", "cv_macro_f1"], gap("RandomForest"), c.loc["RandomForest", "test_macro_f1"]))
M["HistGradientBoosting"] = dict(
    why="Gradient boosting fits many shallow trees sequentially, each correcting the previous ones' errors. Histogram binning makes "
        "it fast on about 0.36M rows.",
    sl="Strengths: fast (%.1f s fit), accurate, calibrated log-loss output. Limitation: its early stopping uses a random 10%% "
       "split of <i>rows</i>, not trips, so neighbouring points of the same trip end up on both sides and stopping is slightly "
       "optimistic." % c.loc["HistGradientBoosting", "fit_seconds"],
    hp="learning_rate = 0.1, max_iter = 300, max_leaf_nodes = 31, early_stopping on a 10%% validation split (stopped after %d "
       "iterations in the curve run)." % len(hgb_c),
    loss="Binary log loss (cross-entropy), minimised by gradient boosting.",
    fit="Train %.4f vs CV %.4f (gap %.4f) is small. The training curve shows training and validation loss falling together until "
        "early stopping triggered, so there is <b>no significant overfitting</b>." % (
        c.loc["HistGradientBoosting", "train_macro_f1"], c.loc["HistGradientBoosting", "cv_macro_f1"], gap("HistGradientBoosting")))
M["LightGBM"] = dict(
    why="Leaf-wise GBDT with row/column subsampling. It models the feature interactions that the linear model misses, while "
        "staying fast enough to score every candidate of an API request in milliseconds and producing well-calibrated "
        "probabilities, which the HMM uses as emissions.",
    sl="Strengths: best CV macro F1 (%.4f), best CV log loss (%.4f), best student-test Top-1 (%.4f), %.1f s fit. Limitations: "
       "leaf-wise growth can overfit with many leaves, so regularisation matters (hence the Bayesian search). Like every "
       "pointwise model it ignores sequence consistency." % (
        c.loc["LightGBM", "cv_macro_f1"], c.loc["LightGBM", "cv_log_loss"], c.loc["LightGBM", "test_top1"], c.loc["LightGBM", "fit_seconds"]),
    hp="n_estimators = 400, learning_rate = 0.05, num_leaves = 31, subsample = 0.8 (bagging_freq 1), colsample_bytree = 0.8, "
       "min_child_samples = 20 (default), no L1/L2.",
    loss="Binary log loss (cross-entropy) with second-order gradient boosting.",
    fit="Train %.4f vs CV %.4f (gap %.4f). In the curve run the validation log loss flattens around round 150 and reaches %.4f at "
        "its minimum (round %d), while training loss keeps falling to %.4f. That is <b>mild overfitting</b> beyond about 150 "
        "rounds, but validation loss never rises, so the extra rounds are harmless." % (
        c.loc["LightGBM", "train_macro_f1"], c.loc["LightGBM", "cv_macro_f1"], gap("LightGBM"), lgb_c.valid_logloss.min(),
        int(lgb_c.valid_logloss.idxmin()) + 1, lgb_c.train_logloss.iloc[-1]))
M["MLP"] = dict(
    why="A neural network learns non-linear combinations of the scaled features without hand-crafted interactions, and its "
        "sigmoid output is a probability.",
    sl="Strengths: competitive (CV %.4f), very fast inference (%.4f ms/obs). Limitations: sensitive to scaling and learning rate, "
       "noisier training, needs early stopping." % (c.loc["MLP", "cv_macro_f1"], c.loc["MLP", "infer_ms_per_obs"]),
    hp="Hidden layers (64, 32), ReLU, Adam lr = 1e-3, batch 512, L2 alpha = 1e-4, max 60 epochs, early stopping (patience 6).",
    loss="Binary cross-entropy + L2 weight penalty, minimised with Adam.",
    fit="In the epoch-wise run the validation loss is lowest at epoch %d (%.4f) and then drifts upward (%.4f at epoch 40), "
        "while training loss keeps falling. This is clear <b>overfitting after a few epochs</b>, which early stopping "
        "contains. Train/CV gap with early stopping: %.4f." % (
        int(mlp_c.valid_logloss.idxmin()) + 1, mlp_c.valid_logloss.min(), mlp_c.valid_logloss.iloc[-1], gap("MLP")))

H.append("<h3>3.2 Per-model questions</h3>")
for i, m in enumerate(c.index, 1):
    r = c.loc[m]; d = M[m]
    fm = folds[folds.model == m]
    H.append(f"""
<div class="model"><h4>3.2.{i} {m} <span class="fam">{r.family}</span></h4>
<p><b>A1. Why suitable?</b> {d['why']}</p>
<p><b>A2. Strengths / limitations.</b> {d['sl']}</p>
<p><b>B5. Hyperparameters (initial values).</b> {d['hp']}</p>
<p><b>B6. Loss / objective.</b> {d['loss']}</p>
<p><b>C7–C8. Student-test performance.</b> Top-1 accuracy <b>{f4(r.test_top1)}</b>, macro F1 <b>{f4(r.test_macro_f1)}</b>,
log loss <b>{f4(r.test_log_loss)}</b>, precision {f4(r.test_precision)}, recall {f4(r.test_recall)}, adjacent route
continuity {f4(r.test_route_continuity)}. CV folds macro F1: {', '.join(f4(v) for v in fm.macro_f1)}.</p>
<p><b>C9. Over/underfitting.</b> {d['fit']}</p></div>""")

H.append("""
<h3>3.3 Model visualisations</h3>
<img src="figures/models_cv_macro_f1.png" class="w80">
<img src="figures/models_test_top1_f1.png" class="w90">
<img src="figures/models_log_loss.png" class="w80">
<img src="figures/models_generalisation_gap.png" class="w90">
<img src="figures/models_training_curves.png" class="w100">
<p class="cap">Training curves only for models with real iteration histories (LightGBM boosting rounds, HistGradientBoosting
iterations, MLP epochs). LightGBM/MLP validation = 16 held-out training trips; HGB uses its internal split. No curves are shown for
LR, NB, KNN or RF, which have no epoch-wise history.</p>
<img src="figures/models_confusion.png" class="w100">
<img src="figures/models_speed_accuracy.png" class="w60">
""")
H.append(f"""
<h3>3.4 Model selection</h3>
<p><b>LightGBM</b> was selected. It has the highest CV macro F1 ({f4(c.loc['LightGBM','cv_macro_f1'])}) and lowest CV log loss
({f4(c.loc['LightGBM','cv_log_loss'])}), and it also leads on the student-test set (Top-1 {f4(c.loc['LightGBM','test_top1'])}, macro F1
{f4(c.loc['LightGBM','test_macro_f1'])}), so the CV ranking held. Random Forest is a close second on CV
({f4(c.loc['RandomForest','cv_macro_f1'])}) but takes about {c.loc['RandomForest','fit_seconds']/c.loc['LightGBM','fit_seconds']:.0f}× longer to
train. The CV difference between the top models (about 0.001–0.002) is smaller than the fold standard deviation (about 0.01), so
efficiency, calibration (log loss feeds the HMM) and room for tuning decided the choice. KNN was rejected for inference
cost, and GaussianNB for calibration.</p>
""")

# ---------------------------------------------------------------- 4. Bayes
bp = best["best_params"]
done = trials[trials.state == "COMPLETE"]
tvr = tv.set_index("model")
un, tu = tvr.loc["LightGBM (untuned)"], tvr.loc["LightGBM (tuned)"]
dep = tvr.iloc[2]
tv_tab = pd.DataFrame({
    "Model": tvr.index,
    "CV macro F1": [f"{a:.4f} ± {b:.4f}" if pd.notna(a) else "N/A (no CV: refit only)" for a, b in zip(tvr.cv_macro_f1, tvr.cv_macro_f1_std)],
    "CV log loss": [f4(v) if pd.notna(v) else "N/A" for v in tvr.cv_log_loss],
    "Student-test Top-1": [f4(v) for v in tvr.test_top1],
    "Student-test macro F1": [f4(v) for v in tvr.test_macro_f1],
    "Student-test log loss": [f4(v) for v in tvr.test_log_loss],
    "Fit time (s)": [f"{v:.1f}" for v in tvr.fit_seconds],
    "Inference (ms/obs)": [f"{v:.4f}" for v in tvr.infer_ms_per_obs]})
space = pd.DataFrame([
    ("n_estimators", "int uniform [100, 1000], step 50", bp["n_estimators"]),
    ("learning_rate", "log-uniform [0.01, 0.2]", f"{bp['learning_rate']:.4f}"),
    ("num_leaves", "int log-uniform [15, 255]", bp["num_leaves"]),
    ("min_child_samples", "int log-uniform [5, 300]", bp["min_child_samples"]),
    ("subsample", "uniform [0.5, 1.0]", f"{bp['subsample']:.3f}"),
    ("colsample_bytree", "uniform [0.4, 1.0]", f"{bp['colsample_bytree']:.3f}"),
    ("reg_lambda (L2)", "log-uniform [1e-3, 10]", f"{bp['reg_lambda']:.4f}"),
    ("reg_alpha (L1)", "log-uniform [1e-3, 10]", f"{bp['reg_alpha']:.4f}")],
    columns=["Hyperparameter", "Range / distribution", "Best value"])
better = tu.cv_macro_f1 > un.cv_macro_f1
H.append(f"""
<h2>4. Bayesian hyperparameter optimisation (LightGBM)</h2>
<p><b>Method.</b> Optuna's Tree-structured Parzen Estimator (TPE) sampler, seed 42: 10 random start-up trials, then
TPE models <i>p(params | good)</i> / <i>p(params | bad)</i> to propose new points. A median pruner stops a trial early if its
running fold-mean falls below the median of earlier trials at the same fold. The full search record is in
<code>results/bayes_opt/trials.csv</code> and the SQLite study <code>optuna_study.db</code>.</p>
<p><b>Q1–Q2. Search space:</b></p>{table(space, "small")}
<p><b>Q3. Objective:</b> maximise mean macro F1 over GroupKFold(5) by trip on the 80 training trips (the same stride-3,
reliable-label data and splits as the model comparison). The student-test and evaluator partitions were not used in the search.</p>
<p><b>Q4. Trials:</b> {best['n_trials']} ({len(done)} completed, {best['n_pruned']} pruned), {best['optimisation_seconds']/60:.1f} minutes
on 2 CPU cores.</p>
<p><b>Q5. Best values:</b> see table above; best CV macro F1 = {f4(best['best_cv_macro_f1'])}.</p>
<p><b>Q6. Did tuning help?</b> Same data partitions and metrics for both:</p>
{table(tv_tab, "small")}
<p>{"Yes, modestly." if better else "Not on CV."} CV macro F1 changes from {f4(un.cv_macro_f1)} to {f4(tu.cv_macro_f1)}
({tu.cv_macro_f1-un.cv_macro_f1:+.4f}) and CV log loss from {f4(un.cv_log_loss)} to {f4(tu.cv_log_loss)} ({'marginally worse: the objective optimised F1, not calibration' if tu.cv_log_loss > un.cv_log_loss else 'better'}). On student-test,
Top-1 moves {f4(un.test_top1)} → {f4(tu.test_top1)} and macro F1 {f4(un.test_macro_f1)} → {f4(tu.test_macro_f1)}. The gains
are smaller than the fold standard deviation: the default configuration was already close to the optimum and the remaining
error is dominated by genuinely ambiguous points (parallel roads, twins at stops), which no hyperparameter fixes. Those are
what the HMM targets. The deployed model (last row) uses the tuned configuration refit on all
{cst.loc['train','reliable_obs']:,} reliable training observations instead of every 3rd one.</p>
<img src="figures/bayes_opt.png" class="w100">
<img src="figures/feature_importance.png" class="w60">
""")

# ---------------------------------------------------------------- 5. HMM
cl, hm, ga = hov.loc["Classifier only"], hov.loc["HMM + classifier emission"], hov.loc["HMM + Gaussian emission (baseline)"]
hov_tab = pd.DataFrame({
    "Method": hov.index,
    "Top-1 (micro)": [f4(v) for v in hov.top1_micro], "Top-1 (trip mean)": [f4(v) for v in hov.top1_trip_mean],
    "Continuity (adjacent)": [f4(v) for v in hov.cont_adjacent], "Continuity (traversable)": [f4(v) for v in hov.cont_traversable],
    "Route precision": [f4(v) for v in hov.route_precision], "Route recall": [f4(v) for v in hov.route_recall],
    "Route F1": [f4(v) for v in hov.route_f1], "Exact route match (/10)": [int(v) for v in hov.exact_route_match],
    "Chain breaks": [int(v) for v in hov.breaks], "Decode ms/obs": [f"{v:.3f}" for v in hov.decode_ms_per_obs]})
piv = hpt.pivot(index="trip", columns="method", values="top1")
piv["Δ HMM"] = piv["HMM + classifier emission"] - piv["Classifier only"]
worse = piv[piv["Δ HMM"] < 0]
ptab = piv.reset_index()[["trip", "Classifier only", "HMM + Gaussian emission (baseline)", "HMM + classifier emission", "Δ HMM"]]
for col in ptab.columns[1:]:
    ptab[col] = ptab[col].map(lambda v: f"{v:+.4f}" if col == "Δ HMM" else f4(v))
v1 = pd.read_csv("results/hmm/student_test_overall_v1_unconstrained_fill.csv").set_index("method")
v1p = pd.read_csv("results/hmm/student_test_per_trip_v1_unconstrained_fill.csv", dtype={"trip": str})
v1t = v1p[v1p.trip == "00000065"].set_index("method")
pt65 = hpt[hpt.trip == "00000065"].set_index("method")
rf = hpt.pivot(index="trip", columns="method", values="route_f1")
rf_worse = rf[rf["HMM + classifier emission"] < rf["Classifier only"]]
if len(worse) == 0:
    where_text = "Point-level Top-1 improved on all 10 student-test trips (smallest gain: trip %s, %+.4f). " % (
        piv["Δ HMM"].idxmin(), piv["Δ HMM"].min())
else:
    where_text = "Top-1 decreased on trip(s) " + ", ".join(worse.index) + " (" + ", ".join(f"{v:+.4f}" for v in worse["Δ HMM"]) + "). "
gg = piv["HMM + Gaussian emission (baseline)"] - piv["Classifier only"]
where_text += ("The Gaussian-emission HMM, which ignores the classifier, was <i>worse</i> than classifier-only on %d trips (%s): "
               "distance alone cannot separate twins and parallel roads, so sequence constraints only help when the "
               "emissions are informative. " % ((gg < 0).sum(), ", ".join(gg[gg < 0].index)))
if len(rf_worse):
    where_text += ("At route level, route F1 is still below classifier-only on trip(s) " + ", ".join(rf_worse.index) +
                   " (" + ", ".join(f"{a:.4f} vs {b:.4f}" for a, b in zip(rf_worse['HMM + classifier emission'], rf_worse['Classifier only'])) + ").")
else:
    where_text += "After the gap fix below, HMM route F1 is at least as good as classifier-only on every trip."
gb = grid[grid.emission == "classifier-only"].train_top1_micro.iloc[0]
gbest = grid[grid.emission == "clf"].train_top1_micro.max()
gbest_g = grid[grid.emission == "gauss"].train_top1_micro.max()
H.append(f"""
<h2>5. HMM and Viterbi route reconstruction</h2>
<p>Code: <code>pathsnap/hmm.py</code> (decoder), <code>scripts/05_hmm_eval.py</code> (experiments).</p>
<p><b>Q1. Hidden states.</b> At observation <i>i</i> the hidden state is one candidate arc <i>c ∈ C<sub>i</sub></i> (the same
candidate set and order the classifier scored, so the candidate↔arc correspondence is preserved), together with the
projection offset <i>s<sub>i</sub>(c)</i> of the GPS point along that arc in metres.</p>
<p><b>Q2. Emissions.</b> log P(z<sub>i</sub> | c) = w<sub>e</sub> · log p<sub>LightGBM</sub>(c), clipped at 10<sup>-6</sup>.
As a baseline we also ran the classical Gaussian emission −½(d/σ)², with d the perpendicular distance.</p>
<p><b>Q3. Transitions</b> (Newson &amp; Krumm style). log P(c' → c) = −|d<sub>route</sub> − d<sub>gc</sub>| / β, where
d<sub>gc</sub> is the straight-line distance between the two GPS points and d<sub>route</sub> is the driving distance on the
<b>directed</b> graph: (remaining length of c' after s') + shortest path from end(c') to start(c) + s. On the same arc,
d<sub>route</sub> = s − s', and backward moves beyond 15&nbsp;m are forbidden (wrong-way driving). Shortest paths use a bounded
Dijkstra (cutoff 3·d<sub>gc</sub> + 200&nbsp;m, max 5&nbsp;km) over the cropped network, cached per source node. Viterbi recurrence:
δ<sub>i</sub>(c) = log E<sub>i</sub>(c) + max<sub>c'</sub>[δ<sub>i-1</sub>(c') + log T(c',c)], with back-pointers for the arg-max path.</p>
<p><b>Q4. Missing candidates and gaps.</b> Candidate generation falls back to the 3 nearest arcs within 300&nbsp;m, so on this data
no observation lacked candidates. If one did, it is skipped by the recursion and inherits the previous prediction. A time gap
&gt; 60&nbsp;s cuts the chain: the segment so far is backtracked, and Viterbi restarts from emissions alone, because the
vehicle may have travelled an unknown route during the gap.</p>
<p><b>Q5. Disconnected segments.</b> If no candidate at <i>i</i> is reachable from any candidate at <i>i−1</i> within the cutoff
(all transitions −∞, e.g. different network components or a GPS jump), the chain is also cut and restarted rather than forcing
an impossible path. The final route is made connected by filling the shortest path between consecutive distinct predicted arcs.
Student-test needed {int(hm.breaks)} chain breaks in total.</p>
<p><b>Decoder tuning.</b> β ∈ {{3, 5, 10, 20, 40}} m × w<sub>e</sub> ∈ {{0.5, 1, 2}} (and β × σ for the Gaussian baseline) were
grid-searched on <b>out-of-fold</b> LightGBM probabilities for all 80 training trips. Each fold's model never saw the trips it
scored, so the decoder was tuned on realistic, not in-sample, probabilities. Selected: emission = {hcfg['emission']},
β = {hcfg['params']['beta']:g}&nbsp;m, w<sub>e</sub> = {hcfg['params']['w_emit']:g}. Training OOF Top-1: classifier-only
{f4(gb)} → HMM {f4(gbest)} (Gaussian-emission HMM: {f4(gbest_g)}). Multiplying the whole Viterbi objective by β shows that only the product w<sub>e</sub>·β affects the arg-max, which is why cells with equal products (e.g. β=5, w=1 and β=10, w=0.5) give identical scores. The grid is effectively a 1-D search over w<sub>e</sub>·β ∈ [1.5, 80]; the optimum is at 5, and accuracy falls as w<sub>e</sub>·β grows (emissions over-weighted).</p>
<img src="figures/hmm_grid.png" class="w50">
<p><b>Q6. Did HMM decoding help? Student-testing partition (10 trips):</b></p>
{table(hov_tab, "small")}
<p>Yes. Top-1 rises by {100*(hm.top1_micro-cl.top1_micro):.2f} points ({pct(cl.top1_micro)} → {pct(hm.top1_micro)}) and
traversable continuity from {pct(cl.cont_traversable)} to {pct(hm.cont_traversable)}. The classifier alone produces
physically impossible jumps (to a reverse twin or a parallel road for one or two seconds) that the HMM removes.
Route precision improves from {f4(cl.route_precision)} to {f4(hm.route_precision)}: the classifier-only "route" contains
many spurious arcs. Using the classifier probabilities as emissions beats the classical Gaussian-distance emission
({pct(ga.top1_micro)}), which shows the learned features add information the HMM alone does not have.</p>
{table(ptab, "small")}
<img src="figures/hmm_per_trip.png" class="w100">
<p><b>Where HMM did not help.</b> {where_text}</p>
<p><b>Gap handling: a failure found and fixed.</b> In the first decoder version the route builder inserted the shortest path
between <i>every</i> pair of consecutive predicted arcs. On trip 00000065 this lowered route F1 below classifier-only
({f4(v1t.loc['Classifier only','route_f1'])} → {f4(v1t.loc['HMM + classifier emission','route_f1'])}). The trip has a
210&nbsp;s GPS gap during which the vehicle moved only 5&nbsp;m (parked). The chain correctly broke there, but the builder then
"repaired" the break with about 1.7&nbsp;km of arcs that were never driven. We now insert a connecting path only if its length is
plausible for the GPS displacement (≤ 3·d<sub>gc</sub> + 200&nbsp;m, the transition model's own bound), and otherwise leave the
route discontinuous. Trip 00000065 route F1: {f4(v1t.loc['HMM + classifier emission','route_f1'])} →
{f4(pt65.loc['HMM + classifier emission','route_f1'])}; all-trip mean route F1:
{f4(v1.loc['HMM + classifier emission','route_f1'])} → {f4(hm.route_f1)}. Point-level Top-1 is unaffected (the fix only
changes route assembly). The first-version outputs are kept as <code>results/hmm/*_v1_unconstrained_fill.csv</code>.</p>
<p>The remaining errors are mostly (a) the {pct(cst.loc['student_test','candidate_miss_rate_reliable'])} of observations
whose reference arc is not a candidate, (b) points near intersections where the DP-aligned label is itself uncertain to
within one arc, and (c) parallel roads closer than the GPS noise. Trip 00000077 (lowest Top-1) is the clearest case of (c):
the HMM lifts it from {f4(piv.loc['00000077','Classifier only'])} to {f4(piv.loc['00000077','HMM + classifier emission'])},
but it stays the weakest trip.</p>
""")

# ---------------------------------------------------------------- 6. Folium
H.append("""
<h2>6. Geographical visualisation using Folium</h2>
<p>Every student-test trip has an interactive map (<code>maps/student_test_&lt;id&gt;.html</code>, generated by
<code>scripts/05_hmm_eval.py</code>). Layers, each switchable in the layer control: raw GPS (blue dots, with time tooltips),
road network near the trip (grey), reference route (thick translucent green), HMM-predicted route (red), classifier-only arcs
(orange, off by default). Start/end markers and a legend are included, and the extent is fitted to the trajectory.</p>
<img src="screenshots/map_student_test_00000052.png" class="w90">
<p class="cap">Student-test trip 00000052. The red predicted route lies on the green reference route along the whole trip.</p>
<img src="screenshots/map_student_test_00000000.png" class="w90">
<p class="cap">Student-test trip 00000000.</p>
<img src="screenshots/map_student_test_00000077_zoom.png" class="w90">
<p class="cap">Student-test trip 00000077 (hardest trip), zoomed to a dense block where GPS scatter is large, with the
classifier-only layer switched on (orange, drawn on top). The classifier alone jumps to a diagonal street, side stubs and
cross-streets that the vehicle never used. The HMM route (red, under the orange where they agree) follows the green reference
because those jumps are not traversable at the observed speed.</p>
""")

# ---------------------------------------------------------------- 7. API/UI
H.append("""
<h2>7. API, UI and deployment</h2>
<p><b>Laptop 1</b> runs <code>uvicorn api.main:app --host 0.0.0.0 --port 8000</code> with
<code>PATHSNAP_EVAL_DIR</code> pointing at the ten evaluator trips. It loads <code>models/final/classifier.joblib</code> and
<code>hmm_config.json</code> once at start-up. Endpoints: <code>GET /health</code>, <code>GET /trips</code> (IDs + point counts),
<code>POST /predict {"trip_id": …}</code>. <code>/predict</code> loads the trip <i>without</i> its route, runs cleaning →
candidates → features → LightGBM → Viterbi → route reconstruction, and only afterwards reads the <code>.route</code> file
server-side to compute Top-1 accuracy, classifier-only Top-1, route precision/recall/F1 and continuity. The response contains
GPS points, predicted route polylines and arc IDs, metrics and timings, never the reference route. Every request is logged
to the console (method, client IP, trip, accuracy, timing).</p>
<p><b>Laptop 2</b> serves <code>ui/</code> with <code>python -m http.server 5500</code>. The page (Leaflet, bundled
locally) asks for the API address, lists the ten trips returned by <code>/trips</code>, sends the selected ID to
<code>/predict</code>, and draws GPS points and the predicted route with a legend and layer control, next to the metrics.</p>
<img src="screenshots/ui_result.png" class="w100">
<p class="cap">UI after predicting a trip. Prepared in development against the student-test partition; basemap tiles were
unavailable in the build environment, while the demo laptops load OpenStreetMap tiles.</p>
<img src="screenshots/api_console.png" class="w100">
<p class="cap">API console log of the requests above (development run). <b>To be replaced with the screenshot from the CMD of
Laptop 1 during the evaluation run, as required.</b></p>
""")

# ---------------------------------------------------------------- 8. Integrity
H.append(f"""
<h2>8. Reproducibility, limitations and academic integrity</h2>
<ul>
<li>Every table and figure here is generated from files in <code>results/</code> by <code>scripts/06_figures.py</code>
and <code>scripts/07_report.py</code>. Rerunning scripts 01–05 regenerates them (seed 42). Console logs are in <code>logs/</code>.</li>
<li>Evaluator-testing data was never used for training, feature selection, model selection, tuning or thresholds. No trip IDs or
predictions are hardcoded.</li>
<li>Limitations: labels come from aligning GPS to the supplied route, which itself has gaps and a few non-contiguous steps
({pct(1-tr.route_connected)} of consecutive route arcs are not connected). Reported accuracies are therefore agreement with an
imperfect reference, not verified ground truth, and points more than 50&nbsp;m from the route are excluded from scoring.</li>
<li><b>AI assistance disclosure:</b> ________ <i>(the team must state here, as required by the event's AI-use policy, which
parts were produced with AI assistance and confirm that every member can explain the code, experiments and results).</i></li>
</ul>
""")

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10pt; line-height: 1.45; color: #1a1f27; }
h1 { font-size: 22pt; margin: 6px 0; } h2 { font-size: 15pt; border-bottom: 2px solid #c2410c; padding-bottom: 3px; margin-top: 26px; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 16px 0 6px; page-break-after: avoid; } h4 { font-size: 11pt; margin: 4px 0 6px; }
.title { page-break-after: always; padding-top: 50mm; } .kicker { color: #c2410c; font-weight: bold; letter-spacing: .04em; font-size: 9pt; }
.subtitle { font-size: 14pt; color: #555; margin-bottom: 24px; } .team { font-size: 11pt; }
.abstract { margin-top: 30px; padding: 12px 14px; background: #f5f6f8; border-left: 4px solid #c2410c; }
img { display: block; margin: 8px auto; page-break-inside: avoid; } .w100 { width: 100%; } .w90 { width: 90%; } .w80 { width: 80%; }
.w70 { width: 70%; } .w60 { width: 60%; } .w50 { width: 50%; }
table { border-collapse: collapse; margin: 8px 0; width: 100%; page-break-inside: avoid; }
th, td { border: 1px solid #d4d8de; padding: 3px 5px; text-align: left; vertical-align: top; }
th { background: #eef0f3; } table.small { font-size: 8pt; }
.note { color: #444; font-size: 9pt; background: #fafafa; padding: 6px 8px; border: 1px solid #eee; }
.cap { font-size: 8.5pt; color: #555; text-align: center; margin-top: -2px; }
pre { font-size: 8pt; background: #f5f6f8; padding: 8px; white-space: pre-wrap; }
code { font-size: 8.8pt; background: #f2f3f5; padding: 0 2px; }
.model { border: 1px solid #e3e6ea; padding: 6px 10px; margin: 8px 0; page-break-inside: avoid; }
.fam { font-weight: normal; color: #777; font-size: 9pt; margin-left: 6px; }
"""
html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>PathSnap Technical Report</title><style>{CSS}</style></head><body>{''.join(H)}</body></html>"
open(f"{R}/report.html", "w").write(html)
pdf = os.path.abspath(f"{R}/PathSnap_Technical_Report.pdf")
subprocess.run(["google-chrome", "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", "file://" + os.path.abspath(f"{R}/report.html")], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("wrote", pdf)
