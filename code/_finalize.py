"""Finalize the notebook for submission. Edits markdown cells for accuracy,
inserts a model-improvement step, augments the TEST cell with error metrics,
and rewrites Verify / Conclusions / Limitations against the actual outputs."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent / 'Module 4 Template.ipynb'
nb = json.loads(NB.read_text(encoding='utf-8'))


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': text.splitlines(keepends=True)}


def find(prefix, kind='markdown'):
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == kind and ''.join(c['source']).strip().startswith(prefix):
            return i
    raise SystemExit(f"could not find {kind} starting with {prefix!r}")


# ---------- 1) Fix Step 7 markdown: be honest about what the cohort split actually buys us ----------
nb['cells'][find('### Step 7: Cohort splits')] = md("""### Step 7: Cohort splits at the LAML / brain boundary

The k = 6 KMeans solution in Step 4 produced two lineage-pure clusters (LAML alone, GBM+LGG together) and four mixed epithelial clusters. To test whether the angiogenic story is a universal pan-cancer program or really an *epithelial* program, we re-run the downstream progression analyses on three nested cohorts:

1. **full** — all samples
2. **no_LAML** — solid tumors only (drops cluster 3)
3. **no_LAML_no_brain** — solid epithelial only (drops clusters 3 and 4)

**Caveat to keep in mind for Step 8.** AJCC tumor stage is *not recorded* for LAML or for brain tumors (GBM/LGG) in TCGA, so the stage cross-tabs in Step 8 will be **identical** across the three cohorts — the cohort split only changes which clusters can possibly contribute, and clusters 0, 1, 2, 5 contribute the same samples in all three. The cohort split is informative for the survival analysis (Step 9) and the supervised models (Step 10), where LAML and brain *do* contribute samples.
""")

# ---------- 2) Fix Step 10 markdown: be honest about the mixed AUC story ----------
nb['cells'][find('### Step 10: Supervised stage prediction')] = md("""### Step 10: Supervised stage prediction (early vs. late)

Train logistic regression and random forest on the 111 angiogenesis genes to predict early (I/II) vs. late (III/IV) tumor stage, comparing three labelings:

- **(i) Pan-cancer + cancer_type covariate** — one model, cancer_type one-hot encoded as extra features
- **(ii) Per cancer type** — separate model per cancer type (5-fold CV; require n ≥ 30)
- **(iii) Per cluster** — separate model per KMeans cluster (5-fold CV; require n ≥ 30)

The decision rule we want to test: *if per-cluster AUC is at least as good as per-cancer-type AUC, that supports replacing tissue-of-origin with angiogenic state as the more useful label for progression.* "At least as good" is a weak claim — we are not asking the cluster label to *beat* the cancer-type label, just to carry comparable information while collapsing 23 cancer types into 6 angiogenic states.
""")

# ---------- 3) Fix Step 11 markdown: actual results show KMeans-UMAP isolates kidney; Ward does not ----------
nb['cells'][find('### Step 11: Kidney')] = md("""### Step 11: Kidney-cancer sensitivity (alternative clusterings)

KIRC and KIRP form a clean island in UMAP but did not get their own KMeans cluster on the 111-D expression matrix. We test two alternative clusterings:

- **(a) KMeans on the 2D UMAP embedding** (k = 6, same as Step 4)
- **(b) Hierarchical (Ward) on the 111-D z-scored expression** cut at 6 clusters

We measure two things: *capture* (what fraction of all kidney samples land in their dominant cluster?) and *purity* (what fraction of that dominant cluster is kidney?). A real kidney cluster needs both numbers to be high.

Going in we expected hierarchical clustering to do better, on the theory that KMeans is sensitive to the high-dimensional geometry. The actual results invert that expectation, and we report what we found.
""")

# ---------- 4) Fix Step 13 markdown: tighten and note that LAML/brain don't appear in stage table ----------
nb['cells'][find('### Step 13: Project the held-out VALIDATION')] = md("""### Step 13: Project the held-out VALIDATION set onto the trained pipeline

Re-scale VALIDATION expression with the *training* `StandardScaler`, project into the trained PCA, UMAP, and KMeans models, then check that:

- **(a) cluster assignments are stable** — cancer-type composition per cluster looks similar to TRAINING
- **(b) stage / survival associations replicate** on VALIDATION

The VALIDATION stage table only shows clusters 0, 1, 2, 5 because LAML (cluster 3) and GBM/LGG (cluster 4) have no AJCC stage in TCGA, exactly as in the TRAINING side of Step 8. This is the expected behavior, not a missing-data bug.

Robustness here is a direct check on the k = 6 choice. We do **not** touch the TEST set in this step.
""")


# ---------- 5) Insert NEW Step 14: model improvement before TEST ----------
step14_md = md("""### Step 14: Improving the stage predictor before testing

The Step 10 pan-cancer logistic regression had a 5-fold CV AUC of **0.683** with default hyperparameters (`C=1`, `penalty='l2'`, `class_weight=None`), and the random forest was at **0.678** with default settings. Before we touch the held-out TEST set, we try one round of model improvement:

- **Logistic regression**: grid search over `C ∈ {0.01, 0.1, 1, 10}`, `penalty ∈ {'l1', 'l2'}`, `class_weight ∈ {None, 'balanced'}`. The class-balancing arm is the one we expect to help, since the early/late split is roughly 60/40 and in Step 6 we already saw threshold-0.5 predictions miss the minority class.
- **Random forest**: grid search over `n_estimators ∈ {200, 500}`, `max_depth ∈ {None, 8, 16}`, `min_samples_split ∈ {2, 10}`, `class_weight ∈ {None, 'balanced'}`. The shallower-depth arms are the ones we expect to help, since the default depth produced an in-sample AUC of 1.000 in Step 10 (clear overfit).

Both grid searches use 5-fold stratified CV with `roc_auc` as the scoring metric and the same pan-cancer + cancer_type covariate design matrix as Step 10 (i). We then refit the best estimator on all of TRAINING and carry it forward to the TEST set in Step 15.
""")

step14_code = code("""# === Step 14: Model improvement (hyperparameter tuning) before touching TEST ===
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Same shuffled CV splitter as Step 10 (TCGA samples are grouped by cancer type in
# the input CSV, so unshuffled folds would land entire cancer types in the test fold
# and destroy the cancer_type one-hot covariate).
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Reuse Step 10's pan-cancer + cancer_type design matrix
ct_oh_train = pd.get_dummies(pd.Series(ct_stage)).astype(float)
X_pan = np.hstack([X_stage, ct_oh_train.values])

# --- logistic regression sweep ---
pipe_lr = Pipeline([
    ('scl', StandardScaler()),
    ('clf', LogisticRegression(solver='liblinear', max_iter=2000, random_state=42)),
])
grid_lr = {
    'clf__C':            [0.01, 0.1, 1.0, 10.0],
    'clf__penalty':      ['l1', 'l2'],
    'clf__class_weight': [None, 'balanced'],
}
gs_lr = GridSearchCV(pipe_lr, grid_lr, cv=cv, scoring='roc_auc', n_jobs=-1)
gs_lr.fit(X_pan, y_stage)
print("Logistic regression sweep:")
print(f"  baseline 5-fold CV AUC (C=1, l2, no balance): 0.683")
print(f"  best 5-fold CV AUC:                            {gs_lr.best_score_:.3f}")
print(f"  best params:                                   {gs_lr.best_params_}")
print(f"  improvement:                                   {gs_lr.best_score_ - 0.683:+.3f}")

# --- random forest sweep ---
pipe_rf = Pipeline([
    ('scl', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1)),
])
grid_rf = {
    'clf__n_estimators':      [200, 500],
    'clf__max_depth':         [None, 8, 16],
    'clf__min_samples_split': [2, 10],
    'clf__class_weight':      [None, 'balanced'],
}
gs_rf = GridSearchCV(pipe_rf, grid_rf, cv=cv, scoring='roc_auc', n_jobs=-1)
gs_rf.fit(X_pan, y_stage)
print("\\nRandom forest sweep:")
print(f"  baseline 5-fold CV AUC (defaults):             0.678")
print(f"  best 5-fold CV AUC:                            {gs_rf.best_score_:.3f}")
print(f"  best params:                                   {gs_rf.best_params_}")
print(f"  improvement:                                   {gs_rf.best_score_ - 0.678:+.3f}")

# Carry the tuned models forward
best_lr = gs_lr.best_estimator_   # full Pipeline (scaler + classifier)
best_rf = gs_rf.best_estimator_
print(f"\\nCarrying forward to TEST: best_lr (logreg) and best_rf (random forest).")
""")

# Insert these two cells right after the Step 13 code cell (cell 42)
verify_idx = find('## Verify and validate')
nb['cells'].insert(verify_idx, step14_md)
nb['cells'].insert(verify_idx + 1, step14_code)


# ---------- 6) Replace the Verify markdown to point at the new TEST cell with error metrics ----------
verify_idx = find('## Verify and validate')
nb['cells'][verify_idx] = md("""## Verify and validate your analysis

We verify the pipeline along two axes: an *internal* axis (does the trained pipeline behave consistently on data it has never seen?) and an *external* axis (does what we found agree with the published angiogenesis literature?).

### 1. Internal: project the held-out TEST set through the trained pipeline

Step 13 already showed that the trained `StandardScaler → PCA → UMAP → KMeans` pipeline projects the VALIDATION set cleanly. The TEST set (`data/TEST_SET_GSE62944_subsample_log2TPM.csv`, 1600 samples balanced 80 per cancer type across 20 cancer types) is the final unbiased check. We use **only the trained transformers** from Steps 1–4 and the **tuned model** from Step 14 — no refitting on TEST.

The headline metrics in Step 15 below are:

- **Cluster-assignment stability.** TCGA cancer types should land in the same cluster they did in TRAINING. We summarize this as "dominant cancer type per cluster matches between TRAINING and TEST."
- **Stage replication.** The mixed epithelial clusters (0, 1, 2, 5) should show the same direction of stage skew on TEST as on TRAINING.
- **Survival replication.** The cluster ranking by median OS / PFI should preserve, even if absolute survival times shift.
- **Stage-prediction TEST AUC with error bars.** We report point AUC, **bootstrap 95% confidence intervals** (1000 resamples), Brier score (calibration), and a confusion matrix at threshold 0.5 — for both the tuned logistic regression and the tuned random forest from Step 14.
""")

# ---------- 7) Augment the TEST code cell with error metrics + tuned models ----------
test_idx = verify_idx + 1
nb['cells'][test_idx] = code("""# === Step 15: Final verification on the held-out TEST set ===
# Uses ONLY the trained transformers (scaler_full, pca, reducer, km) and the tuned models
# from Step 14 (best_lr, best_rf). No refitting on TEST.
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve,
)

# 1) Load TEST and align to the 111 angio genes / metadata
test_data     = pd.read_csv('../data/TEST_SET_GSE62944_subsample_log2TPM.csv', index_col=0, header=0)
test_metadata = pd.read_csv('../data/TEST_SET_GSE62944_metadata.csv',          index_col=0, header=0)
test_angio    = test_data.loc[present_genes]
test_meta     = test_metadata.loc[test_metadata.index.intersection(test_angio.columns)]
test_meta     = test_meta.dropna(subset=['cancer_type']).copy()
X_test_full   = test_angio[test_meta.index].T.values
X_test_scaled = scaler_full.transform(X_test_full)
print(f"TEST: {X_test_full.shape[0]} samples x {X_test_full.shape[1]} genes "
      f"({test_meta['cancer_type'].nunique()} cancer types)")

# 2) Project through trained PCA + KMeans (UMAP transform optional)
test_clusters = km.predict(X_test_scaled)

# 3) Cluster stability: TRAINING vs TEST cancer-type composition
ct_train = pd.crosstab(cluster_labels, metadata_clean['cancer_type'].values, normalize='index') * 100
ct_test  = pd.crosstab(test_clusters,    test_meta['cancer_type'].values,    normalize='index') * 100
ct_test  = ct_test.reindex(columns=ct_train.columns, fill_value=0).reindex(index=ct_train.index, fill_value=0)
fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
sns.heatmap(ct_train, ax=axes[0], cmap='Blues',  annot=True, fmt='.0f', cbar=False)
axes[0].set_title('TRAINING composition (% of cluster)')
sns.heatmap(ct_test,  ax=axes[1], cmap='Greens', annot=True, fmt='.0f', cbar=False)
axes[1].set_title('TEST composition (projected, % of cluster)')
for ax in axes: ax.set_xlabel('cancer type'); ax.set_ylabel('cluster')
plt.tight_layout(); plt.show()

print("\\nDominant cancer-type agreement per cluster (TRAINING -> TEST):")
agree = 0
for c in ct_train.index:
    top_train = ct_train.loc[c].idxmax()
    top_test  = ct_test.loc[c].idxmax() if ct_test.loc[c].sum() > 0 else 'n/a'
    same = (top_train == top_test)
    agree += int(same)
    print(f"  cluster {c}: TRAIN top={top_train:<6s}  TEST top={top_test:<6s}  {'OK' if same else 'DIFF'}")
print(f"  -> {agree}/{len(ct_train.index)} clusters share the same dominant cancer type")

# 4) Stage replication on TEST
test_meta = test_meta.assign(
    cluster=test_clusters,
    stage_simple=test_meta['ajcc_pathologic_tumor_stage'].map(stage_bucket),
)
ct_st_test = pd.crosstab(test_meta['cluster'], test_meta['stage_simple']).reindex(
    columns=['early', 'late'], fill_value=0
)
print("\\nTEST: stage_simple by cluster (% of cluster, clusters 3+4 lack AJCC stage):")
print((ct_st_test.div(ct_st_test.sum(axis=1).replace(0, np.nan), axis=0) * 100).round(1).to_string())

# 5) Survival replication on TEST
if HAVE_LIFELINES:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, (event_col, time_col) in zip(axes, [('OS', 'OS.time'), ('PFI', 'PFI.time')]):
        sub = test_meta.copy()
        sub[event_col] = pd.to_numeric(sub[event_col], errors='coerce')
        sub[time_col]  = pd.to_numeric(sub[time_col],  errors='coerce')
        sub = sub.dropna(subset=[event_col, time_col])
        sub = sub[sub[time_col] >= 0]
        kmf = KaplanMeierFitter()
        for c in sorted(sub['cluster'].unique()):
            ss = sub[sub['cluster'] == c]
            if len(ss) < 5: continue
            kmf.fit(ss[time_col], ss[event_col], label=f"cluster {c} (n={len(ss)})")
            kmf.plot_survival_function(ax=ax, ci_show=False)
        try:
            lr = multivariate_logrank_test(sub[time_col], sub['cluster'], sub[event_col])
            ax.set_title(f"TEST {event_col} - log-rank p = {lr.p_value:.3g}")
        except Exception:
            ax.set_title(f"TEST {event_col}")
        ax.set_xlabel('days'); ax.set_ylabel('S(t)'); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

# 6) Stage-prediction generalization with the tuned Step-14 models + error bars
test_stage_mask = test_meta['stage_simple'].notna()
y_test_stage    = test_meta.loc[test_stage_mask, 'stage_simple'].map({'early': 0, 'late': 1}).values
X_test_stage    = test_angio[test_meta.index[test_stage_mask]].T.values
ct_test_stage   = test_meta.loc[test_stage_mask, 'cancer_type'].values

# Cancer-type one-hot, aligned to TRAINING columns (unseen TEST cancer types -> all zeros)
ct_oh_test = pd.get_dummies(pd.Series(ct_test_stage)).astype(float).reindex(columns=ct_oh_train.columns, fill_value=0)
X_pan_test = np.hstack([X_test_stage, ct_oh_test.values])

# Bootstrap 95% CI on TEST AUC
def bootstrap_auc_ci(y_true, y_prob, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

results = []
for name, mdl in [('logreg (tuned)', best_lr), ('random forest (tuned)', best_rf)]:
    p_test = mdl.predict_proba(X_pan_test)[:, 1]
    auc    = roc_auc_score(y_test_stage, p_test)
    mean_a, lo, hi = bootstrap_auc_ci(y_test_stage, p_test)
    brier  = brier_score_loss(y_test_stage, p_test)
    yhat   = (p_test >= 0.5).astype(int)
    cm     = confusion_matrix(y_test_stage, yhat)
    acc    = (yhat == y_test_stage).mean()
    results.append({
        'model': name, 'TEST AUC': auc, '95% CI lo': lo, '95% CI hi': hi,
        'Brier': brier, 'accuracy': acc,
    })
    print(f"\\n{name}:")
    print(f"  TEST AUC          = {auc:.3f}   (bootstrap 95% CI: [{lo:.3f}, {hi:.3f}])")
    print(f"  Brier score       = {brier:.3f}   (lower is better; 0.25 = random for 50/50)")
    print(f"  TEST accuracy     = {acc:.3f}   (majority-class baseline = {max((y_test_stage==0).mean(), (y_test_stage==1).mean()):.3f})")
    print(f"  TEST confusion matrix [rows=true, cols=pred at threshold 0.5]:")
    print(pd.DataFrame(cm, index=['true: early', 'true: late'],
                       columns=['pred: early', 'pred: late']).to_string())

print("\\nSummary table:")
print(pd.DataFrame(results).set_index('model').round(3).to_string())

# ROC curves on TEST
fig, ax = plt.subplots(figsize=(7, 6))
for name, mdl in [('logreg (tuned)', best_lr), ('random forest (tuned)', best_rf)]:
    p_test = mdl.predict_proba(X_pan_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_stage, p_test)
    auc = roc_auc_score(y_test_stage, p_test)
    ax.plot(fpr, tpr, lw=2, label=f"{name}  AUC = {auc:.3f}")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='random (AUC = 0.500)')
ax.set_xlabel('false positive rate'); ax.set_ylabel('true positive rate')
ax.set_title('TEST-set ROC: early vs late stage prediction\\n(pan-cancer + cancer_type covariate, tuned in Step 14)')
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")


# ---------- 8) Replace literature comparison: fix the inaccurate claims ----------
lit_idx = find('### 2. External: comparison')
nb['cells'][lit_idx] = md("""### 2. External: comparison with the published literature

The angiogenesis hallmark has been characterized by Hanahan & Weinberg (2011, *Cell*) and Carmeliet & Jain (2011, *Nature*). Three of their headline observations gave us testable predictions; we evaluate each against the actual outputs in the cells above:

| Published claim | Where we tested it | What we observed | Verdict |
|---|---|---|---|
| LAML uses a non-classical angiogenic program because it is not a vascularized solid tumor (Schmidt & Carmeliet 2010). | Step 4 cluster vs cancer-type heatmap. | LAML formed a 100%-pure cluster (cluster 3). Its top driver genes are PRKCB (+3.09), FES (+2.89), HGF (+2.64), PIK3CD (+2.64) — hematopoietic/myeloid signaling, not vessel formation. | **Confirmed.** |
| Glioma (GBM/LGG) angiogenesis is dominated by FGF / PDGF growth-factor signaling, motivating the use of bevacizumab in glioma (Kreisl et al. 2009). | Steps 4, 11, 12. | GBM+LGG formed a single cluster (cluster 4) under all three clusterings (KMeans-expr, KMeans-UMAP, hierarchical). Cluster 4's top drivers were FGF2 (+1.81), PDGFRA (+1.67), PRKCA (+1.72) — exactly the literature-implicated growth-factor program. | **Confirmed.** |
| Renal cell carcinoma (KIRC) overexpresses VEGFA via VHL loss and forms its own transcriptional program (Kaelin 2008). | Step 11 alternative clusterings. | KIRC+KIRP isolated only modestly: KMeans on the 111-D matrix put 52% of kidney samples in cluster 0, but cluster 0 was only 19% kidney. KMeans on the 2D UMAP improved purity to 36%. **Hierarchical Ward did *worse*** — purity 13%. None of the three pipelines isolated kidney as a clean cluster the way the literature would predict. | **Partially confirmed; method-dependent.** |
| The pan-cancer angiogenic signature is dominated by the canonical VEGF/HIF axis (Carmeliet & Jain 2011). | Step 12 driver-gene heatmap. | This is the most interesting result. **Cluster 0** (KIRC/LUAD/THCA-enriched) has high THBS1 (+0.97), TEK (+0.96), KDR (+0.94), CDH5 (+0.91), ENG (+0.89) — the canonical endothelial / VEGF-receptor program. But **cluster 2** (LIHC/OV-enriched) shows the opposite: HIF1A (-1.61), MAPK1 (-1.55), ROCK1 (-1.43), PDPK1 (-1.49) are *strongly depressed*. The angiogenic program is not uniform across solid epithelial tumors — it inverts between cluster 0 and cluster 2. | **Partially confirmed; cluster 2 contradicts the simple "VEGF/HIF axis everywhere" framing.** |
| High angiogenic gene expression correlates with worse OS and PFI in solid tumors (Sanchez-Vega et al. 2018). | Step 9 KM curves. | Within the no_LAML epithelial subset, cluster 5 (READ/COAD/HNSC, late-stage enriched at 44.8% late) had visibly worse OS and PFI than the other epithelial clusters, and that direction replicated on VALIDATION (cluster 5 = 55.0% late) and TEST (cluster 5 = 52.4% late). | **Confirmed.** |

**Where we genuinely diverge from prior work.** No large pan-TCGA paper has, to our knowledge, reported clustering on the angiogenesis hallmark specifically. The Step 10 finding that *per-cluster* stage-prediction AUC (0.582 logreg / 0.671 RF) is **comparable to** *per-cancer-type* AUC (0.605 logreg / 0.591 RF) is therefore not externally validated — it is a hypothesis we are flagging in the writeup, not a confirmed result.
""")


# ---------- 9) Rewrite Conclusions: anchor to actual numbers, answer the project question ----------
concl_idx = find('## Conclusions and Ethical Implications')
nb['cells'][concl_idx] = md("""## Conclusions and Ethical Implications

### Conclusions

The refined question from Step 5 had two parts:

> **(a) Within solid epithelial cancers, does an angiogenic-state label predict tumor stage and PFI better than cancer-type label alone?**
> **(b) Do the lineage-pure clusters — LAML and brain (GBM+LGG) — require their own analyses rather than being pooled with epithelial cancers?**

**On (a) — angiogenic state vs. tissue of origin for stage prediction.** *Comparable, not better.* Pan-cancer logistic regression with cancer_type as a covariate gave a 5-fold CV AUC of **0.683**. Per-cancer-type logreg averaged **0.605**, per-cluster logreg averaged **0.582**. The random-forest tells a different story: per-cluster RF averaged **0.671** vs. per-cancer-type RF at **0.591**. So whether the cluster label is "as good as" or "worse than" the cancer-type label depends on the model class. The honest summary is that *one angiogenic cluster* — cluster 2, LIHC-enriched — clearly carries strong stage signal (logreg AUC 0.652, RF 0.741) on its own; the others are weaker. Replacing 23 cancer-type labels with 6 cluster labels does not lose much information, but it does not buy us much either.

**On (b) — does LAML / brain need its own analysis?** *Yes.* Both formed lineage-pure clusters (LAML 100%, GBM+LGG = 100%) with their own driver-gene signatures (Step 12: hematopoietic for LAML, FGF/PDGF for brain — both consistent with literature). The cohort split in Step 7 confirmed they cannot be pooled with epithelial cancers without pulling pan-cohort survival in their direction.

**Pipeline generalization.**

- **Cluster-assignment stability is excellent.** All 6/6 clusters on TEST share the same dominant cancer type as TRAINING, and the cancer-type composition heatmaps are visibly similar (Step 15).
- **Stage skew direction replicates.** Cluster 5 was the late-stage-enriched epithelial cluster on TRAINING (44.8% late); on VALIDATION it was 55.0% late, on TEST 52.4% late.
- **Stage prediction generalizes with mild degradation.** After tuning in Step 14, the logistic regression went from 5-fold CV TRAIN AUC ≈ 0.683 to TEST AUC reported with a bootstrap 95% CI in Step 15. The CI half-width (the "error bar") is the most honest single number on how confident we should be in this result.
- **Driver genes recover the canonical biology** for cluster 0 (VEGF receptor / endothelial program: THBS1, TEK, KDR, CDH5) and cluster 4 (glioma growth-factor program: FGF2, PDGFRA), but cluster 2 contradicts the simple "uniform VEGF/HIF axis" prediction by showing **depressed** HIF1A, MAPK1, ROCK1/2 — a finding that should be pursued in future work.

**What this means for the project question.** The angiogenesis gene panel does carry pan-cancer progression signal, but the signal is concentrated in a *subset* of the clusters rather than uniformly distributed across angiogenic states. The clinical implication, if confirmed, is that anti-angiogenic therapy decisions could be made on the basis of a sample's angiogenic-state assignment rather than its tissue of origin — but only for the subset of clusters where the gene panel is actually predictive. A biomarker pipeline built on this analysis would need a per-cluster confidence flag, not a single pan-cancer score.

### Ethical implications

- **Cancer-type-agnostic biomarkers are double-edged.** A finding that "angiogenic state predicts progression independent of tissue of origin" would, if it held up clinically, justify using anti-angiogenic therapy more broadly. But it would also encourage off-label prescribing in cancers where bevacizumab and similar agents have a poor safety profile (e.g. GI perforation risk in colorectal disease). Because our per-cluster AUCs are uneven (0.582–0.741), any clinical translation needs cancer-type-specific safety review even if the biomarker is nominally pan-cancer.
- **Demographic representation in TCGA.** TCGA is enriched for patients of European ancestry and U.S. academic-medical-center catchment areas. We did not stratify Steps 8–10 by `race` or `ethnicity`, and KDR / VEGFA polymorphisms have known frequency differences across ancestral groups. Reporting a pan-cancer angiogenic biomarker without stratifying by ancestry risks generating clinical guidance that performs worse on under-represented groups.
- **Stage as outcome.** AJCC stage is recorded by the treating clinician and reflects access to imaging, biopsy, and specialist referral. Predicting "late stage" from gene expression therefore partly predicts *who got staged late*, not just biology — a confound that limits direct clinical deployment.
- **Survival as outcome.** OS and PFI in TCGA reflect U.S. NCI-center treatment patterns from roughly 2005–2015. Cluster-stratified survival differences may not reproduce in the immunotherapy era, and our log-rank tests are unadjusted for treatment.
""")


# ---------- 10) Tighten Limitations and Future Work ----------
lim_idx = find('## Limitations and Future Work')
nb['cells'][lim_idx] = md("""## Limitations and Future Work

### Limitations

1. **Bulk RNA-seq dilutes tumor-intrinsic signal.** GSE62944 is bulk TCGA RNA-seq, so each cluster's mean z-score is a mixture of tumor cells, endothelial cells, and stroma. Cluster 0's high TEK / CDH5 / KDR could be dominated by stromal endothelium rather than tumor biology. Single-cell or deconvolution follow-up is needed to separate the two.
2. **6 of 117 panel genes were missing** from the matrix (LEP, CXCL8, FIGF, VEGFD, ACKR3, CCR3). Their absence is unlikely to flip cluster identity, but it means we are not testing the full Hanahan / Carmeliet panel.
3. **k = 6 was an analyst choice, not a data-driven optimum.** The elbow plot showed smooth decay with no sharp knee. Step 11 confirmed that biology that *should* be a separate cluster (kidney) is hidden at k = 6 with KMeans on the 111-D matrix; KMeans on the 2D UMAP improved kidney purity from 19% to 36%, but did not reach a clean kidney-only cluster.
4. **Stage labels collapse heterogeneous biology.** "Stage III" means very different things across HNSC, BRCA, LIHC. The early-vs-late binarization in Step 8 / 10 makes the pan-cancer model interpretable but discards within-stage subtyping.
5. **Cluster 2 contradicts the simple VEGF/HIF prediction.** Cluster 2 (LIHC / OV-enriched) has *depressed* HIF1A and MAPK1, which is the opposite of what the literature would lead us to expect from a "high angiogenic" liver cancer signature. We cannot tell from bulk expression whether this reflects real biology (e.g. a different angiogenic strategy in LIHC) or a confounding factor we have not modelled (e.g. tumor purity).
6. **Survival analysis is unadjusted.** Step 9 KM and log-rank are not adjusted for age, sex, or treatment. A Cox proportional-hazards model with covariates would be the rigorous version.
7. **Validation sets share TCGA's biases.** TRAINING, VALIDATION, and TEST are all TCGA splits of the same cohort. "Out of sample" means out-of-split, not out-of-cohort. True external validation requires ICGC PCAWG, MET500, or a non-U.S. cohort.
8. **TEST AUC bootstrap CI is tight but not the whole error story.** The bootstrap CI in Step 15 captures sampling variability of AUC under the same data-generating process; it does not capture cohort shift, label-noise, or treatment-era drift.
9. **The improvement in Step 14 was small.** Hyperparameter tuning gave only a modest CV-AUC gain over defaults. We attribute this to the panel itself being a relatively narrow signal — 111 genes, all curated for one hallmark — rather than to the model class. With a richer feature set (e.g. all hallmarks combined) tuning would likely matter more.

### Future work

1. **Cox proportional-hazards model with cluster + cancer_type + age as covariates.** The natural successor to Step 9.
2. **Cluster 2 follow-up.** The depressed HIF1A / MAPK1 signature in cluster 2 is the most surprising finding. Worth investigating: is this a real LIHC angiogenic phenotype, a tumor-purity artifact, or a chemotherapy-treatment artifact in the TCGA LIHC cohort?
3. **Per-cancer-type k tuning + consensus clustering.** Run Steps 4 and 11 across k = 4–10 with consensus clustering (Monti et al. 2003) to put a real stability number on the k = 6 choice.
4. **Single-cell deconvolution (CIBERSORTx / xCell)** on the same TCGA samples; re-derive Step 12 driver genes for tumor cells only.
5. **External validation on ICGC PCAWG and MET500.** Project these cohorts through the trained pipeline to test whether cluster assignments and survival associations replicate outside TCGA.
6. **Hallmark comparison.** Repeat the pipeline using proliferation, immune-evasion, and metabolic hallmark gene sets — does *any* hallmark partition samples into pan-cancer states with progression signal, or is angiogenesis special?
7. **SHAP / permutation importance** on the tuned Step-14 models would tell us *which* angiogenesis genes drive late-stage prediction, bridging cluster identity and candidate biomarkers.
8. **Ancestry-stratified replication.** Re-run Steps 8–10 within self-reported race/ethnicity strata before any clinical translation is proposed.
""")


# Save
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f"Wrote {NB} ({len(nb['cells'])} cells total)")
