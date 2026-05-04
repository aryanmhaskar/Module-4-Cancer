"""Smoke-test Step 14 + Step 15 by replaying Steps 1-13 then exec'ing the new cells."""
import json, sys, traceback
import matplotlib
matplotlib.use('Agg')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Step 1
data = pd.read_csv('../data/TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0, header=0)
metadata_df = pd.read_csv('../data/TRAINING_SET_GSE62944_metadata.csv', index_col=0, header=0)
desired_gene_list = ["CXCR4","CXCL13","FLT4","RHOA","TGFB2","GAB1","STAT3","LCK","FZD4","PDPK1",
"CDH5","HIF1A","MTOR","CXCL12","ITGA5","PLCG1","RRAS","ENG","TNFSF12","PTPN11","TGFBR1","TERT","PIK3CD","PTK2B","LEP","FGF2","NOS3","VTN","CCL24","SMO",
"KDR","PRKCG","RAF1","VEGFA","BRCA1","CCL2","TGFBR2","EGFR","VAV2","VEGFB","KRAS","IL6","ERBB2","PRKCA","ROCK1","TNF","IL1B","SDC4","PRKACA","CX3CR1",
"ITGB1","PDGFA","CD40","MAPK1","CXCL8","PIK3R1","FIGF","CSF3","HGF","VEGFD","IL10","MET","TP53","SDC2","TGFB1","AKT1","FLT1","CAV1","ARNT","ARHGEF1",
"PDGFRA","HSPB1","TWIST1","ITGB3","AKT3","MMP1","MAPK3","SPHK1","EPAS1","TIMP3","CXCR2","SRC","CXCL10","MAP2K1","WNT5A","EFNA1","IGF1","IL8","ROCK2","ACKR3",
"GATA2","VEGFC","CCR3","CTNNB1","TEK","FN1","EPHA2","IGF1R","THBS1","FES","ITGAV","CXCL9","CCL11","MYC","CXCR3","PRKCB","ACVRL1","HRAS","ESR1","TGFB3",
"PLAUR","RAC1","NF1","ANGPT2","GRB2","PIK3CA","IL1A"]
present_genes = [g for g in desired_gene_list if g in data.index]
angio_data = data.loc[present_genes]
shared = angio_data.columns.intersection(metadata_df.index)
metadata_clean = metadata_df.loc[shared].dropna(subset=['cancer_type'])
angio_data = angio_data[metadata_clean.index]

# Step 2
X = angio_data.T.values
y = metadata_clean['cancer_type'].values
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2, random_state=42); X_pca = pca.fit_transform(X_scaled)

# Step 3
import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Step 4
k_chosen = 6
km = KMeans(n_clusters=k_chosen, random_state=42, n_init=10)
cluster_labels = km.fit_predict(X_scaled)

# Step 6a/Step 13 prereqs: load val
val_data = pd.read_csv('../data/VALIDATION_SET_GSE62944_subsample_log2TPM.csv', index_col=0, header=0)
val_metadata = pd.read_csv('../data/VALIDATION_SET_GSE62944_metadata.csv', index_col=0, header=0)
val_angio = val_data.loc[present_genes]

# Steps 8/10 prereqs: stage_bucket, X_stage, y_stage, ct_stage
def stage_bucket(s):
    if not isinstance(s, str): return np.nan
    s = s.strip().upper()
    if s.startswith('STAGE IV'): return 'late'
    if s.startswith('STAGE III'): return 'late'
    if s.startswith('STAGE II'): return 'early'
    if s.startswith('STAGE I'): return 'early'
    return np.nan
metadata_clean = metadata_clean.assign(stage_simple=metadata_clean['ajcc_pathologic_tumor_stage'].map(stage_bucket))

stage_mask = metadata_clean['stage_simple'].notna()
y_stage = metadata_clean.loc[stage_mask, 'stage_simple'].map({'early':0,'late':1}).values
X_stage = angio_data[metadata_clean.index[stage_mask]].T.values
ct_stage = metadata_clean.loc[stage_mask, 'cancer_type'].values
cl_stage = cluster_labels[stage_mask.values]

# Step 13 prereq: scaler_full
scaler_full = StandardScaler().fit(X)

# lifelines
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    HAVE_LIFELINES = True
except ImportError:
    HAVE_LIFELINES = False

print("Setup ok. Now exec'ing Step 14 (cell 44) and Step 15 (cell 46)...")
nb = json.loads(open('Module 4 Template.ipynb','r',encoding='utf-8').read())
ns = dict(globals())
for i in [44, 46]:
    src = ''.join(nb['cells'][i]['source'])
    print(f"\n--- exec cell {i} ({src.splitlines()[0][:80]}) ---")
    try:
        exec(compile(src, f'<cell {i}>', 'exec'), ns)
    except Exception as e:
        print(f"FAILED cell {i}: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
print("\n[OK] Step 14 and Step 15 executed without error.")
