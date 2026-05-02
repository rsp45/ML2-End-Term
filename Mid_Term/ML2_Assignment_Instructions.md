# ML-2 Assignment Instructions
## Agglomerative Hierarchical Clustering on BEED Dataset

> **Dataset:** Bangalore EEG Epilepsy Dataset (BEED) · UCI ML Repository ID 1134  
> **DOI:** [10.24432/C5K33B](https://doi.org/10.24432/C5K33B) · License: CC BY 4.0  
> **Authors:** Najmusseher & Nizar Banu P K, CHRIST University, Bangalore (2024)  
> **Total Marks:** 20 pts · **Submission:** Single `.zip` containing `.ipynb` file(s)

---

## Grading Rubric

| Criterion | Marks | What the Grader Tests |
|---|---|---|
| **C1 · Algorithmic Mechanics** | 6 pts | Data structures used, exact lines explained, time/space complexity of distance matrix, linkage function, and merge loop |
| **C2 · Production Verification** | 6 pts | 100% programmatic array match between scratch and sklearn; full understanding of WHY matching is non-trivial |
| **C3 · Interpretability** | 5 pts | Decision tree validates cluster isolation using explicit numerical EEG amplitude thresholds |
| **C4 · Communication & Defense** | 3 pts | Articulate answers, direct responses, logical hypotheses about changed variables |
| **Total** | **20 pts** | |

> **Critical rule:** Every non-trivial code cell must carry inline `# [DEFEND]` comments explaining **why** — not just what. These comments are your oral exam notes.

---

## Dataset Facts (Verified from UCI Page + Paper)

```
File         :  BEED_Data.csv  (same directory as notebook)
Shape        :  8,000 rows × 17 columns — NO missing values
Recording    :  Standard 10-20 electrode system, 256 Hz sampling rate
               20-second EEG segments; 80 subjects total (20 per class)
               Ages 21–55, equal gender representation
Features     :  X1–X16  →  16 EEG channels from different brain regions
               Raw integer amplitude (μV), range ≈ −323 to +296
               X1 = earliest time point, X16 = latest in the 20s window
```

### Class Labels — `y` column (4 clinical EEG states)

| y | Class Name | Description | Count |
|---|---|---|---|
| `0` | **Healthy** | Control group — no epileptic seizures | 2,000 |
| `1` | **Generalised Seizure** | Seizure affecting both brain hemispheres simultaneously | 2,000 |
| `2` | **Focal Seizure** | Seizure localised to a single brain region / hemisphere | 2,000 |
| `3` | **Seizure Event** | Seizure co-occurring with activity (eye blinking, nail biting, staring) | 2,000 |

> **`y` is NEVER used as input to clustering.** Use only X1–X16 as features. `y` appears only in the Phase 2 contingency table and Phase 3 clinical interpretation.

---

## Optimal Sample Size Justification

Agglomerative Hierarchical Clustering requires computing a full pairwise distance matrix of shape `(n, n)`. This creates hard memory and time constraints that must be balanced against statistical representativeness.

### Sample Size Trade-off Analysis

| Sample Size | D Matrix Memory | Merge Loop (est.) | Representativeness |
|---|---|---|---|
| 400 | 1.3 MB | ~1s | Minimal — 5% of data |
| 500 | 1.9 MB | ~1s | Acceptable |
| **800** | **4.9 MB** | **~5s** | **Optimal ✓** |
| 1,000 | 7.6 MB | ~10s | Borderline |
| 1,500 | 17.2 MB | ~34s | Slow |
| 2,000 | 30.5 MB | ~80s | Impractical in notebook |

### Chosen Strategy: Stratified Sample of 800 (200 per class)

**Use stratified sampling — not random sampling.** Because the full dataset is perfectly balanced (2,000 per class), a random sample risks class imbalance in the subset. Stratified sampling guarantees 200 rows from each of the 4 classes, preserving the original distribution.

```python
# OPTIMAL SAMPLE: 800 rows, stratified, 200 per class
np.random.seed(42)
strat_idx = []
for cls in range(4):
    cls_indices = np.where(y_full == cls)[0]                    # all rows of this class
    chosen      = np.random.choice(cls_indices, size=200, replace=False)
    strat_idx.extend(chosen)

strat_idx = np.array(strat_idx)
X_raw    = df.iloc[strat_idx, :16].values.astype(float)        # shape (800, 16)
y_sample = df.iloc[strat_idx, 16].values.astype(int)           # shape (800,)
```

> **[DEFEND]** Why 800 and not 400? At 400 rows, each class has only 100 representatives — too sparse to capture the variance within generalised vs focal seizure patterns. At 800, each class has 200 rows (10% of its population), which is statistically meaningful without exceeding memory limits.  
> **[DEFEND]** Why stratified? A plain `np.random.choice(8000, 800)` gives a random class mix. With 800 draws from a balanced set, we expect ~200 per class, but variance means some classes could get 160 and others 240 — introducing sampling bias into the cluster evaluation.

---

## Notebook Structure — Standard ML Pipeline

Follow this **exact cell sequence**. Do not reorder sections.

```
Cell 00 [MD]   Title · Student name · Dataset citation · Date
Cell 01 [MD]   ## Section 1: Data Loading & Overview
Cell 02 [PY]   Imports
Cell 03 [PY]   Load CSV, inspect shape, print first 5 rows
Cell 04 [PY]   Class distribution (value_counts / Counter)
Cell 05 [PY]   Descriptive statistics (mean, std, min, max per feature)
Cell 06 [MD]   ## Section 2: Data Preprocessing
Cell 07 [PY]   Stratified sampling — 800 rows, 200 per class, seed=42
Cell 08 [PY]   Z-score normalisation (NumPy only — no sklearn yet)
Cell 09 [PY]   Verify: print shape, mean ≈ 0, std ≈ 1 after scaling
Cell 10 [MD]   ## Section 3: Exploratory Data Analysis (EDA)
Cell 11 [PY]   Feature distribution plots (boxplots per class, 4×4 grid)
Cell 12 [PY]   Per-class mean EEG waveform plot (line plot, X1–X16)
Cell 13 [PY]   Correlation heatmap of 16 features
Cell 14 [PY]   Pairplot or scatter matrix (optional — 4–6 features)
Cell 15 [MD]   ## Section 4: Phase 1 — From-Scratch Agglomerative Clustering
Cell 16 [PY]   Compute distance matrix D (broadcasting, no loops)
Cell 17 [PY]   Define complete_linkage() function
Cell 18 [PY]   Merge loop → labels_scratch
Cell 19 [PY]   Print cluster sizes + assertions
Cell 20 [PY]   Dendrogram bar chart (last 20 merges)
Cell 21 [MD]   ## Section 5: Phase 2 — Production Verification
Cell 22 [PY]   sklearn AgglomerativeClustering → labels_sklearn
Cell 23 [PY]   Adjusted Rand Index + print
Cell 24 [PY]   Hungarian remapping → exact match rate → assertions
Cell 25 [PY]   Contingency table: Cluster vs y_sample (with class key)
Cell 26 [MD]   ## Section 6: Phase 3 — Surrogate Interpretability
Cell 27 [PY]   Train DecisionTreeClassifier (max_depth=4, seed=42)
Cell 28 [PY]   export_text → print rules + convert one threshold to μV
Cell 29 [PY]   plot_tree → save decision_tree.png (180 dpi)
Cell 30 [PY]   Feature importance bar chart → save feature_importance.png
Cell 31 [MD]   Clinical interpretation (7+ sentences)
Cell 32 [MD]   ## Conclusion
```

---

## Section 1 — Data Loading & Overview

### What to produce

- Load `BEED_Data.csv` using `pandas.read_csv()`
- Print: shape, column names, dtypes, first 5 rows
- Print class distribution confirming 2,000 samples per class
- Print global descriptive statistics with `df.describe()`

### Key dataset facts to confirm in output

```
Expected shape  : (8000, 17)
Expected columns: X1, X2, ..., X16, y
Missing values  : 0  (must verify with df.isnull().sum())
y distribution  : {0: 2000, 1: 2000, 2: 2000, 3: 2000}
Feature dtype   : int64 (raw EEG amplitude integers)
```

### Reference feature statistics (full 8,000-row dataset)

| Feature | Mean (μV) | Std (μV) | Min (μV) | Max (μV) |
|---|---|---|---|---|
| X1 | −1.5 | 36.8 | −281 | 252 |
| X6 | −2.3 | 36.3 | −277 | 245 |
| X9 | −1.6 | 38.1 | −290 | 280 |
| X12 | −4.8 | 37.5 | −306 | 283 |
| X16 | −4.1 | 35.9 | −317 | 270 |

> All 16 features share similar amplitude ranges (±300 μV), near-zero means, and standard deviations of ~36–38 μV.

---

## Section 2 — Data Preprocessing

### Steps in order

**Step 2.1 — Stratified Sampling (800 rows)**

```python
import numpy as np
import pandas as pd

df    = pd.read_csv("BEED_Data.csv")
y_full = df['y'].values

np.random.seed(42)
strat_idx = []
for cls in range(4):
    cls_indices = np.where(y_full == cls)[0]
    chosen      = np.random.choice(cls_indices, size=200, replace=False)
    strat_idx.extend(chosen)

strat_idx = np.array(strat_idx)
X_raw    = df.iloc[strat_idx, :16].values.astype(float)   # (800, 16)
y_sample = df.iloc[strat_idx, 16].values.astype(int)      # (800,)

print(f"Sampled shape: {X_raw.shape}")
print(f"Class distribution: {dict(zip(*np.unique(y_sample, return_counts=True)))}")
# Expected: {0: 200, 1: 200, 2: 200, 3: 200}
```

**Step 2.2 — Z-Score Normalisation (NumPy ONLY — Phase 1 restriction)**

```python
# PHASE 1 RULE: no sklearn imports allowed here
mu  = X_raw.mean(axis=0)    # shape (16,) — per-channel mean
sig = X_raw.std(axis=0)     # shape (16,) — per-channel std deviation
X   = (X_raw - mu) / sig    # shape (800, 16) — zero mean, unit variance

# Verify normalisation
print(f"After scaling — mean ≈ {X.mean(axis=0).mean():.4f} (expect ≈ 0)")
print(f"After scaling — std  ≈ {X.std(axis=0).mean():.4f}  (expect ≈ 1)")

# [DEFEND] WHY normalise?
# X12 reaches ±306 μV while X1 stays near ±50 μV. Without scaling,
# high-amplitude channels dominate Euclidean distance entirely —
# clustering becomes amplitude-biased, not brain-state-biased.
# Z-score gives all 16 EEG channels equal statistical weight.
```

---

## Section 3 — Exploratory Data Analysis (EDA)

### Plot 1 — Boxplots by class

Create a 4×4 subplot grid. Each subplot = one EEG feature (X1–X16). In each subplot, draw 4 boxplots — one per class (y=0,1,2,3). This reveals which features have the most class-separating spread.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()
class_labels = ['Healthy (0)', 'Generalised (1)', 'Focal (2)', 'Seizure Event (3)']
colors = ['#4a90d9', '#e05252', '#52c06a', '#f5a623']

for i, ax in enumerate(axes):
    feat = f'X{i+1}'
    data_by_class = [X_raw[y_sample == cls, i] for cls in range(4)]
    bp = ax.boxplot(data_by_class, patch_artist=True, notch=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(feat, fontsize=10, fontweight='bold')
    ax.set_xticklabels(['H', 'G', 'F', 'SE'], fontsize=8)
    ax.set_ylabel('Amplitude (μV)', fontsize=7)

fig.suptitle('EEG Amplitude Distribution per Class\n(H=Healthy, G=Generalised, F=Focal, SE=Seizure Event)',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("eda_boxplots.png", dpi=150, bbox_inches='tight')
plt.show()
```

### Plot 2 — Mean EEG Waveform per Class

Plot the mean amplitude across X1–X16 for each class as a line — this is the "average EEG waveform shape" per brain state.

```python
fig, ax = plt.subplots(figsize=(11, 5))
time_points = [f'X{i}' for i in range(1, 17)]

for cls, (label, color) in enumerate(zip(class_labels, colors)):
    class_mean = X_raw[y_sample == cls].mean(axis=0)
    ax.plot(time_points, class_mean, marker='o', markersize=4,
            label=label, color=color, linewidth=2)

ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.set_title('Mean EEG Waveform per Class (X1=t₀ → X16=t₁₅)', fontsize=13)
ax.set_xlabel('EEG Channel (time point within 20s window)')
ax.set_ylabel('Mean Amplitude (μV)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("eda_mean_waveform.png", dpi=150)
plt.show()

# [DEFEND] Class 0 (Healthy) shows negative mean ≈ −12 μV across all channels,
# reflecting the quieter baseline EEG. Generalised seizures (y=1) show
# larger positive amplitudes in early channels, consistent with synchronous
# high-amplitude activity across both hemispheres.
```

### Plot 3 — Feature Correlation Heatmap

```python
corr_matrix = np.corrcoef(X_raw.T)   # (16, 16) correlation matrix

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='Pearson Correlation')
ax.set_xticks(range(16)); ax.set_xticklabels([f'X{i}' for i in range(1,17)], rotation=45)
ax.set_yticks(range(16)); ax.set_yticklabels([f'X{i}' for i in range(1,17)])
ax.set_title('Feature Correlation Matrix — 16 EEG Channels', fontsize=13)
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150)
plt.show()

# [DEFEND] Adjacent channels are highly correlated (X1↔X2: r≈0.89, X2↔X3: r≈0.91)
# because neighbouring time points in a continuous EEG signal overlap.
# There are 23 pairs with |r| > 0.7. This means the 16 features are NOT
# independent — agglomerative clustering on Euclidean distance will naturally
# group samples whose time-series waveform SHAPES are similar.
```

---

## Section 4 — Phase 1: From-Scratch Agglomerative Clustering

> **Library restriction:** `numpy`, `matplotlib`, `pandas` (I/O only) — zero `sklearn` / `scipy` imports.

### Step 4.1 — Distance Matrix

```python
n = X.shape[0]   # 800

# Vectorised broadcasting: no Python loops
diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]   # (800, 800, 16)
D    = np.sqrt((diff ** 2).sum(axis=2))              # (800, 800)
np.fill_diagonal(D, np.inf)   # self-distance = inf → never chosen as minimum

print(f"Distance matrix shape : {D.shape}")
print(f"Memory usage          : {D.nbytes / 1024**2:.1f} MB")
print(f"Min non-inf distance  : {D[D < np.inf].min():.4f}")
print(f"Max distance          : {D.max():.4f}")

# [DEFEND C1] Time complexity  : O(n²·d) = O(800²·16) ≈ 10.24M multiply-adds
# [DEFEND C1] Space complexity : O(n²)   = 800² × 8 bytes ≈ 4.9 MB — within budget
# [DEFEND C1] Why no nested loops? Python interpreter overhead for 640,000 pair
#             iterations would take ~30s. NumPy C-level broadcasting: ~0.1s.
# [DEFEND C4] Why Euclidean distance? It measures the geometric dissimilarity of
#             two EEG amplitude vectors. Points that are "close" have similar
#             waveform shapes and amplitudes — likely the same brain state.
```

### Step 4.2 — Initialise Data Structures

```python
clusters = [{i} for i in range(n)]   # list of 800 singleton sets
history  = []                         # (merge_distance, merged_cluster_size) per step
k        = 4                          # target = 4 clinical EEG states

# [DEFEND C1] WHY list of sets?
# · Merging two clusters = set union operator (|) → O(min(|A|, |B|))
# · Membership lookup → O(1) average (hash table)
# · After merging: list.pop(j) → O(current_k) to maintain compact list
# · Alternative: array of integer labels requires O(n) relabelling per merge
```

### Step 4.3 — Linkage Function

```python
def complete_linkage(ci, cj, D):
    """
    Complete linkage: distance = maximum pairwise distance between clusters.

    Parameters
    ----------
    ci, cj : set of int   — row indices belonging to each cluster
    D      : ndarray      — (n×n) precomputed distance matrix

    Returns
    -------
    float  — max distance across all (a ∈ ci, b ∈ cj) pairs
    """
    i_arr = np.array(list(ci))          # set → sorted index array
    j_arr = np.array(list(cj))
    # np.ix_ constructs an open mesh for 2D submatrix extraction
    return D[np.ix_(i_arr, j_arr)].max()

# [DEFEND C1] WHY complete linkage?
# · Produces compact, roughly equal-sized clusters — good for 4 balanced EEG states
# · "Furthest neighbour" strategy avoids chaining (single linkage problem)
# · Must match Phase 2: sklearn AgglomerativeClustering(linkage='complete')
#
# [DEFEND C4] What if single linkage?
# · Uses min pairwise distance ("nearest neighbour")
# · Creates chaining effect — one giant cluster grows by absorbing nearby points
# · Remaining 3 clusters become tiny fragments
# · ARI against sklearn with complete linkage would be near 0
#
# [DEFEND C4] What if Ward linkage?
# · Minimises increase in within-cluster variance at each merge
# · Even more compact clusters but requires tracking centroids/sizes
# · Harder to implement from scratch correctly
```

### Step 4.4 — Main Merge Loop

```python
print(f"Starting merge loop: {n} clusters → {k} clusters ({n-k} merges needed)")

while len(clusters) > k:
    best_dist = np.inf
    merge_a, merge_b = 0, 1          # indices into clusters list

    # Scan all O(c²) pairs (c = current cluster count)
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):    # j > i → no duplicate pairs
            d = complete_linkage(clusters[i], clusters[j], D)
            if d < best_dist:
                best_dist = d
                merge_a, merge_b = i, j

    # Record for dendrogram
    new_size = len(clusters[merge_a]) + len(clusters[merge_b])
    history.append((best_dist, new_size))

    # Merge: union clusters[merge_b] INTO clusters[merge_a]
    clusters[merge_a] = clusters[merge_a] | clusters[merge_b]
    clusters.pop(merge_b)    # safe: merge_b > merge_a → lower indices unaffected

    merges_done = n - len(clusters)
    if merges_done % 100 == 0:
        print(f"  Step {merges_done}/{n-k} — clusters remaining: {len(clusters)}")

print(f"\nMerge loop complete. Final cluster count: {len(clusters)}")

# [DEFEND C1] Time complexity of full loop:
#   (n-k) merge steps × O(c²) pair scans × O(|ci|·|cj|) linkage per pair
#   ≈ 796 merges × ~400 avg pairs × ~10 avg elements = manageable
# [DEFEND C1] clusters.pop(merge_b): list.pop(j) is O(k) not O(1),
#   but with k≤800 at any step this is trivial compared to linkage cost
# [DEFEND C1] merge_b > merge_a (enforced by j > i in loop),
#   so pop(merge_b) never shifts merge_a's index
```

### Step 4.5 — Assign Labels

```python
from collections import Counter

labels_scratch = np.zeros(n, dtype=int)
for cid, members in enumerate(clusters):
    for idx in members:
        labels_scratch[idx] = cid

sizes = dict(Counter(labels_scratch))
print("Phase 1 complete.")
print(f"Cluster sizes (scratch): {sizes}")
assert len(np.unique(labels_scratch)) == k, "ERROR: Not exactly 4 clusters!"
print("✓ Assertion passed — exactly 4 non-empty clusters.")
```

### Step 4.6 — Dendrogram (Last 20 Merges)

```python
dists = [h[0] for h in history[-20:]]
steps = list(range(len(history) - 20, len(history)))

plt.figure(figsize=(11, 4))
plt.bar(steps, dists, color='#4a90d9', edgecolor='#e8e8f0', linewidth=0.5)
plt.axvline(x=len(history) - k + 0.5, color='#e05252', linestyle='--',
            linewidth=1.5, label=f'Cut at k={k}')
plt.xlabel("Merge Step")
plt.ylabel("Complete Linkage Distance (Z-score units)")
plt.title("Dendrogram — Last 20 Merges\n(Large gap before red line validates k=4)")
plt.legend()
plt.tight_layout()
plt.savefig("dendrogram_scratch.png", dpi=150)
plt.show()

# [DEFEND C4] A visible "elbow" — a sudden jump in bar height just BEFORE
# the red cut line — confirms that stopping at k=4 is the right decision.
# The last k-1 merges should be noticeably more expensive (higher distance)
# than the merges that precede them.
```

---

## Section 5 — Phase 2: Production Verification

> All libraries now permitted.

### Step 5.1 — sklearn Reference Run

```python
from sklearn.cluster import AgglomerativeClustering

# Parameters MUST exactly mirror Phase 1
model = AgglomerativeClustering(
    n_clusters=4,
    metric='euclidean',   # same distance as Phase 1
    linkage='complete'    # MUST match — any difference → ARI < 1.0
)
labels_sklearn = model.fit_predict(X)   # same X: 800 scaled rows

from collections import Counter
print("sklearn cluster sizes:", dict(Counter(labels_sklearn)))

# [DEFEND C2] If we changed linkage here to 'ward' but kept Phase 1 as
# 'complete', the partitions would differ entirely and ARI would be near 0.
# The mathematical equivalence proof only holds when parameters are identical.
```

### Step 5.2 — Adjusted Rand Index

```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(labels_scratch, labels_sklearn)
print(f"Adjusted Rand Index (ARI): {ari:.6f}")

# [DEFEND C2] WHY ARI and not accuracy or simple match?
# Cluster integer labels are arbitrary — our scratch "Cluster 0" could be
# sklearn's "Cluster 3". A direct == comparison gives ~25% even for a
# perfect implementation. ARI measures partition STRUCTURE, independent
# of how integers are assigned to clusters.
# ARI = 1.0  → perfect agreement
# ARI ≈ 0.0  → random-chance agreement
# ARI < 0    → worse than random
# TARGET: ARI ≥ 0.98
```

### Step 5.3 — Hungarian Remapping (100% Exact Array Match)

```python
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# Build (4×4) confusion matrix: rows=sklearn labels, cols=scratch labels
conf = confusion_matrix(labels_sklearn, labels_scratch)
print("Confusion matrix (before remapping):")
print(conf)

# Hungarian algorithm: maximise diagonal sum = optimal 1-to-1 label mapping
# We negate conf because linear_sum_assignment MINIMISES cost
row_ind, col_ind = linear_sum_assignment(-conf)

# Build remapping dictionary: scratch label → sklearn label
label_map       = {col_ind[r]: row_ind[r] for r in range(k)}
labels_remapped = np.array([label_map[lbl] for lbl in labels_scratch])

exact_match = (labels_remapped == labels_sklearn).mean()
print(f"\nLabel remapping applied: {label_map}")
print(f"Exact array match after remapping: {exact_match * 100:.2f}%")

# [DEFEND C2] WHY Hungarian algorithm?
# With k=4 labels there are 4! = 24 possible permutations.
# Hungarian solves the optimal assignment in O(k³) = O(64) operations.
# It treats the confusion matrix as a "reward matrix" and finds the
# permutation that maximises the number of matching assignments.
# [DEFEND C2] "100% match" = labels_remapped[i] == labels_sklearn[i]
# for ALL 800 rows — this is the programmatic proof the grader requires.
```

### Step 5.4 — Hard Assertions

```python
assert ari >= 0.98, f"FAIL: ARI={ari:.4f} — fix Phase 1 before continuing."
assert exact_match >= 0.99, f"FAIL: Exact match={exact_match:.2%} — check linkage."
print("✓ VERIFICATION PASSED — scratch implementation matches sklearn.")
```

### Step 5.5 — Contingency Table vs Ground Truth

```python
import pandas as pd

ct = pd.crosstab(
    pd.Series(labels_sklearn,  name='Cluster (unsupervised)'),
    pd.Series(y_sample,        name='True EEG Class (y)')
)
print("Contingency Table — Unsupervised Clusters vs Clinical Ground Truth:")
print(ct)
print("\nClass key:  y=0 Healthy  |  y=1 Generalised Seizure")
print("            y=2 Focal Seizure  |  y=3 Seizure Event")

# [DEFEND C3] If each cluster row is dominated by ONE column,
# the unsupervised geometry alone separated the 4 clinical EEG states.
# A "messy" table is also a valid finding: it means Euclidean distance
# on raw amplitude cannot cleanly separate all 4 states, and spectral
# features (FFT, wavelet) would be needed — as confirmed by the BEED
# paper (Najmusseher et al., 2024) which used FFT+UMAP for 96.71% accuracy.
```

---

## Section 6 — Phase 3: Surrogate Interpretability

### Step 6.1 — Train Surrogate Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

feat_names = [f'X{i}' for i in range(1, 17)]

dt = DecisionTreeClassifier(
    max_depth=4,           # shallow = readable = easy to defend in viva
    min_samples_leaf=5,    # prevents single-point leaf nodes
    random_state=42
)
dt.fit(X, labels_sklearn)   # features = scaled EEG, target = cluster labels

surr_acc = dt.score(X, labels_sklearn)
print(f"Surrogate accuracy: {surr_acc * 100:.2f}%")

# [DEFEND C3] WHAT IS A SURROGATE MODEL?
# The clustering is a "black box" — it produces labels but no rules.
# A surrogate decision tree is trained to MIMIC those cluster assignments
# using the original features. It converts distance-based groupings into
# human-readable IF-THEN amplitude threshold rules.
#
# [DEFEND C3] max_depth=4 → at most 16 leaves for 4 clusters.
# Deep enough to be accurate, shallow enough to read and defend.
#
# [DEFEND C3] If accuracy < 80%: clusters are not linearly separable in
# raw amplitude space. This is a valid finding — not a bug. The original
# BEED paper achieved 96.71% accuracy only after adding FFT spectral features.
```

### Step 6.2 — Print Text Rules with Threshold Conversion

```python
rules = export_text(dt, feature_names=feat_names)
print("=" * 60)
print("DECISION TREE RULES — EEG Cluster Membership Conditions")
print("=" * 60)
print(rules)

# Convert Z-score thresholds back to original μV (clinical units)
print("\n--- Threshold Conversion: Z-score → Original μV ---")
print("Formula: original_μV = (z_threshold × σ_channel) + μ_channel")
print()
for i in range(16):
    print(f"  X{i+1:2d}: σ={sig[i]:.2f} μV,  μ={mu[i]:.2f} μV")

# Example conversion for the first split feature (read from rules output above)
# REPLACE with the actual feature and threshold from your tree output:
example_feat_idx = 7          # X8 (index 7) — adjust after running
example_z_thresh = 0.42       # z-score threshold — adjust after running
original_uv = (example_z_thresh * sig[example_feat_idx]) + mu[example_feat_idx]
print(f"\nExample: X{example_feat_idx+1} ≤ {example_z_thresh} (Z) → ≤ {original_uv:.1f} μV (original)")

# [DEFEND C3] The grader will ask about SPECIFIC thresholds.
# After running, identify the root split feature and both child thresholds.
# Convert each to μV and state whether it indicates high-amplitude (seizure)
# or low-amplitude (healthy) EEG activity.
# Clinical reference: normal EEG ≈ ±100 μV; seizure spikes up to ±500 μV
```

### Step 6.3 — Tree Visualisation

```python
fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(
    dt,
    feature_names=feat_names,
    class_names=[f'Cluster {i}' for i in range(4)],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax
)
ax.set_title(
    "Surrogate Decision Tree — EEG Cluster Membership Rules\n"
    "BEED Dataset · 4 EEG States · k=4 Agglomerative Clustering (Complete Linkage)",
    fontsize=13, pad=14
)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=180, bbox_inches='tight')
plt.show()
print("✓ Saved: decision_tree.png")
```

### Step 6.4 — Feature Importance Chart

```python
importances = dt.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]

bar_colors = ['#4a90d9' if importances[i] >= 0.05 else '#8a8aaa' for i in sorted_idx]

plt.figure(figsize=(12, 5))
plt.bar([feat_names[i] for i in sorted_idx], importances[sorted_idx],
        color=bar_colors, edgecolor='#1a1a2e', linewidth=0.5)
plt.axhline(y=0.05, color='#e05252', linestyle='--', linewidth=1.2,
            label='Importance threshold (0.05)')
plt.title("Feature Importances — Which EEG Channels Drive Cluster Separation?\n"
          "(X1=earliest signal, X16=latest; blue bars = most important features)")
plt.xlabel("EEG Channel (X1–X16)")
plt.ylabel("Gini Importance Score")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

# [DEFEND C3] Gini importance = total reduction in node impurity, weighted
# by the proportion of samples reaching each node.
# If early channels (X1–X5) dominate → clusters differ in EEG ONSET patterns
# If late channels (X12–X16) dominate → clusters differ in RECOVERY patterns
# Generalised seizures (y=1) show high-amplitude activity across all channels;
# focal seizures (y=2) show localised patterns in specific channel ranges.
```

### Step 6.5 — Clinical Interpretation (Mandatory Markdown Cell)

Write a Markdown cell with **minimum 7 sentences** covering:

1. Name the top 2–3 EEG channels (Xn) appearing at the tree's root and first child splits.
2. State the exact Z-score threshold AND the converted μV value for the root split.
3. Interpret clinically: is the threshold indicating high-amplitude seizure activity or low-amplitude healthy baseline?
4. From the contingency table (Phase 2), identify which cluster most closely aligns with `y=0` (Healthy) and which with `y=1` (Generalised Seizure).
5. Address the "what if single linkage?" hypothetical — explain the chaining effect and why it would invalidate the comparison.
6. Address the "what if k=2?" hypothetical — would it map to the binary seizure/healthy split? What information would be lost?
7. State one key limitation: raw amplitude (X1–X16) misses frequency-domain patterns. The original BEED paper (Najmusseher et al., 2024) used FFT + UMAP and achieved 96.71% accuracy — significantly above what unsupervised clustering on raw amplitudes can achieve.

---

## Hard Constraints

| Rule | Consequence if violated |
|---|---|
| Phase 1: ZERO `sklearn`/`scipy` imports | Criterion 1 automatic fail |
| Seed = `np.random.seed(42)` | All downstream results change |
| `k = 4` throughout | Misaligned with clinical dataset structure |
| Distance matrix = broadcasting only, no Python loops | Criterion 1 deducted |
| ARI assertion `≥ 0.98` must pass before Phase 3 | Phase 3 results are meaningless otherwise |
| `decision_tree.png` must be `plt.savefig()`-saved to disk | Criterion 3 deliverable missing |
| Every non-trivial cell must have `# [DEFEND]` comments | Criterion 4 — cannot defend what isn't documented |
| Notebook must run clean after kernel restart → Run All | Grader cannot verify any result |

---

## Grader Checklist — All 20 Items Must Pass

### C1 · Algorithmic Mechanics (6 pts)
- [ ] `clusters = [{i} for i in range(n)]` — list-of-sets initialised
- [ ] Distance matrix uses `np.newaxis` broadcasting, zero Python loops
- [ ] `complete_linkage()` uses `np.ix_` fancy indexing
- [ ] Merge loop has `# [DEFEND]` on complexity + `pop()` safety logic
- [ ] `history` list records merge distances for dendrogram
- [ ] Cluster sizes printed with `Counter`; assert 4 unique labels passes

### C2 · Production Verification (6 pts)
- [ ] sklearn call: `linkage='complete'`, `metric='euclidean'`, `n_clusters=4`
- [ ] ARI printed and `≥ 0.98` (assertion passes without error)
- [ ] Confusion matrix built and printed before remapping
- [ ] Hungarian remapping with `linear_sum_assignment(-conf)`
- [ ] Exact match `≥ 99%` printed after remapping
- [ ] Contingency table vs `y_sample` printed with class key

### C3 · Interpretability (5 pts)
- [ ] `decision_tree.png` saved at 180 dpi (`plt.savefig` confirmed in output)
- [ ] `export_text` output shows feature names AND numeric thresholds
- [ ] Feature importance chart saved as `feature_importance.png`
- [ ] At least ONE Z-score threshold converted to original μV
- [ ] Markdown cell connects cluster identities to `y=0/1/2/3` EEG states
- [ ] Generalised vs focal seizure distinction mentioned

### C4 · Communication & Defense (3 pts)
- [ ] Every non-trivial cell has `# [DEFEND]` comment(s)
- [ ] "What if single linkage?" answered in interpretation Markdown
- [ ] Limitation of raw amplitude vs spectral features mentioned with paper citation
- [ ] Conclusion cell summarises all 3 phases in plain English
- [ ] Cell 00 has student name, dataset DOI citation, and date

---

## Submission

```
your_name_ML2_assignment.zip
├── ML2_Clustering_BEED.ipynb    ← all 32 cells, runs clean top-to-bottom
├── BEED_Data.csv                ← original dataset file
├── decision_tree.png            ← 180 dpi, required deliverable
├── feature_importance.png       ← supporting visual
├── dendrogram_scratch.png       ← Phase 1 merge history
├── eda_boxplots.png             ← EDA deliverable
├── eda_mean_waveform.png        ← EDA deliverable
└── eda_correlation.png          ← EDA deliverable
```

> **Final check before zipping:** Kernel → Restart & Run All. Confirm zero errors, all assertions pass, all 3 PNG files saved.

---

*Dataset: Najmusseher & Nizar Banu P K (2024). BEED: Bangalore EEG Epilepsy Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5K33B*
