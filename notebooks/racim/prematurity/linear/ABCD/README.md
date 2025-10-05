# Prematurity Classification Pipeline (ABCD Embeddings)

> **Purpose**  
> Train simple linear classifiers on regional/whole-brain sulcal embeddings to predict prematurity, while **harmonizing site effects** (ComBat) and **residualizing confounds**. The pipeline reports cross-validated AUCs, runs a label-permutation test, and writes results per *(region, threshold)*.

---

## Table of Contents

1. [Overview](#overview)
2. [Inputs & Outputs](#inputs--outputs)
3. [Environment](#environment)
4. [Core Steps](#core-steps)
5. [Parallelization & Checkpointing](#parallelization--checkpointing)
6. [How to Run](#how-to-run)
7. [Key Functions](#key-functions)
8. [Modeling Choices](#modeling-choices)
9. [Configuration Knobs](#configuration-knobs)
10. [Common Pitfalls](#common-pitfalls)
11. [Minimal Example](#minimal-example)
12. [Optional: Fold Balance Plots](#optional-fold-balance-plots)
13. [Reproducibility Notes](#reproducibility-notes)

---

## Overview

### Key Ideas

- **Train-only harmonization and residualization** per fold (no leakage).
- **ComBat** preserves the **prematurity label** by passing it as a discrete covariate to avoid removing true signal.
- **OLS residualization** removes confounds (age, sex, income, etc.) from embeddings.
- **Logistic regression (L2)** with **light global tuning** of `C`; evaluation via **OOF AUC** and an **empirical permutation test**.

---

## Inputs & Outputs

### Inputs

1. **Embeddings (CSV per region)**  
    Path: `/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation/embeddings/ABCD_embeddings/`  
    Columns:
    - `ID`, `ID_clean` (normalized)
    - Embedding columns: `dim1`, `dim2`, … (matched via regex `^dim`)

2. **Labels**  
    Path: `/neurospin/dico/rmenasria/Runs/03_main/Input/ABCD/prematurity_labels_true_classes.csv`  
    Required columns:
    - `src_subject_id`, `prem_class` (e.g., `"28-32"` or `">=37"`)

3. **Confounds** (merged internally):  
    - `/neurospin/dico/rmenasria/Runs/03_main/Input/ABCD/all_labels_clean_abcd_new_classes.csv`
    - `/neurospin/dico/rmenasria/Runs/03_main/Input/ABCD/income.csv`  
    Processed into:
    - One-hot `siteXX`, mapped `sex`, z-scored `interview_age→scan_age`, and income fields.

### Outputs

- **Append-only CSV** (one row per *(region, threshold)*):  
  Example path: `/neurospin/dico/rmenasria/Runs/03_main/Program/2025_rmenasria_prematurity/notebooks/racim/ABCD_prematurity_results_final_28_32_corrected.csv`  
  Columns:
  - `region`, `threshold`
  - `auc_per_fold` (list of 5 floats), `auc_mean`, `auc_std`, `auc_oof`
  - `fixed_C`
  - `perm_scores` (array), `perm_pvalue`, `ci95_null`, `auc_threshold_at_alpha`
  - `n_permutations_effective`
  - `final_estimator` (sklearn Pipeline), `final_coef_original`, `final_intercept_original`
  - `duration_min`  

> **Note**: Array/object fields are stringified; parse later with `ast.literal_eval` if needed.

---

## Environment

- **Python packages (non-exhaustive)**: `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `tqdm`, `neurocombat-sklearn`, `matplotlib`, `seaborn`.
- **Compatibility Note**: `neurocombat_sklearn` expects the old `OneHotEncoder(sparse=...)`. A small wrapper maps `sparse` → `sparse_output` and is injected as:
  ```python
  import neurocombat_sklearn.neurocombat_sklearn as ncs
  ncs.OneHotEncoder = OneHotEncoderCompat
  ```

---

## Core Steps

1. **Load embeddings**  
    - Find the CSV for a region, read it, and create `ID_clean` from `ID` (drop `sub-`, remove underscores).

2. **Build confounds**  
    - One-hot `site_id_l → siteXX`.
    - Map `sex`: `{1.0→0, 2.0→1, 3.0→1}`.
    - Z-score `scan_age` from `interview_age`.
    - Merge income fields.

3. **Prepare CV data**  
    - Merge embeddings, labels, and confounds on IDs.
    - Filter binary task: `{threshold, ">=37"}`; set `y∈{0,1}`.
    - Build stratification labels by concatenating `y` with binned confounds.

4. **Residualization with ComBat (per outer fold)**  
    - Detect site labels or auto-detect (e.g., `site`, `scanner`).
    - Apply ComBat and OLS residualization.

5. **Classification & evaluation**  
    - Global tuning: GridSearchCV for `C` (L2 logistic regression).
    - Compute OOF AUC, permutation test, and final refit.

---

## Parallelization & Checkpointing

- **Outer parallelism**: `ProcessPoolExecutor(max_workers=n_jobs_outer)` over `(region, threshold)` jobs.
- **Inner parallelism**: `sklearn` `n_jobs` for GridSearch/permutations.
- **Checkpointing**: Results are appended incrementally to CSV with `flush()`.

---

## How to Run

1. **Set configuration paths**:
    ```python
    labels_path = "/.../prematurity_labels_true_classes.csv"
    base_path   = "/.../ABCD_embeddings/"
    output_csv  = "/.../ABCD_prematurity_results_final_28_32_corrected.csv"
    thresholds  = ["28-32"]
    ```

2. **Get region list**:
    ```python
    region_list = get_region_list("/neurospin/dico/data/deep_folding/current/sulci_regions_champollion_V1.json")
    ```

3. **Choose parallel settings**:
    ```python
    n_jobs_outer = 3
    n_jobs_inner = 28
    ```

4. **Launch processing**:
    Execute the block to process all `(region, threshold)` combinations.

---

## Key Functions

- `load_embeddings(region) → DataFrame`: Reads a region file, builds `ID_clean`.
- `set_confound_df() → DataFrame`: Builds confounds (e.g., one-hot sites, z-scored age).
- `prepare_cv_data(...) → dict`: Prepares `X`, `y`, and stratification labels.
- `residualize_in_folds_from_prep_combat_final(...) → list[dict]`: Performs ComBat + OLS residualization.
- `classify_with_resid(...) → dict`: Returns AUC metrics, permutation stats, and final estimator.


# Key Functions (Detailed)

Below is a detailed breakdown of the key building blocks of the pipeline — each encapsulates a major stage of the preprocessing, harmonization, or modeling process.  

---

## `load_embeddings(region) → pd.DataFrame`

**Purpose:**  
Load the embeddings corresponding to one sulcal region, clean subject IDs, and prepare for merging with labels and confounds.

**Details:**
- Scans the embeddings directory (`base_path`) to find the `.csv` file whose name starts with the given `region` and ends with `.csv`.
- Reads the file with `pandas.read_csv`.
- Constructs a standardized `ID_clean` column:
  ```python
  emb_df['ID_clean'] = (
      emb_df['ID'].astype(str)
      .str.replace(r'^sub-', '', regex=True)
      .str.replace('_', '', regex=False)
  )
  ```
  This ensures consistency across all datasets (`labels`, `confounds`, etc.) before merging.
- Returns a clean DataFrame containing:
  - `ID_clean`
  - `dim1, dim2, …` (embedding features)

**Why it matters:**  
Different datasets (labels, confounds, embeddings) often have slightly inconsistent ID formats. This step ensures correct joins across tables.

---

## `set_confound_df() → pd.DataFrame`

**Purpose:**  
Build and preprocess the confound matrix used for both stratification and residualization.

**Pipeline:**
1. **Load base confounds**:  
   Reads `/all_labels_clean_abcd_new_classes.csv` (includes site, sex, interview age, etc.)  
2. **One-hot encode site information:**
   ```python
   pd.get_dummies(conf_df, columns=['site_id_l'], prefix='', prefix_sep='', drop_first=True)
   ```
   → yields columns like `site02`, `site03`, …, each a binary indicator.
3. **Load and merge income data**:  
   Reads `/income.csv`, cleans `src_subject_id_clean`, and merges `income_continuous`, `missing_income`.
4. **Encode categorical sex:**
   - Mapping `{1.0:0, 2.0:1, 3.0:1}` collapses multiple “female” labels to one category.
   - Encoded via an internal helper `define_sex_class_mapping`.
5. **Normalize age:**  
   - Fill missing `interview_age` with a sentinel (115).  
   - Apply `StandardScaler` to z-score the variable into `scan_age`.
6. **Select relevant columns:**  
   Keeps only:
   ```
   ['scan_age', 'sex', 'src_subject_id_clean', 'income_continuous', 'missing_income'] + [all site one-hots]
   ```

**Output:**  
A clean confound DataFrame ready for merging, containing **numeric** and **categorical** covariates.

**Why it matters:**  
This step standardizes all nuisance variables (age, site, income, sex), making them usable both for residualization and ComBat harmonization.

---

## `prepare_cv_data(...) → dict`

**Purpose:**  
Merge embeddings, labels, and confounds; binarize the task; and generate stratification labels ensuring balanced folds.

**Inputs:**
- `embeddings_df`, `labels_df`, `confounds_df`
- `threshold`: the prematurity class boundary (e.g., `"28-32"`)
- Optional configuration: regex for embeddings, number of CV folds, number of quantile bins, etc.

**Process:**
1. Convert all ID columns to `str` for safe merging.
2. **Merge datasets**:  
   - `embeddings_df + labels_df` (on `ID_clean` ↔ `src_subject_id`)  
   - then merge with `confounds_df` (on `ID_clean` ↔ `src_subject_id_clean`).
3. **Select samples of interest:**  
   Keep only those with `prem_class ∈ {threshold, ">=37"}`.
4. **Binarize labels:**
   ```python
   df['y'] = (df['prem_class'] == threshold).astype(int)
   ```
5. **Extract feature matrix:**
   - Select columns matching `r'^dim'` → `X_all_df`, `X_all`.
   - Labels and IDs → `y_all`, `ids_all`.
6. **Define confounds for residualization:**  
   If none specified, use all columns from `confounds_df` except the ID.
7. **Build stratification labels:**  
   - For each confound:
     - If categorical or has ≤10 unique values → convert to string.
     - If continuous → discretize into quantile bins using `pd.qcut`.
   - Concatenate all parts:  
     `strat_label = y + "||" + conf1 + "|" + conf2 + ...`
   - If any stratum is too small for CV (less than `n_splits`), reduce number of bins adaptively.

**Outputs:**
- `X_all_df`, `X_all`, `y_all`, `ids_all`
- `stratify_labels` for stratified cross-validation
- `confounds_resid_df` for residualization

**Why it matters:**  
Ensures perfect alignment across all sources and creates stratified folds that remain balanced w.r.t both labels and nuisance variables.

---

## `residualize_in_folds_from_prep_combat_final(...) → list[dict]`

**Purpose:**  
Perform **ComBat harmonization** (removing site effects) and **OLS residualization** (removing confounds) **per cross-validation fold**, strictly on training data to avoid leakage.

**Detailed Steps:**
1. **Identify confounds:**
   - Splits confounds into *continuous* (`scan_age`, `income_continuous`, …) and *discrete* (`sex`, `missing_income`).
   - Detects the site column automatically (or via `site_col_name`).
2. **Prepare cross-validation (StratifiedKFold)**:
   Uses the `stratify_labels` built by `prepare_cv_data`.
3. **For each fold:**
   - Extract training/testing samples.
   - **ComBat harmonization:**
     - Fit ComBat only on training data:
       ```python
       combat.fit_transform(X_train, batch_train, discrete_covs, continuous_covs)
       ```
     - Transform the test set using the trained model.
     - Discrete covariates include the **prematurity label (`__y_preserve`)**, ensuring true signal is preserved.
   - **OLS residualization:**
     - Build a confound design matrix (excluding site and `__y_preserve`).
     - Fit either:
       - **Statsmodels OLS per dimension** (accurate but slow), or
       - **Sklearn LinearRegression** (multi-output, fast).
     - Compute residuals:  
       \( r = Y - \hat{Y}_{conf} \).
   - Store:
     - `X_train_resid`, `X_test_resid`, `y_train`, `y_test`
     - Corresponding subject IDs.

4. **Full-dataset residualization:**
   - Also returns one full harmonized and residualized dataset (`X_all_resid`) for visualization or global refitting.

**Outputs:**  
List of fold dictionaries plus one “full” entry:
```python
[
  {'X_train_resid': ..., 'y_train': ..., 'X_test_resid': ..., 'y_test': ...},
  ...
  {'X_all_resid': ..., 'y_all': ..., 'ids_all': ...}
]
```

**Why it matters:**  
This is the **core leak-safe harmonization** step — it ensures site and confound correction are always trained *only* on the training subset, never contaminating test data.

---

## `classify_with_resid(...) → dict`

**Purpose:**  
Train, tune, and evaluate the logistic regression classifier on the residualized embeddings.

**Steps:**
1. **Global tuning of `C`:**
   - Uses a light `GridSearchCV` over 7 logarithmic values (`10^-3` → `10^3`).
   - Evaluates via ROC-AUC on the *full residualized data* (slight leak tolerated).
2. **Cross-validated evaluation (OOF AUC):**
   - For each fold:
     - Fit LR on `X_train_resid, y_train`.
     - Predict probabilities on `X_test_resid`.
     - Compute ROC-AUC and accumulate scores across all folds.
   - Aggregate:
     - `auc_mean`, `auc_std` (per-fold stats)
     - `auc_oof` (AUC from concatenated OOF predictions)
3. **Permutation test:**
   - Shuffle `y_all` while keeping residualized folds fixed.
   - Repeat training/testing loop for `n_permutations` times.
   - Compute empirical null distribution of AUCs.
   - Derive:
     - `perm_pvalue` = one-sided p-value
     - `ci95_null` = 95th percentile of null AUCs
     - `auc_threshold_at_alpha` = AUC significance threshold
4. **Final model refit:**
   - Refit logistic regression on **all residualized data**.
   - Recover coefficients in the **original feature scale**:
     \[
     w_x = \frac{w_z}{\sigma}, \quad b_x = b_z - w_x^\top \mu
     \]
   - Store both standardized and de-standardized weights.

**Outputs:**
```python
{
  'auc_per_fold': [...],
  'auc_mean': float,
  'auc_std': float,
  'auc_oof': float,
  'perm_scores': [...],
  'perm_pvalue': float,
  'ci95_null': float,
  'auc_threshold_at_alpha': float,
  'fixed_C': float,
  'final_estimator': sklearn.Pipeline,
  'final_coef_original': np.array,
  'final_intercept_original': np.array,
}
```

**Why it matters:**  
This step quantifies both **predictive performance** and **statistical robustness**, distinguishing real signal from chance-level fluctuations through empirical permutation testing.

---

## Summary Dependency Graph

```
load_embeddings()       →  embeddings_df
set_confound_df()       →  confounds_df
prepare_cv_data()       →  merged data + stratify_labels
residualize_in_folds_from_prep_combat_final() →  harmonized + residualized folds
classify_with_resid()   →  tuned classifier + AUC + permutation stats
```

Together, these functions form a **leak-safe, site-harmonized, confound-controlled classification pipeline** for sulcal embedding–based prematurity prediction.




## Modeling Choices

- **ComBat preserves `y`**: Adds `__y_preserve` to discrete covariates.
- **Train-only fitting**: ComBat and OLS are fit only on training folds.
- **Simple linear model**: L2 logistic regression with modest grid over `C`.
- **Permutation test**: Empirical p-values and null thresholds.

---

## Configuration Knobs

- **Task**: `thresholds = ["28-32"]` (optionally include `"32-37"`).
- **Dimension regex**: `dim_regex=r'^dim'`.
- **CV**: `n_splits=5`, adaptive stratification.
- **Residualization backend**: `use_statsmodels=True` (exact OLS) vs `False` (faster multi-output).
- **ComBat**: `use_combat=True`.

---

## Common Pitfalls

- **Site detection**: Supply `site_col_name` if auto-detection fails.
- **Memory pressure**: Tune `n_jobs_outer` to fit CPU/RAM budgets.
- **CSV serialization**: Parse stringified lists/arrays with:
  ```python
  import ast
  df['auc_per_fold'] = df['auc_per_fold'].apply(ast.literal_eval)
  ```

---

## Minimal Example

```python
embeddings_df = load_embeddings("STs_right")
confounds_df  = set_confound_df()
preps         = prepare_cv_data(embeddings_df, labels_df, confounds_df, threshold="28-32")
folds_resids  = residualize_in_folds_from_prep_combat_final(preps, use_combat=True, use_statsmodels=True)
results       = classify_with_resid(folds_resids, n_jobs=-1, n_permutations=1000)

print("Mean AUC:", results["auc_mean"], "Perm p-value:", results["perm_pvalue"])
```

---

## Optional: Fold Balance Plots

Use `plot_fold_distributions(...)` to compare train vs test distributions (e.g., `scan_age`, `sex`, `site`, `prematurity`, `income`) per fold.

---

## Reproducibility Notes

- Fixed `random_state` for CV and models.
- ComBat + OLS are refit per fold on train only.
- Results are appended incrementally with `flush()`, enabling safe resume after interruptions.
