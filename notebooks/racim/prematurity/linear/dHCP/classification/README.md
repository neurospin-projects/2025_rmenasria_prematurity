# 🧠 Prematurity Classification on the dHCP Database

> **Goal:**  
> Perform region-wise classification of **premature birth** using sulcal **embeddings** from the dHCP database, with confound control (sex residualization) and **permutation-based significance testing**.  

This pipeline runs **logistic regression classifiers** on each sulcal embedding, controlling for sex effects and computing **cross-validated AUCs** with permutation testing.  
It generates:
- Regional performance summaries (`AUC_mean`, `p-values`)
- Linear classification directions (coefficients)
- CSV reports for visualization or meta-analysis.

---

## 🧩 Overview

### Pipeline Steps
1. **Load inputs**
   - Embeddings per region (from `Champollion_V1_after_ablation`)
   - Participant metadata (birth age, sex)
   - Compute prematurity class labels.

2. **Preprocessing**
   - Normalize IDs (`sub-xxx` → `xxx`).
   - Derive prematurity classes:  
     `<28`, `28–32`, `32–37`, `≥37`.
   - Encode sex numerically (`male=0`, `female=1`).

3. **Residualization**
   - Remove the **sex effect** from embeddings using a custom transformer `ResidualizerSexeFromX`.
   - Ensures classifier cannot exploit sex imbalance.

4. **Classification**
   - 5-fold stratified cross-validation.
   - Logistic regression (L2 penalty) with hyperparameter grid over `C ∈ {0.01, 0.1, 1, 10}`.
   - Compute AUC per fold and average.

5. **Permutation test**
   - Estimate significance of the AUC via label permutation (`n_perm ≈ 10k+`).
   - Compute empirical p-value and null 95% confidence interval.

6. **Outputs**
   - Performance summary (`AUC_mean`, `perm_pval`).
   - Coefficients (interpretable sulcal direction).
   - Region-by-region CSV exports and visualization tables.

---

## 📂 Inputs

| Type | Description | Path Example |
|------|--------------|---------------|
| **Embeddings** | Regional sulcal latent embeddings | `/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation/FCMpost-SpC_right/.../full_embeddings.csv` |
| **Participants** | Metadata including `birth_age` | `/neurospin/dico/data/deep_folding/current/datasets/dHCP_374_subjects/participants.csv` |
| **Cognition/Sex** | Participant sex information | `/neurospin/dico/rmenasria/Runs/03_main/Input/dHCP/cognitive_scores_with_age_dHCP.csv` |

---

## 📈 Output Files

| File | Description |
|------|--------------|
| `classif_prematurity_dHCP_final_<THR>.csv` | Summary of regional AUCs and permutation p-values |
| `classif_prematurity_dHCP_directions_final_<THR>.csv` | Classifier coefficients (direction vectors) |
| `prematurity_AUC_by_region_<date>.csv` | Pivoted summary per region × threshold |
| `prematurity_thresholded_auc_<THR>.csv` | Thresholded AUCs (Bonferroni-corrected significance) |

---

## ⚙️ Key Components

### `prepare_dhcp_for_prematurity()`
> Merges embeddings, sex, and labels; builds the dataset for classification.

- Filters subjects to the two target classes (`prem_class`).
- Concatenates sex as last column of `X` for later residualization.
- Adapts the number of folds to the minority class count.

**Returns:**  
A `PreparedData` dataclass with:
- `X` (embeddings + sex)
- `y` (binary prematurity label)
- `groups` (subject IDs)
- `embedding_cols`
- `n_splits`

---

### `ResidualizerSexeFromX`
> Custom transformer removing sex effects per embedding dimension via **closed-form OLS**.

- Fits `x_j ~ α_j + β_j * sex`
- Returns residuals `x_j - (α_j + β_j * sex)`
- Applied fold-wise within the sklearn pipeline.

---

### `classify_prematurity_with_perm()`
> Trains the logistic model with cross-validation and computes permutation-based significance.

Steps:
1. Cross-validation tuning of `C`.
2. Evaluate best model via 5-fold CV.
3. Compute permutation null AUC distribution.
4. Extract:
   - `AUC_mean`, `AUC_std`, `perm_pval`, `perm_ci95`
   - Unscaled coefficients and intercept for interpretability.

---

### `prem_direction()`
> Projects subjects along the learned classification axis.

- Fits the pipeline on full data (with residualization and scaling).
- Computes predicted scores (`projection`).
- Returns sorted dataframe of subjects with projection values (prematurity direction).

---

### `run_regions_prematurity()`
> Loops over all sulcal regions and runs the full pipeline.

- Executes:  
  `prepare_dhcp_for_prematurity()` → `classify_prematurity_with_perm()`
- Handles failures gracefully.
- Writes performance and coefficient CSVs.

**Outputs:**
- Summary CSV (AUCs, p-values)
- Coefficient CSV (direction vectors)

---

## ⚙️ Configuration Example

```python
regions = get_region_list(base_path)
prem_filter = ["<28", ">=37"]
prem_target = "<28"

out = run_regions_prematurity(
    regions=regions,
    labels_path=labels_path,
    prem_class=prem_filter,
    prem_target=prem_target,
    Cs=[0.01, 0.1, 1, 10],
    n_perm=11200,
    n_jobs=-1,
    output_csv="/neurospin/dico/rmenasria/Runs/03_main/Output/final/classif_prematurity_dHCP_final_28.csv",
    output_coef_csv="/neurospin/dico/rmenasria/Runs/03_main/Output/final/classif_prematurity_dHCP_directions_final_28.csv"
)
