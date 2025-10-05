# 🧠 Cognitive Regression Pipeline (ABCD – Ridge Model)

This script extends the **prematurity classification pipeline** to predict **cognitive scores** from regional sulcal embeddings, while carefully controlling for confounds and site effects.

For a full methodological explanation of data preparation, ComBat harmonization, and residualization, **refer to the main README of the prematurity pipeline**.

---

## 🔍 Purpose

Instead of classifying preterm vs term subjects, this version **regresses cognitive scores** (e.g. *NIH Toolbox* measures such as `nihtbx_picvocab_agecorrected`, `nihtbx_totalcomp_agecorrected`, etc.) across regions of the ABCD dataset.

The core steps remain the same as in the prematurity workflow.

---

## ⚙️ Key Differences from the Prematurity Pipeline

| Aspect | Prematurity pipeline | Cognitive regression variant |
|:-------|:----------------------|:------------------------------|
| **Target (`y`)** | Binary (preterm vs control) | Continuous (cognitive score) |
| **Model** | Logistic / SVM classifier | Ridge regression (`R²`, Pearson, Spearman) |
| **Evaluation** | AUC, permutation p-values | R², correlations, confidence intervals |
| **Residualization** | ComBat + OLS per fold | Same, but validated with cognition data |
| **Outputs** | Classification metrics | Regression metrics with bootstrap & permutation tests |

---

## 📦 Outputs

- `ABCD_cog_regression_terms.csv` → region × cognitive score performance (R², r, CI, p-values).  
- `ABCD_cog_directions_terms.csv` → fitted regression weights for interpretability.  
- Optional heatmaps for top-performing regions/tests.

---

## 🧩 Key Functions

- `prepare_cv_data()` → data assembly and stratification (same as in prematurity pipeline).  
- `residualize_in_folds_from_prep_combat_final()` → cross-validated ComBat + OLS residualization.  
- `regress_cognition_with_resid()` → regression, bootstrap CI, and permutation-based p-values.  
- `process_combo_cognition(region, score)` → runs one full (region, cognitive score) combination.  

---

## 🚀 Execution

All `(region, score)` pairs are processed in parallel using `ProcessPoolExecutor`, with progress tracking via `tqdm`.

Results are saved incrementally to avoid recomputation on crash or interruption.

---

## 📈 Visualization

A dedicated plotting section builds **heatmaps** of top 20–30 `(region, test)` results, annotated with:
- `r_oof [95% CI]`
- `R²`
- `*` if significant (`p < 0.001`, Bonferroni-corrected)

---

## 🔗 See Also

Refer to the main **Prematurity Pipeline README** for:
- detailed explanation of the preprocessing, confound handling, and harmonization logic,
- function-level documentation shared between both pipelines.

---
