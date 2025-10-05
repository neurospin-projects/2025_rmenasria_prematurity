# 🧩 Cognitive Prediction and Prematurity Analysis — README

This document summarizes the workflow, methodology, and purpose of three complementary Jupyter notebooks analyzing the relationship between **brain morphology**, **cognitive performance**, and **prematurity** using ABCD cohort embeddings.

---

## 🧠 1. `cognition_premas.ipynb` — Cognitive Descriptive Analysis

### 🎯 Objective
Explore **basic associations** between cognitive test scores and **gestational age** or **prematurity categories**, before introducing embeddings or model-based predictions.

### ⚙️ Workflow
1. **Load and merge data**
   - Cognitive scores (`nc_y_nihtb.csv`)
   - Prematurity classes (`prematurity_labels_true_classes.csv`)
   - Subject-level metadata (sex, gestational age, etc.)
2. **Compute statistics**
   - Pearson and Spearman correlations between gestational age (`gest_age`) and each cognitive score.
   - Welch’s t-test and **Cohen’s d** between preterm and full-term groups.
3. **Visualize**
   - Histograms of cognitive scores by prematurity class and sex.
   - Heatmaps showing correlation strengths and effect sizes.

### 🧩 Outputs
- **CSV:** `prematurity_stats_simple.csv`  
  Columns:  
  `score`, `pearson_r_ga_pre`, `pearson_r_ga_smallpre`, `cohen_d_pre_vs_term`, etc.
- **Plots:**
  - Heatmap of correlations by gestational group.
  - Heatmap of Cohen’s d across cognitive scores.
  - Histograms of score distributions across prematurity classes.

### 🧠 Interpretation
- Quantifies how **cognitive performance** scales with **gestational age**.
- Identifies which cognitive domains are most affected by prematurity.
- Serves as a **baseline layer** for higher-level modeling.

---

## 🧬 2. `cognition_predictions.ipynb` — Morphology-to-Cognition Associations

### 🎯 Objective
Assess whether **morphological embedding–based prematurity probabilities** correlate with **cognitive outcomes**, within specific prematurity subgroups.

### ⚙️ Workflow
1. **Load data**
   - Cognitive + gestational dataset (`df_with_cognition_and_ages.csv`)
   - Add per-region classification outputs (`*_confidence.csv`) for each region (prematurity probability).
2. **Compute Spearman correlations**
   - Between `region_confidence` (from classifier) and cognitive test scores.
   - Done separately for each prematurity group:  
     - 28–32 weeks  
     - 32–37 weeks
3. **Export and visualize**
   - Results stored as `prematurity_pearson_28_32.csv` and `prematurity_pearson_32_37.csv`.
   - Heatmaps of Spearman r-values (region × cognitive score).

### 🧩 Outputs
| File | Group | Description |
|------|--------|-------------|
| `prematurity_pearson_28_32.csv` | 28–32 weeks | Correlations between confidence and cognitive scores |
| `prematurity_pearson_32_37.csv` | 32–37 weeks | Same for moderate preterms |

### 📊 Interpretation
- Positive r-values indicate that **higher morphometric prematurity probability** relates to **lower cognitive performance**.
- Highlights regions where morphological classification confidence is **predictive of cognition**.
- Establishes a **region-wise morphology–function link** within preterm subgroups.

---

## 🔁 3. `r2cog_bootstrap.ipynb` — Bootstrapped Cognitive Prediction

### 🎯 Objective
Quantify how much **variance in cognitive performance** is explained by regional brain embeddings, using **bootstrapped Ridge regressions**.

### ⚙️ Workflow
1. **Prepare data**
   - Load regional embeddings (`.csv` per region).
   - Merge with cognitive and prematurity labels.
2. **Bootstrap regression**
   - Ridge regression (with CV-based alpha tuning).
   - Repeated bootstrap resampling (`n_boot=750`).
   - For each iteration:
     - Train model on bootstrap sample.
     - Evaluate R² on out-of-bag samples.
3. **Aggregate results**
   - Compute mean R² and 95% confidence intervals.
   - Export results for each region and cognitive score.

### 🧩 Outputs
| File | Subgroup | Description |
|------|-----------|-------------|
| `r2boostrap_prema_all_regions.csv` | 28–37 weeks | Mean and CI of R² values |
| `r2boostrap_terms_all_regions.csv` | ≥37 weeks | Term-born controls |
| `r2boostrap_extrprema_all_regions.csv` | <28 weeks | Extremely preterm group |

Each file contains:
- `region`, `score`, `n`, `r2_mean`, and `r2_ci`.

