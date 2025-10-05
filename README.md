# 🧠 2025_rmenasria_prematurity -

This repository contains all **notebooks for analysis and visualization** developed as part of the *Detection of cortical folding abnormalities
associated with extreme prematurity* project.  
The analyses are based on **latent embeddings** derived from the *Champollion V1* model and cover multiple aspects of the study: pattern visualization, cognitive scores regression, database exploration, prematurity classification, and morphological shift analysis.

---

## 📁 Repository structure

The parent folder includes **five main subdirectories**, each corresponding to a thematic or methodological component of the project.

---

### 🧩 1. `anatomist/` — Visualizations

Contains all scripts and outputs related to **Anatomist**, used for:
- visualization of **average and individual sulcal morphologies**,  
- **Quality Check (QC)** of sulcal graphs and cortical surfaces,  
- **whole-brain visualizations** of modeling results (e.g., shifts, prediction maps),  
- **volume and surface renderings** of averaged results or group-level effects.

**Purpose:** visual inspection and high-quality rendering of morphological results.

---

### 🧠 2. `cognition/` — Cognitive Score Analyses

Includes statistical and regression analyses linking **latent embeddings** to **cognitive performance**:
- linear regressions on cognitive scores in **ABCD** and later **DHCP**,  
- exploratory statistical analyses (correlations, region-wise effects),  
- directional regression analyses in latent space.

**Goal:** identify latent regions associated with cognition and their modulation by prematurity.

---

### 📊 3. `data_analysis/` — Database Exploration

Gathers **descriptive and exploratory analyses** across the available datasets:
- statistical summaries and data visualization for **ABCD**,  
- embedding distribution analysis,  
- **selection of top-performing regions** using a composite score,  


**Goal:** provide a statistical and visual overview of the input data prior to modeling.

---

### 👶 4. `prematurity/` — Modeling Prematurity

The **core part** of the project, dedicated to modeling and visualization of the relationship between **prematurity** and **cortical folding**.  
This folder includes several complementary analytical pipelines:

#### Sub-sections:
- **Latent clustering:** grouping subjects based on latent embeddings (with or without fixed cluster size) to explore morphological patterns linked to prematurity.  
- **Linear modeling:**
  - on **ABCD**: both whole-brain and region-wise analyses,  
  - on **DHCP**: prematurity classification and gestational age regression.  
- **Site effect analysis:** early site-based investigation (pre-residualization).  
- **UMAP:** large-scale UMAP projections across datasets (**DHCP**, **ABCD**, **UKB**) for unsupervised exploration of latent space structure.

**Goal:** characterize morphological signatures of prematurity at local, global, and cross-dataset levels.

---

### 🔄 5. `shifts/` — Morphological Shift Analysis

Contains computations and analyses related to **morphological shift values**, quantifying differences between groups:
- **regional** and **sulcal-level** shift measurements,  
- group comparisons across different age brackets,  
- **correlations** between shift magnitudes and classification/regression performance.

**Goal:** quantify structural displacements in latent space and link them to observed between-group differences.

---

## ⚙️ Data and environment

- All notebooks use embeddings derived from the **Champollion V1** model.  
- The datasets analyzed include **ABCD**, **DHCP**, and **UKB**.  
- The analyses were conducted in **Python 3.11+**, with key dependencies gathered in the /neurospin/dico/rmenasria/Runs/03_main/Program/main_pixi/pixi.toml associated environment.

---



## ✍️ Author and context

This work was conducted by **Racim Menasria**  
within the *Detection of cortical folding abnormalities
associated with extreme prematurity* master's project at **NeuroSpin (CEA, France)**, 2025.

The analyses aim to uncover morphological signatures of prematurity through **deep latent representations** of cortical folding patterns.
