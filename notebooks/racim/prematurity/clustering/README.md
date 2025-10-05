# 🧩 Clustering of Prematurity Patterns in ABCD Embeddings

> **Goal:**  
> Identify **clusters of subjects** within ABCD sulcal embeddings that show **enrichment in preterm birth**.  
> The script combines K-means clustering, balanced cluster construction, and **statistical enrichment testing** (binomial and chi-square) to detect latent phenotypic structure.

---

## 📘 Overview

### Concept
- The ABCD cohort includes thousands of subjects with sulcal embeddings.
- Subjects are classified by **prematurity class** (`<28`, `28–32`, `32–37`, `≥37`).
- The script:
  1. Loads embeddings and merges with prematurity metadata.
  2. Applies **K-means clustering** on embeddings.
  3. Builds **equal-size clusters** (250 subjects each) with overlap.
  4. Estimates **preterm rate per cluster**.
  5. Tests for **statistical enrichment** using:
     - **Binomial test** (per cluster)
     - **Chi² test** (global contingency)

---

## 🧩 Data Inputs

| Type | Description | Example Path |
|------|--------------|--------------|
| **Embeddings** | Sulcal embedding vectors (`dim1…dimN`) | `/neurospin/dico/.../embeddings/ABCD_embeddings/STs_right.csv` |
| **Labels** | Prematurity classes per subject | `/neurospin/dico/.../prematurity_labels_true_classes.csv` |

### Key Columns
- `src_subject_id` : unique subject identifier  
- `prem_class` : categorical prematurity group  
- `dim1`, `dim2`, … : embedding dimensions  

---

## ⚙️ Parameters

| Variable | Description | Example |
|-----------|-------------|----------|
| `region` | Brain sulcus or region | `"STs_right"` |
| `thresholds` | Prematurity groups to include | `["<28", "28-32", "32-37"]` |
| `n_clusters` | Number of K-means clusters | `40` |
| `n_target` | Target number of subjects per cluster | `250` |

---

## 🔄 Workflow

1. **Load and merge data** — Read regional embeddings and merge with prematurity labels using cleaned subject IDs.  
2. **Prepare targets** — Create a binary label `y` for prematurity (`1` = preterm, `0` = term).  
3. **K-Means clustering** — Cluster embeddings into `n_clusters` (e.g., 40).  
4. **Fix cluster sizes** — For each cluster, keep the 250 closest subjects to the centroid; fill any deficit with nearest global points to ensure equal cluster sizes.  
5. **Compute cluster statistics** — For each cluster, compute the number and proportion of preterm subjects (`preterm_rate`).  
6. **Visualize** — Plot the preterm rate per cluster as a barplot to identify enriched or depleted clusters.  
7. **Statistical testing** — Apply binomial and chi² tests to detect clusters with significantly higher or lower preterm proportions.  
8. **Export results** — Save ranked cluster assignments and summary statistics for downstream visualization or reporting.

