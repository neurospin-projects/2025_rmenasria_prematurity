# 🌍 Cross-Cohort Latent Space Visualization (UMAP – dHCP / ABCD / UKB)

> **Goal:**  
> Visualize and compare sulcal latent embeddings across **newborns (dHCP)**, **children (ABCD)**, and **adults (UKB)** using **UMAP projections** of deep folding embeddings.  
> The same UMAP model is trained on adult embeddings and used to project the other populations for a cross-age visualization of cortical folding spaces.

---

## 🧠 Overview

This pipeline:
- Loads precomputed **sulcal embeddings** (latent representations of cortical folds).
- Merges with demographic labels (`birth_age`, `gest_age`).
- Applies **StandardScaler** normalization and **UMAP (2D)** reduction.
- Projects **dHCP** (newborns) and **ABCD** (children) embeddings into the **UKB** (adult) embedding space.
- Visualizes the shared 2D manifold colored by gestational age.

---

## 📂 Input Data

| Dataset | Description | Example Path |
|----------|--------------|---------------|
| **dHCP** | Newborn embeddings & metadata | `/neurospin/dico/data/deep_folding/.../dHCP_random_embeddings/full_embeddings.csv` |
| **ABCD** | Child embeddings & prematurity labels | `/neurospin/dico/.../embeddings/ABCD_embeddings/` |
| **UKB**  | Adult embeddings | `/neurospin/dico/.../ukb40_random_embeddings/full_embeddings.csv` |

Each embedding file contains:
- Columns `dim1`, `dim2`, …, `dimN` (latent features).
- Subject IDs (`ID` or `Subject`).

---

## ⚙️ Workflow

1. **Load embeddings**
   - Read embeddings for dHCP, ABCD, and UKB for a given region (e.g., `STs_right`).
   - Clean subject IDs for consistent merging.

2. **Merge with labels**
   - Attach `birth_age` or `gest_age` from corresponding metadata files.
   - Filter out invalid or missing entries.

3. **Normalize**
   - Apply `StandardScaler` to all embeddings to standardize latent dimensions.

4. **Train UMAP**
   - Fit UMAP (`n_components=2`) on **UKB adult embeddings**.
   - Transform both **dHCP** and **ABCD** embeddings into this space.

5. **Visualization**
   - Create 2D scatterplots:
     - **dHCP**: colored by birth age (`viridis` colormap).
     - **ABCD**: colored by gestational age (`plasma` colormap).
     - **UKB**: plotted as light-gray density map (optionally with hexbin).

6. **Multi-Region Visualization**
   - Automatically loops over all regions (left/right hemispheres).
   - Generates a grid of subplots (rows = regions, columns = populations).
   - Saves figure as `umap_grid_left_latents32.png`.

---

## 📊 Example Figure

Each figure row corresponds to one sulcus:
| dHCP (newborns) | ABCD (children) | UKB (adults – density) |
|:----------------:|:----------------:|:----------------------:|
| colored by birth age | colored by gestational age | population density in UMAP space |

---

## 🧩 Function Highlights

### `visualize_embeddings_babies(region, latent_size)`
- Loads embeddings for one region.
- Trains UMAP on UKB adults.
- Projects and plots dHCP + ABCD embeddings into the same 2D space.

### `visualize_embeddings_babies_ax(region, latent_size, ax_baby, ax_child, ax_adult)`
- Same as above but draws directly into given Matplotlib axes (used for grid visualization).

### `get_region_list(base_path)`
- Returns all available regions from model directories.

---

## 🖼️ Output

- Individual region UMAPs (interactive display).
- Multi-region grid figure:



- Optional: adapt latent size (e.g., 32 or 256) via function parameter.

---

## 🧮 Dependencies

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `umap-learn`
- `scikit-learn`

---

## 🚀 Usage Example

```python
# Visualize one region
visualize_embeddings_babies(region="STs_right", latent_size=32)

# Generate a full grid across regions
region_list = get_region_list("/neurospin/dico/.../Champollion_V1_after_ablation")
for region in region_list:
  visualize_embeddings_babies(region, latent_size=32)
