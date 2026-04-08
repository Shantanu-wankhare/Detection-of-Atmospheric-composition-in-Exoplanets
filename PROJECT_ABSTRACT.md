# CS6140 Machine Learning – Spring 2026 – Northeastern University San Jose
## Project Title: Exploring ML Models for Detection of Atmospheric Composition in Exoplanets

**Team Members**
- Shantanu Wankhare — wankhare.s@northeastern.edu
- Bhalchandra Shinde — shinde.b@northeastern.edu
- Asad Mulani — mulani.a@northeastern.edu

**Dataset**: INARA ATMOS (NASA FDL / Zorzan et al. 2025)

---

## Overview

This project applies machine learning to exoplanet atmospheric retrieval — the task of inferring the chemical composition of a planet's atmosphere from its climate state. Traditional retrieval methods based on Bayesian sampling are computationally expensive, often requiring hours per planet. As space missions such as the James Webb Space Telescope (JWST) and the planned Habitable Worlds Observatory (HWO) generate data at unprecedented scale, fast and automated retrieval is a scientific necessity.

We trained two models on the INARA ATMOS dataset — a per-molecule Random Forest (baseline) and a 1D ResNet — to predict the log₁₀ surface volume mixing ratios of 12 atmospheric molecules from simulated climate profiles. Both models infer all 12 abundances in milliseconds. The deep model achieves a mean test R² of **0.44** across all molecules on the 20,000-sample run, rising to **0.99** on the full 124,000-sample dataset.

---

## Dataset

The **INARA ATMOS** dataset (Zorzan et al. 2025) contains synthetic rocky planet atmosphere simulations generated with the CLIMA photochemical-climate model, designed as a machine learning training resource. The full archive contains approximately **124,000 samples** across ten tar archives. We ran experiments at two scales: 20,000 samples (development) and the full 124,320-sample dataset.

**Input — CLIMA profiles**: Each sample is a climate profile of 12 atmospheric variables measured at **101 altitude levels** (surface to top of atmosphere), forming a `(12, 101)` tensor:

| Channel | Variable | Physical meaning |
|---|---|---|
| 0 | `J` | Actinic radiative flux |
| 1 | `P` | Pressure |
| 2 | `ALT` | Altitude (km) |
| 3 | `T` | Temperature |
| 4 | `CONVEC` | Convective flux |
| 5 | `DT` | Temperature tendency |
| 6 | `TOLD` | Previous temperature |
| 7 | `FH2O` | Water vapour flux |
| 8 | `FSAVE` | Saved radiative flux |
| 9 | `FO3` | Ozone flux |
| 10 | `TCOOL` | Radiative cooling rate |
| 11 | `THEAT` | Radiative heating rate |

**Targets — molecular surface abundances**: log₁₀ volume mixing ratios of 12 molecules at the surface layer. Values range from ~0 (major constituent) to −40 (detection floor).

**Processing**: Raw per-sample `.npy.npz` files are extracted from the tar archives by `process_inara.py` and saved as:
- `spectra.npy` — `(N, 12, 101)` CLIMA profiles
- `molecules.npy` — `(N, 12)` log₁₀ surface abundances
- `aux_params.npy` — `(N, 11)` metadata scalars (input fluxes, conditions)
- `wavelengths.npy` — `(101,)` altitude axis in km

**Data split**: 70% train / 15% validation / 15% test (fixed seed 42).

---

## Models

### Baseline — Per-Molecule Random Forest

Each of the 12 molecules gets its own `RandomForestRegressor` trained on PCA-reduced CLIMA features:

- Flatten `(12, 101)` → `(1,212,)`, reduce with PCA to **300 components** (whiten=True)
- Molecule-specific hyperparameters: `n_estimators` 150–400, `max_depth` 10–22
- Trace/volatile molecules (SO₂, NH₃, H₂S) use deeper trees and more estimators

### Deep Model — 1D SpectralResNet (5.03M parameters)

A 1D ResNet backbone with 12 per-molecule output heads operating directly on the `(12, 101)` CLIMA profile:

```
Input  : (B, 12, 101)  — 12 CLIMA channels × 101 altitude levels
Stem   : Conv1d(12→64, k=11, s=1) + BN + ReLU
Stage 1: 2 × ResBlock(64→64,   stride=1)
Stage 2: 2 × ResBlock(64→128,  stride=2)
Stage 3: 2 × ResBlock(128→256, stride=2)
Stage 4: 2 × ResBlock(256→512, stride=2)
Pool   : AdaptiveAvgPool1d(1)  → (B, 512)
Shared : Dropout(0.3) + FC(512→256) + LayerNorm + ReLU
Heads  : 12 × molecule-specific MLP → scalar log₁₀ abundance
```

Per-molecule heads are tuned for prediction difficulty: trace molecules (SO₂, NH₃, H₂S) get deeper heads `[256, 128, 64]` and higher dropout (0.35–0.40); bulk molecules (CO₂, O₂, N₂) use shallower heads `[128, 64]`.

Training: AdamW (lr=1e-3, weight_decay=1e-4), CosineAnnealingLR, WeightedMSE loss with per-molecule importance weights, early stopping (patience=30), Gaussian noise augmentation (std=0.01).

---

## Results

### 20,000-sample run (development)

| Molecule | Role | RF R² | ResNet R² | ΔR² |
|---|---|---|---|---|
| H₂O | Habitability | 0.367 | 0.630 | +0.263 |
| CO₂ | Greenhouse | 0.326 | 0.601 | +0.274 |
| O₂ | Biosignature | 0.599 | 0.594 | −0.005 |
| **O₃** | **Biosignature** | 0.755 | **0.895** | +0.141 |
| CH₄ | Biosignature | 0.479 | 0.698 | +0.219 |
| N₂ | Bulk filler | −0.042 | 0.022 | +0.064 |
| N₂O | Biosignature | 0.213 | 0.114 | −0.100 |
| CO | Photochemical | 0.046 | 0.001 | −0.046 |
| H₂ | Primordial | 0.201 | 0.019 | −0.182 |
| **H₂S** | Volcanic | 0.546 | **0.816** | +0.270 |
| SO₂ | Volcanic | 0.220 | 0.354 | +0.134 |
| NH₃ | Nitrogen cycle | 0.411 | 0.565 | +0.155 |
| **MEAN** | | **0.343** | **0.442** | **+0.099** |

### Full 124,320-sample run

| | RF mean R² | ResNet mean R² |
|---|---|---|
| All 12 molecules | 0.966 | 0.992 |

**Key observations:**
- O₃ and H₂S are the best-predicted molecules at both scales
- N₂, CO, H₂ are hardest — low variance and weak climate signal
- **NH₃ is at the log-floor (−40) for 100% of test samples** — the INARA ATMOS simulations produce no NH₃, making it unlearnable regardless of model; it is retained as an output slot for schema compatibility
- The ResNet outperforms RF on 8/12 molecules; RF is competitive on O₂, CO, H₂ where the deep model also struggles
- Full dataset dramatically improves both models, confirming the task is data-limited at 20K samples

---

## Evaluation

Models are evaluated on a held-out test set (15%) using per-molecule:
- **R²** — fraction of log₁₀ abundance variance explained
- **RMSE** — in log₁₀ units (1 unit = one order of magnitude)
- **MAE** — robust to outliers

---

## Deliverables

1. **EDA notebook** (`notebooks/eda.ipynb`) — statistical analysis of CLIMA profiles, molecular distributions, correlation structure, and dataset split validation
2. **Visualisation notebook** (`notebooks/visualize.ipynb`) — model loading, training curve, R² comparison, scatter plots, residual distributions, RF feature importance, ResNet architecture breakdown
3. **Training scripts** — `run_baseline.py`, `run_deep_model.py` with configurable data scale and hyperparameters
4. **Processing pipeline** — `process_inara.py` for extracting CLIMA profiles from raw INARA tar archives at any sample count (5K–124K)
5. **Saved models** — `models/processed/baseline_rf.joblib`, `models/processed/spectral_resnet.pt`
6. **Final report** — background, methodology, results, and discussion

---

## References

1. Zorzan et al. (2025) — *ApJS* 277:38 — INARA dataset and baseline retrieval model
2. Márquez-Neila, P. et al. (2018) — *Nature Astronomy* 2 — random forest atmospheric retrieval
3. Vasist, M. et al. (2023) — *A&A* — neural posterior estimation with normalising flows
4. Gebhard, T. et al. (2024) — *A&A* — flow matching for full posterior atmospheric retrieval
5. JWST Transiting Exoplanet Community ERS Team (2023) — *Nature* 614:649 — WASP-39 b benchmark

- **Channel 0**: Observed SNR spectrum (star+planet / noise), normalised per spectrum
- **Channel 1**: Mean-subtracted SNR, removes the stellar continuum and isolates molecular absorption features
- **Channel 2**: Log transit depth 

The 12 target molecules span habitability indicators (H₂O, CO₂), biosignatures (O₂, O₃, CH₄, N₂O), and geochemical tracers (CO, H₂S, SO₂, NH₃, N₂, H₂). Four stellar types (F, G, K, M) are represented, with eight auxiliary features per system including stellar temperature, planetary radius, surface gravity, and orbital distance.

---

## Methodology

### Model 
The preferred models would be Random Forest(Baseline) and further XGBoost or 1D CNN. We would still want to do source data analysis and EDA to finalize the models being used. We can try to use some baseline model and further some other model as well. This can be confirmed once the project progresses and completes the EDA and feature engineering steps. 


### Source data - INARA Dataset

The [INARA dataset](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/FDL/nph-fdl) (Zorzan et al. 2025) contains 3.1 million synthetic rocky planet spectra generated with NASA's Planetary Spectrum Generator, designed specifically for machine learning retrieval.

**CSV format** (confirmed from sample files):

| Column | Description |
|---|---|
| `wavelength_(um)` | Wavelength in µm |
| `star_planet_signal_(erg/s/cm2)` | Combined star + planet observation |
| `noise_(erg/s/cm2)` | Instrument noise floor |
| `stellar_signal_(erg/s/cm2)` | Stellar flux only |
| `planet_signal_(erg/s/cm2)` | Atmospheric signal only |

- **Total rows per file**: 15,346 (full PSG output, 0.2–639.9 µm)
- **Usable rows**: 4,378 (where `noise > 0`, covering 0.2–2.0 µm — HWO/LUVOIR instrument range)

We would read all the csv files (like 0001108.csv) and get the data preprocessed and ready into the below files as processed input data:
1. spectra.npy
2. molecules.npy
3. aux_params.npy
4. wavelengths.npy

### Expected Targets/Outcomes — 

We would target most of the molecules from below:

| Molecule | Scientific role |
|---|---|
| H₂O | Habitability indicator — strong NIR bands |
| CO₂ | Greenhouse gas |
| **O₂** | **Biosignature** — 0.76 µm A-band |
| **O₃** | **Biosignature** — UV Hartley band |
| **CH₄** | **Biosignature** when co-present with O₂ |
| N₂ | Bulk atmospheric filler |
| **N₂O** | **Biosignature** — microbial source |
| CO | Volcanic / photochemical indicator |
| H₂ | Primordial outgassing |
| H₂S | Volcanic outgassing |
| SO₂ | Volcanic activity |
| NH₃ | Nitrogen cycle |

The output can be a vector of 12 values.

---

## Evaluation Plan

The model is evaluated on a held-out test set (15% of data) using:

- **RMSE and MAE** per molecule 
- **MAE** per Molecule
- **R²** per molecule — measures fraction of variance explained
- **Calibration error** — mean absolute deviation between empirical and nominal coverage across confidence levels
correlation


## Deliverables:
1.	Notebook 1: EDA Jupyter notebook with statistical analysis and visualisations of INARA dataset
2.	Codebase: Model training and evaluation codebase 
3.	Results: Experimental results: per-molecule RMSE/R² tables, calibration curves, scatter plots
4.	Validation: Domain transfer analysis: predictions on real JWST spectra vs. published retrievals
5.	Report: Final project report covering background, methodology, results, and discussion


---

## References

1. Zorzan et al. (2025) — *ApJS* 277:38 — INARA dataset and baseline 1D CNN retrieval model
2. Márquez-Neila, P. et al. (2018) — *Nature Astronomy* 2 — random forest atmospheric retrieval (seminal ML retrieval work)
3. Vasist, M. et al. (2023) — *A&A* — neural posterior estimation with normalising flows
4. Gebhard, T. et al. (2024) — *A&A* — flow matching for full posterior atmospheric retrieval (current state of the art)
5. Gal, Y. & Ghahramani, Z. (2016) — *ICML* — MC Dropout as approximate Bayesian inference
6. JWST Transiting Exoplanet Community ERS Team (2023) — *Nature* 614:649 — WASP-39 b benchmark
