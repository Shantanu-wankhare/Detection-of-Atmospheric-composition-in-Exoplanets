# CS6140 Machine Learning – Spring 2026 - Northeastern University San Jose
## Project title: Exploring ML Models for detection of Atmospheric composition in Exoplanets

Team Members 
•	Shantanu Wankhare	wankhare.s@northeastern.edu
•	Bhalchandra Shinde 	shinde.b@northeastern.edu
•	Asad Mulani 		mulani.a@northeastern.edu

## 1. Project Overview

**Task:** Given a simulated exoplanet's atmospheric CLIMA profile (12 variables × 101 altitude levels), predict the log₁₀ surface volume mixing ratios of 12 molecular species.

**Models:**

| Model | Description | Training samples |
|---|---|---|
| Random Forest (baseline) | Per-molecule RF on PCA-reduced features | ≤ 10,000 (always capped) |
| SpectralResNet (deep) | 1D ResNet + per-molecule output heads | Full training split |

**Molecules predicted:** H₂O · CO₂ · O₂ · O₃ · CH₄ · N₂ · N₂O · CO · H₂ · H₂S · SO₂ · NH₃

---
## 2. Dataset

**Source:** INARA ATMOS — synthetic exoplanet atmospheric simulations  
**Full size:** ~124,314 samples · ~575 MB processed

| Array | Shape | Description |
|---|---|---|
| `spectra.npy` | (N, 12, 101) | CLIMA atmospheric profiles — input features |
| `molecules.npy` | (N, 12) | log₁₀ surface molecular mixing ratios — targets |
| `aux_params.npy` | (N, 11) | Input fluxes & conditions metadata |
| `wavelengths.npy` | (101,) | Altitude axis (km, 0 = surface) |

Raw data: 10 tar.gz archives (`dir_0.tar.gz` … `dir_9.tar.gz` + `Dir_alpha.tar.gz`) + `pyatmos_summary.csv`.

---

## Project Structure

```
inara_vsode/
│
├── src/                           # Core ML modules (do not modify)
│   ├── data_utils.py              # Loading, splitting, normalisation, PCA, metrics
│   ├── baseline_model.py          # Per-molecule Random Forest model
│   └── deep_model.py              # SpectralResNet + Trainer + SpectralDataset
│
├── pipeline/
│   ├── config.yaml                # All configuration (paths, hyperparameters, toggles)
│   └── steps/                     # Numbered pipeline steps — run in order
│       ├── config_loader.py       # Shared config reader (used by all steps)
│       ├── 01_extract.py          # Step 1: raw archives → processed numpy arrays
│       ├── 02_feature_engineer.py # Step 2: split + normalise + PCA → artifacts
│       ├── 03_train_baseline.py   # Step 3: Random Forest (≤10k samples)
│       ├── 04_train_deep.py       # Step 4: SpectralResNet training
│       └── 05_evaluate.py         # Step 5: unified test eval + comparison report
│
├── jobs/
│   ├── slurm/                     # Northeastern Explorer HPC job scripts
│   │   ├── 01_extract.sh
│   │   ├── 02_feature_engineer.sh
│   │   ├── 03_train_baseline.sh
│   │   ├── 04_train_deep.sh
│   │   ├── 05_evaluate.sh
│   │   └── submit_pipeline.sh     # Submit all jobs with dependency chain
│   └── local/
│       └── run_pipeline.sh        # Mac M5 — runs all steps sequentially
│
├── notebooks/
│   └── eda.ipynb                  # Exploratory Data Analysis (standalone)
│
├── inara_data/
│   ├── processed/                 # Output of Step 1 (spectra.npy etc.)
│   └── engineered/                # Output of Step 2 (normalised arrays + PCA)
│
├── models/                        # Saved model weights
│   ├── baseline_rf.joblib
│   └── spectral_resnet.pt
│
├── results/                       # Metrics, predictions, training history
│
├── process_inara.py               # Standalone data extraction script
├── run_baseline.py                # Standalone baseline training script
└── run_deep_model.py              # Standalone deep model training script
```

---

## Setup

### Requirements

- Python 3.10+
- pandas >= 2.0
- numpy >= 1.24
- torch >=2.0
- `conda` or `pip` with the ML environment active

## 4. Environment Setup

### Local (Mac)

```bash
# Create and activate conda environment
conda create -n inara_env python=3.11 -y
conda activate inara_env

# Install dependencies
pip install numpy pandas scikit-learn torch joblib tqdm
pip install streamlit plotly pyyaml jupyter

# Verify PyTorch sees MPS (Apple Silicon)
python -c "import torch; print(torch.backends.mps.is_available())"
# Expected: True
```

### HPC (Northeastern Explorer)

```bash
# SSH into Explorer
ssh <user>@login.explorer.northeastern.edu

# Load base modules (check available with: module avail)
module load anaconda3

# Create environment (run once, from login node)
conda create -n inara_env python=3.11 -y
conda activate inara_env

# Install packages — include CUDA-enabled PyTorch
#pip install numpy pandas scikit-learn joblib tqdm pyyaml
#pip install torch --index-url https://download.pytorch.org/whl/cu118
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install scikit-learn numpy pandas matplotlib

# Verify GPU packages
#python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

> **Important:** Run `conda activate inara_env` before any pipeline commands.  
> Update `slurm.conda_env` in `pipeline/config.yaml` if you use a different environment name.

### Upload code and data to HPC

```bash
# Upload code to /home (never purged)
scp -r /path/to/inara_vscode  <user>@login.explorer.northeastern.edu:/home/<user>/

# Upload raw data to /scratch (fast Lustre filesystem)
scp /path/to/full_inara/*.tar.gz  <user>@login.explorer.northeastern.edu:/scratch/<user>/inara/raw/
scp /path/to/full_inara/pyatmos_summary.csv  <user>@login.explorer.northeastern.edu:/scratch/<user>/inara/raw/

# Or rsync (resumes interrupted transfers)
rsync -avzP /path/to/full_inara/  <user>@login.explorer.northeastern.edu:/scratch/<user>/inara/raw/
```



### Clone and navigate

```bash
git clone <repo-url>
cd inara
```

---
## 6. Running Locally (Mac M5)

### Full pipeline

```bash
cd /path/to/inara_vscode
bash jobs/local/run_pipeline.sh
```

### Skip extraction (data already processed)

```bash
bash jobs/local/run_pipeline.sh --skip-extract
```

### Skip extraction + feature engineering (engineered/ exists)

```bash
bash jobs/local/run_pipeline.sh --skip-extract --skip-feature-eng
```

**Expected runtimes (Mac M5 Pro, full 124k dataset):**

| Step | Time |
|---|---|
| Step 1 — Extraction | ~10–15 min |
| Step 2 — Feature Engineering | ~3–5 min |
| Step 3 — Baseline RF (10k) | < 1 min |
| Step 4 — Deep Model (MPS) | ~10–20 min |
| Step 5 — Evaluation | < 1 min |

---

## 7. Running on HPC (Northeastern Explorer)

### First-time setup checklist

1. Code uploaded to `/home/shinde.b/inara_vscode/`
2. Raw data uploaded to `/scratch/shinde.b/inara/raw/`
3. `inara_env` conda environment created and tested
4. `pipeline/config.yaml` updated:
   - `slurm.conda_env` → your environment name
   - `slurm.email` → your email for job notifications
   - Check partition names match what your account can access (`sinfo -s`)

### Submit the full pipeline

```bash
cd /home/shinde.b/inara_vscode
bash jobs/slurm/submit_pipeline.sh
```

This automatically submits all 5 jobs with the correct dependency chain.

### Submit with flags

```bash
# Skip extraction (processed/ already on scratch)
bash jobs/slurm/submit_pipeline.sh --skip-extract

# Skip extraction + feature engineering
bash jobs/slurm/submit_pipeline.sh --skip-extract --skip-feature-eng
```

### Monitor jobs

```bash
# View your queue
squeue -u shinde.b

# Watch live (refreshes every 5s)
watch -n 5 squeue -u shinde.b

# View job output in real-time
tail -f /scratch/shinde.b/inara/logs/04_train_deep_<JOBID>.out

# Cancel a specific job
scancel <JOBID>

# Cancel all your jobs
scancel -u shinde.b
```

## 8. Running Individual Steps

Each step is a standalone Python script. Run any step directly:

```bash
# Step 1 — Extract data
python pipeline/steps/01_extract.py --profile local

# Step 1 — Resume a partial extraction
python pipeline/steps/01_extract.py --profile local --resume

# Step 1 — Override n_samples from config
python pipeline/steps/01_extract.py --profile local --n-samples 10000

# Step 2 — Feature engineering
python pipeline/steps/02_feature_engineer.py --profile local

# Step 3 — Train baseline (saves model)
python pipeline/steps/03_train_baseline.py --profile local --save

# Step 3 — Override sample cap
python pipeline/steps/03_train_baseline.py --profile local --max-train-samples 5000

# Step 4 — Train deep model (saves checkpoint)
python pipeline/steps/04_train_deep.py --profile local --save

# Step 4 — Resume from checkpoint
python pipeline/steps/04_train_deep.py --profile local --resume models/spectral_resnet.pt

# Step 4 — Override hyperparameters
python pipeline/steps/04_train_deep.py --profile local --epochs 50 --patience 10

# Step 5 — Evaluate and compare
python pipeline/steps/05_evaluate.py --profile local
```

For HPC, add `--profile hpc` to any command above.

All steps accept `--config` to point to an alternative config file:
```bash
python pipeline/steps/03_train_baseline.py --config pipeline/config.yaml --profile hpc
```

---

## 9. Configuration Reference

`pipeline/config.yaml` controls all pipeline behaviour.

### Step toggles
```yaml
steps:
  extract:          true   # Step 1
  feature_engineer: true   # Step 2
  train_baseline:   true   # Step 3
  train_deep:       true   # Step 4
  evaluate:         true   # Step 5
```

### Data & extraction
```yaml
data:
  val_frac:  0.15    # 15% validation
  test_frac: 0.15    # 15% test
  seed:      42      # reproducibility seed for splits

extraction:
  n_samples: 124000  # samples to extract (reduce for dev/test)
  n_workers: 4       # parallel archive workers
  seed:      42
```

### Model & training
```yaml
model:
  in_channels:    12    # 12 CLIMA channels
  pca_components: 300   # PCA features for RF baseline

baseline:
  max_train_samples: 10000   # RF cap — always enforced

training:
  epochs:       150
  batch_size:   32
  lr:           0.001
  weight_decay: 0.0001
  patience:     30
```

### Profiles
```yaml
profiles:
  local:
    raw_dir:        /Users/bhalchandra/Downloads/full_inara
    processed_dir:  inara_data/processed       # relative paths OK for local
    engineered_dir: inara_data/engineered
    results_dir:    results
    models_dir:     models
    device:         auto                        # auto → MPS / CPU
    n_workers:      4

  hpc:
    raw_dir:        /scratch/shinde.b/inara/raw
    processed_dir:  /scratch/shinde.b/inara/processed
    engineered_dir: /scratch/shinde.b/inara/engineered
    results_dir:    /scratch/shinde.b/inara/results
    models_dir:     /scratch/shinde.b/inara/models
    device:         auto                        # auto → CUDA / CPU
    n_workers:      8
```

### SLURM
```yaml
slurm:
  account:   ""                          # leave blank if not required
  email:     shinde.b@northeastern.edu
  conda_env: inara_env
  code_dir:  /home/shinde.b/inara_vscode
```

---
