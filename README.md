# MSML 605 — Milestone 1

## Project Overview
This milestone builds the foundational data pipeline for a face verification system using the LFW (Labeled Faces in the Wild) dataset. It covers deterministic dataset ingestion, identity-disjoint splits, and a vectorized similarity module with benchmarking. We are only writing the scripts that do not involve any machine learning training scripts, which will be covered in future milestones.

## How to Run

**1. Create and activate the virtual environment (Windows PowerShell):**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**2. Install dependencies:**
```powershell
pip install -r requirements.txt
```

**3. Run ingestion:**
```powershell
python scripts/ingest_lfw.py --config configs/milestone1.yaml
```

**4. Run pair generation:**
```powershell
python scripts/make_pairs.py --config configs/milestone1.yaml
```

**5. Run similarity benchmark:**
```powershell
python scripts/bench_similarity.py --config configs/milestone1.yaml
```

## Outputs
All outputs are written to the `outputs/` directory, which is gitignored and recreated on each run.

| File Name | Description |
|------|-------------|
| `outputs/manifest.json` | Dataset manifest with seed, split policy, and image/identity counts per split |
| `outputs/splits/train.csv` | Training split records (identity, filename, image_path) |
| `outputs/splits/val.csv` | Validation split records |
| `outputs/splits/test.csv` | Test split records |
| `outputs/pairs/train.csv` | Sampled verification pairs for training |
| `outputs/pairs/val.csv` | Sampled verification pairs for validation |
| `outputs/pairs/test.csv` | Sampled verification pairs for testing |
| `outputs/bench/results.txt` | Benchmark timings and correctness check results |