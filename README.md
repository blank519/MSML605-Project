# MSML/MSAI 605 — Face Verification Pipeline
## William Peng and Eric Wang
## Project Overview

This project builds a face verification system using the Labeled Faces in the Wild (LFW) dataset via TensorFlow Datasets. Given two face images, the system produces a similarity score and a same-person vs. different-person decision using a trained Siamese neural network.

- **Milestone 1** — Deterministic LFW ingestion, identity-level splits, pair generation, and vectorized similarity benchmarking.
- **Milestone 2** — Iterative evaluation loop with experiment tracking, TAR@FAR=0.01 threshold calibration, error analysis, a data-centric improvement to ingestion, and a full test suite.

---

## Repository Structure

```
configs/
    milestone1.yaml          # Milestone 1 configuration
    milestone2.yaml          # Milestone 2 configuration
scripts/
    ingest_lfw.py            # Original LFW ingestion (Milestone 1)
    ingest_lfw_updated.py    # Updated ingestion with quality filtering (Milestone 2)
    make_pairs.py            # Deterministic pair generation
    model.py                 # SiameseVerifier + FaceEmbedder architecture
    metrics.py               # TAR@FAR metric + Keras callback
    utils.py                 # Shared utilities (load_config, find_lfw_root)
    bench_similarity.py      # Vectorized similarity benchmark
    score_pairs.py           # Run inference and attach scores to pairs CSV
    run_threshold_sweep.py   # Threshold sweep + ROC plot (Milestone 2)
    run_evaluation.py        # Locked-threshold evaluation (Milestone 2)
    verify_pair_paths.py     # Diagnostic tool for missing image paths
src/
    train.py                 # Training loop
    evaluate.py              # Core evaluation logic (Milestone 2)
    tracker.py               # Run tracker backed by run_log.json (Milestone 2)
tests/
    test_evaluate_and_tracker.py  # Unit + integration tests (Milestone 2)
reports/
    milestone2_report.pdf    # Milestone 2 evaluation report
.gitignore
README.md
requirements.txt
```

## Environment Setup

> **Windows PowerShell note:** use `python -m pip install` instead of `pip install` to avoid launcher path issues.

**Create and activate the virtual environment:**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Install dependencies:**
```powershell
python -m pip install -r requirements.txt
```

---

## How to Run — Milestone 1

**Run ingestion:**
```powershell
python scripts/ingest_lfw.py --config configs/milestone1.yaml
```

**Generate pairs:**
```powershell
python scripts/make_pairs.py --config configs/milestone1.yaml
```

**Run similarity benchmark:**
```powershell
python scripts/bench_similarity.py
```

---

## How to Run — Milestone 2

All commands are single-line to avoid PowerShell `--` parsing issues.

### Ingest with the updated script
```powershell
python scripts/ingest_lfw_updated.py --config configs/milestone2.yaml
```

### Generate pairs
```powershell
python scripts/make_pairs.py --config configs/milestone2.yaml
```

### Train the model
```powershell
python src/train.py --config configs/milestone2.yaml
```

### Score the splits
```powershell
python scripts/score_pairs.py --config configs/milestone2.yaml --split val
python scripts/score_pairs.py --config configs/milestone2.yaml --split test
```

### Run 1: Baseline threshold sweep on val
```powershell
python scripts/run_threshold_sweep.py --config configs/milestone2.yaml --note "Run 1 - baseline threshold sweep"
```
After this completes, copy the printed `selected_threshold` value into `configs/milestone2.yaml` under `milestone2.locked_threshold`.

### Run 2: Baseline evaluation on val
```powershell
python scripts/run_evaluation.py --config configs/milestone2.yaml --split val --note "Run 2 - baseline val"
```

### Run 3: Baseline final reporting on test
```powershell
python scripts/run_evaluation.py --config configs/milestone2.yaml --split test --note "Run 3 - baseline test"
```

### Apply data-centric change and retrain
```powershell
python scripts/ingest_lfw_update.py --config configs/milestone2.yaml
python scripts/make_pairs.py --config configs/milestone2.yaml
python src/train.py --config configs/milestone2.yaml
```

### Score the updated splits
```powershell
python scripts/score_pairs.py --config configs/milestone2.yaml --split val
python scripts/score_pairs.py --config configs/milestone2.yaml --split test
```

### Run 4: Post-change threshold sweep on val
```powershell
python scripts/run_threshold_sweep.py --config configs/milestone2.yaml --data-version v2 --note "Run 4 - post data-centric change sweep"
```
Update `configs/milestone2.yaml`: set `milestone2.locked_threshold` to the new value and `milestone2.data_version` to `v2`.

### Run 5: Post-change final reporting on test
```powershell
python scripts/run_evaluation.py --config configs/milestone2.yaml --split test --data-version v2 --note "Run 5 - post data-centric change test"
```

### Run tests
```powershell
python -m pytest tests/test_evaluate_and_tracker.py -v
```

## Outputs

All outputs are written to `outputs/` which is gitignored and recreated on each run. The LFW dataset is downloaded via TFDS and is not included in the repository.

| Path | Description |
|------|-------------|
| `outputs/manifest.json` | Dataset manifest with seed, split policy, image/identity counts |
| `outputs/splits/` | train/val/test split CSVs |
| `outputs/pairs/` | Verification pair CSVs for each split |
| `outputs/scores/` | Scored pair CSVs with model similarity scores |
| `outputs/models/` | Saved Siamese model (.keras) |
| `outputs/runs/` | Per-run artifacts (sweep CSV, plots, metrics JSON, run_log.json) |
| `outputs/bench/` | Similarity benchmark results |

## Milestone 3

### Replacing the Embedder
In Milestone 2, the placeholder model in 'src/models.py' also used the image embedding -> similarity score approach, but the embedder was a neural network created from scratch. In Milestone 3, the embedder was replaced with a pre-trained FaceNet model.  

The reason why FaceNet was chosen over other pretrained models is due to how well-established it is as a baseline for face recognition tasks.