# MSML/MSAI 605 — Face Verification Pipeline

## Project Overview

This project builds a face verification system using the Labeled Faces in the Wild (LFW) dataset via TensorFlow Datasets. Given two face images, the system produces a similarity score and a same-person vs. different-person decision using a trained Siamese neural network with FaceNet embeddings.

- **Milestone 1** — Deterministic LFW ingestion, identity-level splits, pair generation, and vectorized similarity benchmarking.
- **Milestone 2** — Iterative evaluation loop with experiment tracking, TAR@FAR=0.01 threshold calibration, error analysis, a data-centric improvement to ingestion, and a full test suite.
- **Milestone 3** — Embedding-based inference using FaceNet (keras-facenet / VGGFace2), a CLI inference interface, Docker packaging, and a concurrency load test.

---

## Repository Structure

```
configs/
    milestone1.yaml               # Milestone 1 configuration
    milestone2.yaml               # Milestone 2 configuration
    milestone3.yaml               # Milestone 3 configuration
scripts/
    ingest_lfw.py                 # Original LFW ingestion (Milestone 1)
    ingest_lfw_updated.py         # Updated ingestion with quality filtering (Milestone 2)
    make_pairs.py                 # Deterministic pair generation
    model.py                      # SiameseVerifier + FaceEmbedder + FaceNetEmbedder
    metrics.py                    # TAR@FAR metric + Keras callback
    utils.py                      # Shared utilities (load_config, find_lfw_root)
    bench_similarity.py           # Vectorized similarity benchmark
    score_pairs.py                # Run inference and attach scores to pairs CSV
    run_threshold_sweep.py        # Threshold sweep + ROC plot
    run_evaluation.py             # Locked-threshold evaluation
    verify_pair_paths.py          # Diagnostic tool for missing image paths
    verify.py                     # CLI inference interface (Milestone 3)
    load_test.py                  # Concurrency load test (Milestone 3)
src/
    train.py                      # Training loop
    evaluate.py                   # Core evaluation logic
    tracker.py                    # Run tracker backed by run_log.json
    embedder.py                   # Standalone FaceNet embedder for CLI (Milestone 3)
tests/
    test_evaluate_and_tracker.py  # Unit + integration tests (Milestone 2)
    test_milestone3.py            # Unit + smoke tests for inference (Milestone 3)
reports/
    milestone2_report.pdf         # Milestone 2 evaluation report
Dockerfile                        # Docker packaging (Milestone 3)
.dockerignore
.gitignore
README.md
requirements.txt
```

---

## Milestone 3 Summary

### Embedding-Based Inference
Milestone 3 replaces the plain CNN baseline with **FaceNet embeddings** (InceptionResnetV1 pretrained on VGGFace2, via `keras-facenet`). The updated `SiameseVerifier` uses `FaceNetEmbedder` internally during training and scoring. The standalone `src/embedder.py` module provides the same embeddings for the CLI and load test.

### Inference Pipeline
Each pair goes through four clearly separated stages:
1. **Preprocess** — load image, resize to 160×160 RGB
2. **Embed** — run FaceNet to produce a 512-dim L2-normalised embedding
3. **Score** — compute cosine similarity, apply sigmoid → score in [0, 1]
4. **Decide** — compare score against threshold; compute confidence

### Confidence
Confidence is a linear distance from the decision boundary:
- `SAME (score ≥ threshold)`: `confidence = (score − threshold) / (1 − threshold)`
- `DIFFERENT (score < threshold)`: `confidence = (threshold − score) / threshold`
- Range: 0.0 = at the boundary, 1.0 = maximum distance from boundary

### Threshold
Selected using the same **TAR@FAR=0.01 rule** from Milestone 2, applied to the validation split using FaceNet sigmoid scores. The current locked threshold is `0.709294`.

### Load Test
`scripts/load_test.py` runs N pairs across W concurrent threads using a shared model instance, and reports throughput (req/s), mean, p50, p95, and p99 latency. Results are saved to `outputs/load_test_results.json`.

---

## Milestone 2 Summary

### Baseline
The baseline system uses a Siamese CNN trained from scratch on LFW pairs. Threshold selection follows the **TAR@FAR=0.01 rule**: on the validation split, the threshold that maximises TAR while keeping FAR ≤ 1% is chosen. The baseline selected threshold was **0.7166**.

### Data-Centric Improvement
`ingest_lfw_updated.py` introduces two changes over the original ingestion:
1. **Minimum image filter** — identities with fewer than 10 images are excluded, ensuring every identity has enough examples to form meaningful pairs.
2. **Per-identity proportional splitting** — each identity's images are split individually at 70/15/15 (train/val/test), so every identity appears in all three splits.

After retraining on the improved data (v2), the selected threshold was **0.7750**, and TAR on the test split improved from 0.02% (1 TP) to 4.98% (249 TP) at FAR = 0.84%.

### Tracked Runs

| Run | Split | Data | Threshold | TAR | FAR | Accuracy |
|-----|-------|------|-----------|-----|-----|----------|
| 1 – Baseline sweep     | val  | v1 | 0.7166 | 0.02% | 0.00% | 50.01% |
| 2 – Baseline val eval  | val  | v1 | 0.7166 | 0.00% | 0.00% | 50.00% |
| 3 – Baseline test eval | test | v1 | 0.7166 | 0.02% | 0.00% | 50.01% |
| 4 – Post-change sweep  | val  | v2 | 0.7750 | 5.30% | 0.98% | 52.16% |
| 5 – Post-change test   | test | v2 | 0.7750 | 4.98% | 0.84% | 52.07% |

Full run history is logged at `outputs/runs/run_log.json`.

---

## Artifact Locations

| Artifact | Path |
|----------|------|
| Run log (all runs) | `outputs/runs/run_log.json` |
| Baseline ROC curve | `outputs/runs/2ada8840/roc_curve.png` |
| Post-change ROC curve | `outputs/runs/49164c9b/roc_curve.png` |
| Baseline confusion matrix | `outputs/runs/03b57698/confusion_matrix.png` |
| Post-change confusion matrix | `outputs/runs/098c18f4/confusion_matrix.png` |
| Load test results | `outputs/load_test_results.json` |
| Milestone 2 report | `reports/milestone2_report.pdf` |

---

## Environment Setup

> **Windows PowerShell note:** use `python -m pip install` instead of `pip install` to avoid launcher path issues.

**1. Create and activate the virtual environment:**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**2. Install dependencies:**
```powershell
python -m pip install -r requirements.txt
```

---

## How to Run — Milestone 3

All commands are single-line to avoid PowerShell `--` parsing issues.

### Step 1 — Ingest and generate pairs
```powershell
python scripts/ingest_lfw_updated.py --config configs/milestone3.yaml
python scripts/make_pairs.py --config configs/milestone3.yaml
```

### Step 2 — Train the model with FaceNet embeddings
```powershell
python src/train.py --config configs/milestone3.yaml
```

### Step 3 — Score the splits
```powershell
python scripts/score_pairs.py --config configs/milestone3.yaml --split val --scoring-mode embedding
python scripts/score_pairs.py --config configs/milestone3.yaml --split test --scoring-mode embedding
```

### Step 4 — Run threshold sweep and lock the threshold
```powershell
python scripts/run_threshold_sweep.py --config configs/milestone3.yaml --note "M3 FaceNet threshold sweep"
```
Copy the printed `selected_threshold` into `configs/milestone3.yaml` under `milestone3.locked_threshold`.

### Step 5 — Run evaluation
```powershell
python scripts/run_evaluation.py --config configs/milestone3.yaml --split val --note "M3 val eval"
python scripts/run_evaluation.py --config configs/milestone3.yaml --split test --note "M3 test eval"
```

### Step 6 — Run unit and smoke tests
```powershell
python -m pytest tests/test_milestone3.py -v
python -m pytest tests/test_evaluate_and_tracker.py -v
```

### Step 7 — Run the CLI on a pair of images
```powershell
python scripts/verify.py --config configs/milestone3.yaml --image-a path\to\face1.jpg --image-b path\to\face2.jpg
```

Expected output:
```
============================================================
  Image A      : path\to\face1.jpg
  Image B      : path\to\face2.jpg
  ────────────────────────────────────────────────────────
  Score        : 0.682785  (sigmoid probability, range [0, 1])
  Cosine sim   : 0.766602  (raw embedding similarity, range [-1, 1])
  Threshold    : 0.709294  (TAR@FAR=0.01, selected on val split)
  Decision     : DIFFERENT
  Confidence   : 0.0374  (0=boundary, 1=max confidence)
  ────────────────────────────────────────────────────────
  Latency breakdown:
    Preprocess :     2.96 ms
    Embed      :  1326.90 ms
    Score+dec  :     0.03 ms
    Total      :  1329.89 ms
============================================================
```

For batch inference across multiple pairs:
```powershell
python scripts/verify.py --config configs/milestone3.yaml --pairs-csv outputs/pairs/val_pairs.csv --lfw-root data\tfds_cache\downloads\extracted\...\lfw --max-pairs 20
```

### Step 8 — Run the load test
```powershell
python scripts/load_test.py --config configs/milestone3.yaml --workers 2 --n-pairs 50
```

Results are saved to `outputs/load_test_results.json`.

### Step 9 — Build and verify Docker
```powershell
docker build -t msml605-verifier .
```

Run tests inside the container:
```powershell
docker run --rm msml605-verifier python -m pytest tests/test_milestone3.py -v
```

Run the load test inside the container:
```powershell
docker run --rm msml605-verifier python scripts/load_test.py --config configs/milestone3.yaml --workers 2 --n-pairs 10
```

---

## How to Run — Milestone 2

All commands are single-line to avoid PowerShell `--` parsing issues.

### Step 1 — Ingest with the updated script
```powershell
python scripts/ingest_lfw_updated.py --config configs/milestone2.yaml
```

### Step 2 — Generate pairs
```powershell
python scripts/make_pairs.py --config configs/milestone2.yaml
```

### Step 3 — Train the model
```powershell
python src/train.py --config configs/milestone2.yaml
```

### Step 4 — Score the splits
```powershell
python scripts/score_pairs.py --config configs/milestone2.yaml --split val
python scripts/score_pairs.py --config configs/milestone2.yaml --split test
```

### Step 5 — Run 1: Baseline threshold sweep on val
```powershell
python scripts/run_threshold_sweep.py --config configs/milestone2.yaml --note "Run 1 - baseline threshold sweep"
```
After this completes, copy the printed `selected_threshold` value into `configs/milestone2.yaml` under `milestone2.locked_threshold`.

### Step 6 — Run 2: Baseline evaluation on val
```powershell
python scripts/run_evaluation.py --config configs/milestone2.yaml --split val --note "Run 2 - baseline val"
```

### Step 7 — Run 3: Baseline final reporting on test
```powershell
python scripts/run_evaluation.py --config configs/milestone2.yaml --split test --note "Run 3 - baseline test"
```

### Step 8 — Apply data-centric change and retrain
```powershell
python scripts/ingest_lfw_updated.py --config configs/milestone2.yaml
python scripts/make_pairs.py --config configs/milestone2.yaml
python src/train.py --config configs/milestone2.yaml
```

### Step 9 — Score the updated splits
```powershell
python scripts/score_pairs.py --config configs/milestone2.yaml --split val
python scripts/score_pairs.py --config configs/milestone2.yaml --split test
```

### Step 10 — Run 4: Post-change threshold sweep on val
```powershell
python scripts/run_threshold_sweep.py --config configs/milestone2.yaml --data-version v2 --note "Run 4 - post data-centric change sweep"
```
Update `configs/milestone2.yaml`: set `milestone2.locked_threshold` to the new value and `milestone2.data_version` to `v2`.

### Step 11 — Run 5: Post-change final reporting on test
```powershell
python scripts/run_evaluation.py --config configs/milestone2.yaml --split test --data-version v2 --note "Run 5 - post data-centric change test"
```

### Step 12 — Run tests
```powershell
python -m pytest tests/test_evaluate_and_tracker.py -v
```

---

## How to Run — Milestone 1

### Step 1 — Run ingestion
```powershell
python scripts/ingest_lfw.py --config configs/milestone1.yaml
```

### Step 2 — Generate pairs
```powershell
python scripts/make_pairs.py --config configs/milestone1.yaml
```

### Step 3 — Run similarity benchmark
```powershell
python scripts/bench_similarity.py
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'facenet_pytorch'`**
The Dockerfile was built before the switch to `keras-facenet`. Rebuild the image:
```powershell
docker build -t msml605-verifier .
```

**`FaceEmbedder.__init__() got an unexpected keyword argument 'pretrained'`**
An old version of `verify.py` or `load_test.py` is being used. Replace with the current versions from the repository.

**`Could not locate class 'SiameseVerifier'`**
The model was saved before the Keras serialization decorators were added. Retrain with the current `scripts/model.py` and re-run scoring.

**`Scored CSV not found`**
Run `score_pairs.py` for the relevant split before running the sweep or evaluation scripts.

**`FileNotFoundError` for image paths**
Run the diagnostic tool to identify missing files:
```powershell
python scripts/verify_pair_paths.py --config configs/milestone3.yaml
```
Most likely cause is an incomplete TFDS extraction. Delete `data/tfds_cache` and re-run ingestion.

**`pip install` fails with launcher error**
Use `python -m pip install <package>` instead.

**PowerShell `--` parsing error**
Run commands as a single line without backtick line continuation.

**Docker build context is very large (2+ GB)**
Make sure `.dockerignore` is in the project root. It excludes `data/`, `outputs/`, and `.venv/` from the build context.

---

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
| `outputs/load_test_results.json` | Load test throughput and latency summary |
| `outputs/bench/` | Similarity benchmark results |
