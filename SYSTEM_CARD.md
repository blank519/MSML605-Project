# System Card — MSML/MSAI 605 Face Verification Pipeline

## System overview

This repository implements a **pairwise face verifier**: given two face images, it produces:

- A **similarity score** in `[0, 1]`.
- A **binary decision**: `same` person vs. `different` people.
- A **confidence** value in `[0, 1]` that measures distance from the decision threshold.

### Inputs

- **Single-pair CLI mode**: two image paths (`.jpg`, `.jpeg`, `.png`).
- **Batch mode**: a CSV of pairs with at least:
  - `left_path`
  - `right_path`
  - (optionally) `label` for evaluation pipelines

Paths in the generated pairs CSVs are typically **relative to the LFW root** (e.g., `Person_Name/Person_Name_0001.jpg`). The CLI supports resolving these paths via `--lfw-root`, which can be either the actual `lfw/` directory or a TFDS `downloads/extracted` root.

### High-level pipeline (Milestone 3 “deployed” scoring)

The authoritative deployed inference pipeline is implemented in `src/embedder.py` and used by:

- `scripts/verify.py` (CLI)
- `scripts/load_test.py` (concurrency/load)
- `scripts/score_pairs.py --scoring-mode embedding` (threshold selection + evaluation scoring)

Stages:

1. **Preprocess**
   - Load image from disk
   - Convert to RGB
   - Resize to `160×160`

2. **Embed**
   - Run FaceNet via `keras-facenet` (InceptionResnetV1 pretrained on VGGFace2)
   - Output: 512-dimensional embedding

3. **Score**
   - Compute cosine similarity of embeddings
   - Convert to score: `score = sigmoid(cosine_similarity)`

4. **Decision + confidence**
   - `same` if `score >= threshold`, else `different`
   - Confidence rule (linear distance from the boundary):
     - If `score >= threshold`: `(score - threshold) / (1 - threshold)`
     - Else: `(threshold - score) / threshold`

## Intended use

### Intended / supported use cases

- **Educational / research** face verification experiments on LFW-style data.
- Reproducible threshold selection using **TAR@FAR=0.01** on a validation split.
- CLI-style verification of two images as a demonstration.
- Basic performance and latency exploration via the included load test.

### Non-intended / out-of-scope uses

- **Identity recognition** (1-to-N search) is not implemented.
- **Face detection, alignment, or cropping** is not implemented; the system assumes the provided images are already face crops or are sufficiently face-dominant.
- **Security- or safety-critical deployments** (e.g., access control, law enforcement, high-stakes decisions) are out of scope.
- Claims of real-world performance beyond LFW-like distributions are out of scope.

## Data summary

### Data source

- **Labeled Faces in the Wild (LFW)** loaded via **TensorFlow Datasets**.

### Data limitations relevant to interpretation

- LFW is a benchmark dataset and is not representative of all real-world capture conditions.
- Demographic coverage and capture settings are uneven; performance may vary substantially across subgroups.
- Images are “in the wild” but still constrained by the dataset’s collection and labeling assumptions.
- Pair generation is deterministic and configured via YAML; results depend on the split/pair sampling.

## Operating threshold and metrics

### Score definition (final system version)

- `score = sigmoid(cosine_similarity(emb_a, emb_b))`
- Score range: `[0, 1]`
- Higher score means “more likely same person”.

### Threshold selection rule

- **Selection rule**: maximize TAR subject to **FAR ≤ 0.01** (i.e., TAR@FAR=0.01)
- **Selection split**: validation (`val`)

### Selected operating point (most recent embedding-scored sweep)

From `scripts/run_threshold_sweep.py` run on `outputs/scores/val_scored.csv` produced with `scripts/score_pairs.py --scoring-mode embedding`:

- **Selected threshold**: `0.617447`
- **TAR**: `0.9268`
- **FAR**: `0.0088`
- **Accuracy**: `0.9590`
- **F1**: `0.9576`

## Failure modes and limitations

Where the system can become less reliable or fail:

- **Non-face / non-frontal / multi-face images**: no detector is used, so embeddings may be meaningless or inconsistent.
- **Occlusion** (masks, sunglasses), extreme pose, blur, low resolution, heavy compression.
- **Lighting and color shifts** (strong shadows, very low light) can change embeddings.
- **Domain shift**: performance can drop when images are from cameras/conditions unlike LFW.
- **Threshold mismatch risk**: if you select a threshold using a different scoring definition than the deployed CLI path, the locked threshold will not transfer. This repo’s current best practice is to use `score_pairs.py --scoring-mode embedding` to match CLI scoring.
- **Path resolution failures**: batch mode requires correct `--lfw-root` so relative paths resolve to actual files.

## Fairness-risk discussion

This project uses pretrained embeddings (VGGFace2-trained FaceNet) and LFW evaluation:

- **Uneven subgroup performance is plausible**: demographic imbalances in pretraining data and evaluation data can yield different false accept/reject rates across groups.
- **Misuse risk**: even “verification” scores can be used for surveillance or discriminatory gatekeeping if deployed irresponsibly.
- **Overconfidence risk**: the confidence value is derived from distance to a single threshold, not from calibrated uncertainty estimation.

Recommended practice for any real-world extension:

- Perform subgroup evaluations (if permitted) and document uncertainty.
- Avoid deployment in high-stakes settings.
- Add consent, transparency, and human oversight policies.

## Operational constraints

### Hardware / performance

- Runs on CPU by default; TensorFlow may log that GPU drivers are not found.
- First inference call can be slower due to model initialization and graph warm-up.
- Latency is dominated by embedding generation (`keras-facenet`).

### Input format assumptions

- Images should be readable by PIL and convertible to RGB.
- Expected size is resized to `160×160`; no face alignment is performed.
- Batch pipelines assume the pair CSV schema described above.

### Deployment assumptions

- CLI (`scripts/verify.py`) is the primary “deployment-like” interface.
- Docker support exists via `Dockerfile` (see README).

## Reproducibility pointer

- **Primary documentation**: `README.md`
- **Configuration**: `configs/milestone3.yaml`
- **Key commands (Milestone 3)**:
  - Ingest + pairs:
    - `python scripts/ingest_lfw_updated.py --config configs/milestone3.yaml`
    - `python scripts/make_pairs.py --config configs/milestone3.yaml`
  - Score (aligned with deployed scoring):
    - `python scripts/score_pairs.py --config configs/milestone3.yaml --split val --scoring-mode embedding`
    - `python scripts/score_pairs.py --config configs/milestone3.yaml --split test --scoring-mode embedding`
  - Threshold sweep:
    - `python scripts/run_threshold_sweep.py --config configs/milestone3.yaml`
  - Evaluation:
    - `python scripts/run_evaluation.py --config configs/milestone3.yaml --split val`
    - `python scripts/run_evaluation.py --config configs/milestone3.yaml --split test`
  - CLI inference:
    - `python scripts/verify.py --config configs/milestone3.yaml --image-a <a.jpg> --image-b <b.jpg>`
  - Load test:
    - `python scripts/load_test.py --config configs/milestone3.yaml --workers 2 --n-pairs 50`

- **Run artifacts**: `outputs/runs/<run_id>/` (metrics JSON, confusion matrix, plots)
- **Tracked run log**: `outputs/runs/run_log.json`
- **Supporting report**: `reports/milestone2_report.pdf`
