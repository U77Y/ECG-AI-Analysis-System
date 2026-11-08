# ECG-AI Analysis System — Research V2

A reproducible research codebase for **12-lead ECG multi-label classification** using the PTB-XL dataset and PyTorch.

> **Research use only.** This software is not a medical device and must not be used to diagnose, triage, or guide treatment. No clinical performance is claimed. Clinical use would require independent external validation, regulatory review, and qualified clinician oversight.

## Research objective

Train and evaluate models that predict the five PTB-XL diagnostic superclasses:

- NORM — normal ECG
- MI — myocardial infarction
- STTC — ST/T change
- CD — conduction disturbance
- HYP — hypertrophy

These are multi-label targets: a record may have more than one positive superclass.

## Dataset

Download PTB-XL from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) and place the extracted dataset at `data/ptb-xl/` (or pass another path with `--data-dir`). The expected files include `ptbxl_database.csv`, `scp_statements.csv`, and the waveform record folders.

Read and follow the dataset's PhysioNet license and citation requirements. Do not commit the dataset or trained model weights to Git.

## Environment

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt
```

## First experiment

Run from the repository root:

```bash
python -m src.ecg_research.train --data-dir data/ptb-xl --epochs 20 --batch-size 32
```

The training script uses folds 1–8 for training, fold 9 for validation, and fold 10 for the held-out test set. The validation set selects the checkpoint; the test set is evaluated only after training. Keep patient-level folds intact.

Outputs are written to `experiments/ptbxl_cnn/` and are excluded from Git.

## Current pipeline

1. Read PTB-XL metadata and SCP diagnostic statements.
2. Map SCP codes to five diagnostic superclasses.
3. Load 12-lead waveform records with WFDB.
4. Apply a configurable 0.5–40 Hz Butterworth band-pass filter.
5. Normalize each lead per recording.
6. Train a compact 1D CNN with multi-label binary cross-entropy.
7. Report macro AUROC, macro AUPRC, macro F1, sensitivity, and specificity.

Metrics are research estimates on the specified split, not clinical accuracy. Report per-class results and class prevalence; do not substitute prediction confidence for accuracy.

## Project structure

```text
configs/                 Experiment configuration
src/ecg_research/         Dataset, preprocessing, model, metrics, training
tests/                    Lightweight unit tests
experiments/              Local checkpoints and metrics (git-ignored)
data/                     Local datasets (git-ignored)
```

## Reproducibility

- Fixed random seed (default: 42)
- Explicit fold-based train/validation/test split
- Saved configuration and metrics with each run
- No synthetic ECG data used for model evaluation
- Raw data and model artifacts remain outside version control

## Next research milestones

- Verify the data loader against a small set of PTB-XL records.
- Establish a transparent baseline and per-class error analysis.
- Add patient-independent repeated experiments and confidence intervals.
- Compare the CNN with classical baselines and a ResNet1D.
- Add explainability only after baseline evaluation is stable.

## Citation

When using PTB-XL, cite the dataset publication and the PhysioNet dataset record:

Wagner et al., *PTB-XL, a large publicly available electrocardiography dataset*, Scientific Data (2020). https://doi.org/10.1038/s41597-020-0495-6

Dataset: https://physionet.org/content/ptb-xl/1.0.3/
