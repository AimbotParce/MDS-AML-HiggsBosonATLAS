# Practical work scripts

Files:

- `load_data.py` - helpers to read the gzipped CSV and perform light cleaning.
- `preprocess.py` - build a per-feature ColumnTransformer based on skewness measured on the training set.
- `train.py` - orchestrates loading, splitting, building a leakage-free pipeline, cross-validation and evaluation.
- `evaluate.py` - metrics helpers including the AMS metric used in the notebook.
- `requirements.txt` - minimal dependencies.

Quick start
1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r "Practical work/requirements.txt"
```

2. Run the training script (default path expects the CSV in the current directory):

```bash
python3 "Practical work/train.py" --path "atlas-higgs-challenge-2014-v2.csv.gz"
```

Notes
- The scripts intentionally keep the "weights" column (if present) for AMS computation.
- The preprocessing is built using only the training set (leakage-free).
