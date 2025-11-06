"""Train and evaluate a logistic regression model using a leakage-free pipeline.

This script performs the following steps:
 - load the data
 - drop rows with missing values
 - split into train/test (keeps weights for AMS)
 - build a per-feature preprocessor using only the training set
 - cross-validate on the training set and evaluate on the held-out test set
"""
import argparse
import os
import sys
import time
from load_data import load_data
from preprocess import build_preprocessor
from evaluate import report_metrics
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import numpy as np


def main(path: str, test_size: float = 0.2, random_state: int = 42, fast: bool = False, cv: bool = False):
    # fail fast with a helpful message if the dataset path does not exist
    if not os.path.exists(path):
        msg = (
            f"Dataset file not found: {path}\n\n"
            "Suggestions:\n"
            "  - Pass the correct path with --path, e.g.\n"
            "      python \"Practical work/train.py\" --path \"/full/path/to/atlas-higgs-challenge-2014-v2.csv.gz\"\n"
            "  - Place the dataset file in the current working directory where you run the script.\n"
            "  - If you have the dataset inside the project folder, provide the relative path, for example:\n"
            "      python \"Practical work/train.py\" --path \"Practical work/atlas-higgs-challenge-2014-v2.csv.gz\"\n"
        )
        print(msg, file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    print("Loading data...", flush=True)
    df = load_data(path, drop_missing=True)
    print("Loaded dataframe shape:", df.shape, "(t=%.1fs)" % (time.time() - t0), flush=True)

    # Keep weights for AMS evaluation
    if 'Weight' in df.columns:
        weights = df['Weight']
    else:
        weights = pd.Series(np.ones(len(df)), index=df.index)

    X = df.drop(columns=['Label', 'KaggleSet', 'KaggleWeight', 'Weight'], errors='ignore')
    y = df['Label']

    if fast:
        # quick debug mode: sample up to 2000 rows for much faster runs
        n = min(2000, len(X))
        print(f"FAST mode: sampling first {n} rows for quick run", flush=True)
        X = X.iloc[:n]
        y = y.iloc[:n]

    # split first (no preprocessing fitted on full dataset)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train/test sizes: {X_train.shape} / {X_test.shape}", flush=True)

    print("Building preprocessor (StandardScaler for all features)...", flush=True)
    preprocessor, groups = build_preprocessor(X_train, simple=True)
    print("Feature groups:")
    print({k: len(v) for k, v in groups.items()})

    pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, random_state=random_state))
    ])

    print("Fitting pipeline on full training set...", flush=True)
    t2 = time.time()
    pipe.fit(X_train, y_train.map({'s': 1, 'b': 0}))
    print("Fit complete (t=%.1fs)" % (time.time() - t2), flush=True)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps['clf'], 'predict_proba') else None

    # y_test is 's'/'b' -> convert to numeric for report
    y_test_num = y_test.map({'s': 1, 'b': 0})

    report_metrics(y_test_num, y_pred, y_proba if y_proba is not None else np.zeros_like(y_pred, dtype=float), weights=w_test.values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='atlas-higgs-challenge-2014-v2.csv.gz')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()
    main(args.path, args.test_size, args.random_state)
