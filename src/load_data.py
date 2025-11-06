"""Helper to load and clean the ATLAS Higgs CSV file.

Functions:
 - load_data: read CSV, set index, replace missing sentinel (-999.0) with NaN and optionally drop rows with NaN
"""
from typing import Tuple
import pandas as pd
import numpy as np


def load_data(path: str, drop_missing: bool = True) -> pd.DataFrame:
    """Load the dataset and perform light cleaning.

    Parameters
    ----------
    path : str
        Path to the gzipped CSV file.
    drop_missing : bool
        If True, replace -999.0 with NaN and drop rows containing NaNs.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with EventId as index.
    """
    df = pd.read_csv(path, compression='gzip')
    if 'EventId' in df.columns:
        df.set_index('EventId', inplace=True)

    # Notebook uses -999.0 as missing sentinel
    
    if drop_missing:
        df.replace(-999.0, np.nan, inplace=True)
        df = df.dropna()
    return df


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('path', nargs='?', default='atlas-higgs-challenge-2014-v2.csv.gz')
    args = p.parse_args()
    df = load_data(args.path)
    print(df.shape)