"""Preprocessing helpers: build per-feature ColumnTransformer based on skewness.


"""
from typing import Tuple, Dict, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline


def build_preprocessor(X_train: pd.DataFrame, simple: bool = True) -> Tuple[ColumnTransformer, Dict[str, List[str]]]:
    """Build a ColumnTransformer for X_train.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features dataframe.
    simple : bool
        If True, apply StandardScaler to all columns. If False, use the skewness-based
        grouping (original notebook behaviour).

    Returns
    -------
    ColumnTransformer, dict
        The fitted transformer and a mapping of column groups.
    """
    if simple:
        cols = X_train.columns.tolist()
        transformers = [('std_all', StandardScaler(), cols)] if cols else []
        preprocessor = ColumnTransformer(transformers, remainder='drop')
        if cols:
            preprocessor.fit(X_train)
        groups = dict(standard=cols)
        return preprocessor, groups

    # fallback to skewness-based grouping when simple=False
    skews = X_train.skew().abs()

    normal_cols = skews[skews <= 0.5].index.tolist()
    moderate_cols = skews[(skews > 0.5) & (skews <= 1.0)].index.tolist()
    heavy_cols = skews[skews > 1.0].index.tolist()

    transformers = []
    if normal_cols:
        transformers.append(('std', StandardScaler(), normal_cols))
    if moderate_cols:
        transformers.append(('pt_std', Pipeline([('pt', PowerTransformer(method='yeo-johnson')), ('std', StandardScaler())]), moderate_cols))
    if heavy_cols:
        transformers.append(('robust', RobustScaler(), heavy_cols))

    preprocessor = ColumnTransformer(transformers, remainder='drop')
    preprocessor.fit(X_train)

    groups = dict(normal=normal_cols, moderate=moderate_cols, heavy=heavy_cols)
    return preprocessor, groups


if __name__ == '__main__':
    # minimal smoke test
    import numpy as np
    import pandas as pd
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        'a': rng.normal(size=100),
        'b': rng.exponential(scale=2.0, size=100),
        'c': rng.normal(loc=5, scale=2.0, size=100)
    })
    pre, groups = build_preprocessor(df)
    print(groups)