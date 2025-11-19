from typing import Generic, List, Optional, Self, TypeVar

import numpy as np
from numpy.typing import NDArray

from . import Array1DFloat, Array1DInt, ProbabilityEstimator


class CategoricalEstimator(ProbabilityEstimator):
    """
    Categorical estimator for a single feature.

    This estimator does not support NaN values in the input data.
    """

    _unique: NDArray[np.int64]
    _probabilities: NDArray[np.float64]

    def __init__(self):
        """
        Initialize the HistogramEstimator.
        """
        pass

    def fit(self, X: Array1DInt) -> Self:
        """Fit the histogram to the data X."""
        if np.isnan(X).any():
            raise ValueError("CategoricalEstimator does not support NaN values in the input data.")
        unique, counts = np.unique(X, return_counts=True)
        probabilities = counts / counts.sum()
        return self.copy_with(_unique=unique, _probabilities=probabilities)

    def predict(self, X: Array1DInt) -> Array1DFloat:
        """Compute the probability estimations for the data X."""
        if np.isnan(X).any():
            raise ValueError("FittedCategoricalEstimator does not support NaN values in the input data.")
        res = np.zeros((X.shape[0],), dtype=np.float64)
        for val, prob in zip(self._unique, self._probabilities):
            res[X == val] = prob
        return res


class RobustCategoricalEstimator(ProbabilityEstimator):
    """
    Categorical estimator for a single feature.

    This estimator does support NaN values in the input data.
    """

    _unique: NDArray[np.int64]
    _probabilities: NDArray[np.float64]
    _nan_probability: float

    def __init__(self):
        """
        Initialize the HistogramEstimator.
        """
        pass

    def fit(self, X: Array1DInt) -> Self:
        """Fit the histogram to the data X."""
        if np.isnan(X).any():
            raise ValueError("CategoricalEstimator does not support NaN values in the input data.")
        unique, counts = np.unique(X, return_counts=True)
        probabilities = counts / counts.sum()
        nan_probability = np.isnan(X).sum() / X.shape[0]
        return self.copy_with(_unique=unique, _probabilities=probabilities, _nan_probability=nan_probability)

    def predict(self, X: Array1DInt) -> Array1DFloat:
        """Compute the probability estimations for the data X."""
        res = np.zeros((X.shape[0],), dtype=np.float64)
        res[np.isnan(X)] = self._nan_probability
        for val, prob in zip(self._unique, self._probabilities):
            res[X == val] = prob
        return res
