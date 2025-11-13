from typing import Generic, List, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

from . import Array1DFloat, Array1DInt, FittedProbabilityEstimator, ProbabilityEstimator


class FittedCategoricalEstimator(FittedProbabilityEstimator):
    """
    Fitted categorical estimator for a single feature.

    This estimator does not support NaN values in the input data.
    """

    def __init__(self, unique: NDArray[np.int64], probabilities: NDArray[np.float64]):
        self.unique = unique
        self.probabilities = probabilities

    def predict(self, X: Array1DInt) -> Array1DFloat:
        """Compute the probability estimations for the data X."""
        if np.isnan(X).any():
            raise ValueError("FittedCategoricalEstimator does not support NaN values in the input data.")
        res = np.zeros((X.shape[0],), dtype=np.float64)
        for val, prob in zip(self.unique, self.probabilities):
            res[X == val] = prob
        return res


class CategoricalEstimator(ProbabilityEstimator):
    """
    Categorical estimator for a single feature.

    This estimator does not support NaN values in the input data.
    """

    def __init__(self):
        """
        Initialize the HistogramEstimator.
        """
        pass

    def fit(self, X: Array1DInt) -> FittedCategoricalEstimator:
        """Fit the histogram to the data X."""
        if np.isnan(X).any():
            raise ValueError("CategoricalEstimator does not support NaN values in the input data.")
        unique, counts = np.unique(X, return_counts=True)
        probabilities = counts / counts.sum()
        return FittedCategoricalEstimator(unique, probabilities)
