from typing import Generic, List, Optional, Self, TypeVar

import numpy as np
from numpy.typing import NDArray

from . import Array1DFloat, ProbabilityEstimator


class HistogramEstimator(ProbabilityEstimator):
    """
    Histogram-based estimator for a single feature.

    This estimator does not support NaN values in the input data.
    """

    _histogram: NDArray[np.float64]
    _bin_edges: NDArray[np.float64]

    def __init__(self, bins: Optional[int] = None):
        """
        Initialize the HistogramEstimator.

        Parameters:
        bins (Optional[int]): The number of bins to use for the histogram. If None, it will be
            determined automatically during fitting using the square root of the number of samples.
        """
        self.bins = bins

    def fit(self, X: Array1DFloat) -> Self:
        """Fit the histogram to the data X."""
        if np.isnan(X).any():
            raise ValueError("HistogramEstimator does not support NaN values in the input data.")
        if self.bins is None:
            bin_count = int(np.sqrt(len(X)))
        else:
            bin_count = self.bins
        histogram, bin_edges = np.histogram(X, bins=bin_count, density=True)
        return self.copy_with(_histogram=histogram, _bin_edges=bin_edges)

    def predict(self, X: Array1DFloat) -> Array1DFloat:
        """Compute the probability estimations for the data X."""
        if np.isnan(X).any():
            raise ValueError("FittedHistogramEstimator does not support NaN values in the input data.")
        bin_indices = np.digitize(X, self._bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(self._histogram) - 1)
        return self._histogram[bin_indices] * np.diff(self._bin_edges)[bin_indices]


class RobustHistogramEstimator(ProbabilityEstimator):
    """
    Histogram-based estimator for a single feature.

    This implementation is designed to be completely robust to missing values
    both in the training and prediction phases. If a feature is missing in the
    training stage, its missingness is considered as a separate "bin" in the
    histogram. This allows the model to learn from the absence of data as well.
    """

    _histogram: NDArray[np.float64]
    _bin_edges: NDArray[np.float64]
    _nan_probability: float

    def __init__(self, bins: Optional[int] = None, laplace_smoothing: float = 1e-9):
        self.bins = bins
        self.laplace_smoothing = laplace_smoothing

    def fit(self, X: Array1DFloat) -> Self:
        """Fit the histogram to the data X."""
        nan_mask = np.isnan(X)
        nan_probability = (np.mean(nan_mask, axis=0) + self.laplace_smoothing) / (1 + 2 * self.laplace_smoothing)
        non_nan_X = X[~nan_mask]

        if non_nan_X.size == 0:
            # All values are NaN, so we will assume arbitrary parameters for the Histogram. If in inference time
            # a non-NaN value is given, this will predict probability zero, which would give an error (on the log),
            # but the BespokeNB already handles this case by adding a very small probability to all classes.
            return self.copy_with(_histogram=np.array([1.0]), _bin_edges=np.array([0.0, 1.0]), _nan_probability=1.0)

        if self.bins is None:
            bin_count = int(np.sqrt(len(non_nan_X)))
        else:
            bin_count = self.bins
        histogram, bin_edges = np.histogram(non_nan_X, bins=bin_count, density=True)
        return self.copy_with(_histogram=histogram, _bin_edges=bin_edges, _nan_probability=nan_probability)

    def predict(self, X: Array1DFloat) -> Array1DFloat:
        """Compute the probability estimations for the data X."""
        nan_mask = np.isnan(X)
        likelihoods = np.zeros_like(X)
        likelihoods[nan_mask] = self._nan_probability

        bin_indices = np.digitize(X[~nan_mask], self._bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(self._histogram) - 1)
        likelihoods[~nan_mask] = (
            self._histogram[bin_indices] * np.diff(self._bin_edges)[bin_indices] * (1 - self._nan_probability)
        )
        return likelihoods
