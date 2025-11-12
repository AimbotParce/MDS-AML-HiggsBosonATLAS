from typing import Generic, List, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

from . import Array1DFloat, FittedProbabilityEstimator, ProbabilityEstimator


class FittedRobustHistogramEstimator(FittedProbabilityEstimator):
    """
    Fitted histogram-based estimator for a single feature.

    This implementation is designed to be completely robust to missing values
    both in the training and prediction phases. If a feature is missing in the
    training stage, its missingness is considered as a separate "bin" in the
    histogram. This allows the model to learn from the absence of data as well.
    """

    def __init__(self, histogram: NDArray[np.float64], bin_edges: NDArray[np.float64], nan_probability: float):
        self.histogram = histogram
        self.bin_edges = bin_edges
        self.nan_probability = nan_probability

    def predict(self, X: Array1DFloat) -> Array1DFloat:
        """Compute the probability estimations for the data X."""
        nan_mask = np.isnan(X)
        likelihoods = np.zeros_like(X)
        likelihoods[nan_mask] = self.nan_probability

        bin_indices = np.digitize(X[~nan_mask], self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(self.histogram) - 1)
        likelihoods[~nan_mask] = (
            self.histogram[bin_indices] * np.diff(self.bin_edges)[bin_indices] * (1 - self.nan_probability)
        )
        return likelihoods


class RobustHistogramEstimator(ProbabilityEstimator):
    """
    Histogram-based estimator for a single feature.

    This implementation is designed to be completely robust to missing values
    both in the training and prediction phases. If a feature is missing in the
    training stage, its missingness is considered as a separate "bin" in the
    histogram. This allows the model to learn from the absence of data as well.
    """

    def __init__(self, bins: Optional[int] = None, laplace_smoothing: float = 1e-9):
        self.bins = bins
        self.laplace_smoothing = laplace_smoothing

    def fit(self, X: Array1DFloat) -> FittedRobustHistogramEstimator:
        """Fit the histogram to the data X."""
        nan_mask = np.isnan(X)
        nan_probability = (np.mean(nan_mask, axis=0) + self.laplace_smoothing) / (1 + 2 * self.laplace_smoothing)
        non_nan_X = X[~nan_mask]
        if self.bins is None:
            bin_count = int(np.sqrt(len(non_nan_X)))
        else:
            bin_count = self.bins
        histogram, bin_edges = np.histogram(non_nan_X, bins=bin_count, density=True)
        return FittedRobustHistogramEstimator(histogram, bin_edges, nan_probability)
