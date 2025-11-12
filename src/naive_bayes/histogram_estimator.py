from typing import Generic, List, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

from . import Array1DFloat, FittedProbabilityEstimator, ProbabilityEstimator


class FittedHistogramEstimator(FittedProbabilityEstimator):
    """
    Fitted histogram-based estimator for a single feature.

    This estimator does not support NaN values in the input data.
    """

    def __init__(self, histogram: NDArray[np.float64], bin_edges: NDArray[np.float64]):
        self.histogram = histogram
        self.bin_edges = bin_edges

    def predict(self, X: Array1DFloat) -> Array1DFloat:
        """Compute the probability estimations for the data X."""
        if np.isnan(X).any():
            raise ValueError("FittedHistogramEstimator does not support NaN values in the input data.")
        bin_indices = np.digitize(X, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(self.histogram) - 1)
        return self.histogram[bin_indices] * np.diff(self.bin_edges)[bin_indices]


class HistogramEstimator(ProbabilityEstimator):
    """
    Histogram-based estimator for a single feature.

    This estimator does not support NaN values in the input data.
    """

    def __init__(self, bins: Optional[int] = None):
        """
        Initialize the HistogramEstimator.

        Parameters:
        bins (Optional[int]): The number of bins to use for the histogram. If None, it will be
            determined automatically during fitting using the square root of the number of samples.
        """
        self.bins = bins

    def fit(self, X: Array1DFloat) -> FittedHistogramEstimator:
        """Fit the histogram to the data X."""
        if np.isnan(X).any():
            raise ValueError("HistogramEstimator does not support NaN values in the input data.")
        if self.bins is None:
            bin_count = int(np.sqrt(len(X)))
        else:
            bin_count = self.bins
        histogram, bin_edges = np.histogram(X, bins=bin_count, density=True)
        return FittedHistogramEstimator(histogram, bin_edges)
