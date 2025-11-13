import math
import warnings
from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Self, TypeVar

import numpy as np
from numpy.typing import NDArray

from .. import Array1DFloat, ProbabilityEstimator


def silverman_bandwidth_rule_of_thumb(X: Array1DFloat) -> float:
    """Compute the bandwidth using Silverman's rule of thumb."""
    if len(X) < 2:
        warnings.warn("Not enough data points to compute bandwidth. Using arbitrary value of 1.0.")
        return 1.0  # Arbitrary bandwidth for single data point
    std_dev = np.std(X, ddof=1)
    n = len(X)
    return 1.06 * std_dev * n ** (-1 / 5)


class KDEstimatorBase(ProbabilityEstimator, ABC):
    """
    Kernel density estimator base class for a single feature.

    This estimator does not support NaN values in the input data.

    Note: This is a Lazy implementation, which computes the density estimation
    on-the-fly during prediction.
    """

    _X: Array1DFloat
    _bandwidth: float

    def __init__(self, bandwidth: Optional[float] = None):
        """
        Initialize the KDEstimatorBase.
        """
        self.bandwidth = bandwidth

    @staticmethod
    @abstractmethod
    def _kernel(u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Kernel function."""
        pass

    def fit(self, X: Array1DFloat):
        """Fit the histogram to the data X."""
        if np.isnan(X).any():
            raise ValueError("GaussianKDEstimator does not support NaN values in the input data.")
        if self.bandwidth is None:
            bandwidth = silverman_bandwidth_rule_of_thumb(X)
        else:
            bandwidth = self.bandwidth

        return self.copy_with(_X=X, _bandwidth=bandwidth)

    def predict(self, X: Array1DFloat) -> Array1DFloat:
        """Compute the probability estimations for the data X."""
        if np.isnan(X).any():
            raise ValueError("FittedKDEstimator does not support NaN values in the input data.")
        distances = (X[:, np.newaxis] - self._X[np.newaxis, :]) / self._bandwidth
        kernel_values = self._kernel(distances)
        return np.sum(kernel_values, axis=1) / (self._X.shape[0] * self._bandwidth)


class EagerKDEstimatorBase(ProbabilityEstimator, ABC):
    """
    Kernel density estimator base class for a single feature.

    This estimator does not support NaN values in the input data.

    Note: This is an Eager implementation, which precomputes the density estimation
    during the fitting phase.
    """

    _bin_edges: Array1DFloat
    _density: Array1DFloat

    def __init__(
        self,
        bandwidth: Optional[float] = None,
        num_points: int = 1000,
        range_padding: float = 0.1,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize the KDEstimatorBase.

        Args:
            bandwidth: Bandwidth for the kernel density estimation. If None, Silverman's rule of thumb is used.
            num_points: Number of points to precompute the density estimation.
            range_padding: Padding to add to the min and max of the data range. As a fraction of the iqr.
            batch_size: If specified, compute the density estimation in batches to save memory.
        """
        self.bandwidth = bandwidth
        self.num_points = num_points
        self.range_padding = range_padding
        self.batch_size = batch_size

    @staticmethod
    @abstractmethod
    def _kernel(u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Kernel function."""
        pass

    def fit(self, X: Array1DFloat):
        """Fit the histogram to the data X."""
        if np.isnan(X).any():
            raise ValueError("GaussianKDEstimator does not support NaN values in the input data.")
        if self.bandwidth is None:
            bandwidth = silverman_bandwidth_rule_of_thumb(X)
        else:
            bandwidth = self.bandwidth

        iqr = np.subtract(*np.percentile(X, [75, 25]))
        min_x, max_x = (np.min(X) - self.range_padding * iqr, np.max(X) + self.range_padding * iqr)
        bin_edges = np.linspace(min_x, max_x, self.num_points + 1)
        midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        if self.batch_size is not None:
            density = None
            for batch in np.array_split(midpoints, math.ceil(midpoints.shape[0] / self.batch_size)):
                distances = (batch[:, np.newaxis] - X[np.newaxis, :]) / bandwidth
                kernel_values = self._kernel(distances)
                batch_density = np.sum(kernel_values, axis=1) / (X.shape[0] * bandwidth)
                if density is None:
                    density = batch_density
                else:
                    density = np.concatenate([density, batch_density])

        else:
            distances = (midpoints[:, np.newaxis] - X[np.newaxis, :]) / bandwidth
            kernel_values = self._kernel(distances)
            density = np.sum(kernel_values, axis=1) / (X.shape[0] * bandwidth)

        return self.copy_with(_bin_edges=bin_edges, _density=density)

    def predict(self, X: Array1DFloat) -> Array1DFloat:
        """Compute the probability estimations for the data X."""
        if np.isnan(X).any():
            raise ValueError("FittedKDEstimator does not support NaN values in the input data.")
        bin_indices = np.digitize(X, self._bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(self._density) - 1)
        return self._density[bin_indices]


class RobustKDEstimatorBase(ProbabilityEstimator, ABC):
    """
    Robust kernel density estimator base class for a single feature.

    This implementation is designed to be completely robust to missing values
    both in the training and prediction phases. If a feature is missing in the
    training stage, its missingness is considered as a separate "component" in the
    density estimation. This allows the model to learn from the absence of data as well.

    Note: This is a Lazy implementation, which computes the density estimation
    on-the-fly during prediction.
    """

    _nan_probability: float
    _non_nan_X: Array1DFloat
    _bandwidth: float

    def __init__(self, bandwidth: Optional[float] = None, laplace_smoothing: float = 1e-9):
        """
        Initialize the RobustKDEstimatorBase.
        """
        self.bandwidth = bandwidth
        self.laplace_smoothing = laplace_smoothing

    @staticmethod
    @abstractmethod
    def _kernel(u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Kernel function."""
        pass

    def fit(self, X: Array1DFloat):
        """Fit the histogram to the data X."""
        nan_mask = np.isnan(X)
        nan_probability = (np.mean(nan_mask, axis=0) + self.laplace_smoothing) / (1 + 2 * self.laplace_smoothing)
        non_nan_X = X[~nan_mask]
        if self.bandwidth is None:
            bandwidth = silverman_bandwidth_rule_of_thumb(X)
        else:
            bandwidth = self.bandwidth

        return self.copy_with(_non_nan_X=non_nan_X, _nan_probability=nan_probability, _bandwidth=bandwidth)

    def predict(self, X: Array1DFloat) -> Array1DFloat:
        """Compute the probability estimations for the data X."""
        nan_mask = np.isnan(X)
        likelihoods = np.zeros_like(X)
        likelihoods[nan_mask] = self._nan_probability
        distances = (X[:, np.newaxis] - self._non_nan_X[np.newaxis, :]) / self._bandwidth
        kernel_values = self._kernel(distances)
        likelihoods[~nan_mask] = (
            np.sum(kernel_values, axis=1) * (1 - self._nan_probability) / (self._non_nan_X.shape[0] * self._bandwidth)
        )
        return likelihoods


class RobustEagerKDEstimatorBase(ProbabilityEstimator, ABC):
    """
    Kernel density estimator base class for a single feature.

    This estimator does support NaN values in the input data.

    Note: This is an Eager implementation, which precomputes the density estimation
    during the fitting phase.
    """

    _bin_edges: Array1DFloat
    _density: Array1DFloat
    _nan_probability: float

    def __init__(
        self,
        bandwidth: Optional[float] = None,
        num_points: int = 1000,
        range_padding: float = 0.1,
        laplace_smoothing: float = 1e-9,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize the KDEstimatorBase.

        Args:
            bandwidth: Bandwidth for the kernel density estimation. If None, Silverman's rule of thumb is used.
            num_points: Number of points to precompute the density estimation.
            range_padding: Padding to add to the min and max of the data range. As a fraction of the iqr.
        """
        self.bandwidth = bandwidth
        self.num_points = num_points
        self.range_padding = range_padding
        self.laplace_smoothing = laplace_smoothing
        self.batch_size = batch_size

    @staticmethod
    @abstractmethod
    def _kernel(u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Kernel function."""
        pass

    def fit(self, X: Array1DFloat):
        """Fit the histogram to the data X."""
        nan_mask = np.isnan(X)
        non_nan_X = X[~nan_mask]
        nan_probability = (np.mean(nan_mask, axis=0) + self.laplace_smoothing) / (1 + 2 * self.laplace_smoothing)

        if self.bandwidth is None:
            bandwidth = silverman_bandwidth_rule_of_thumb(non_nan_X)
        else:
            bandwidth = self.bandwidth

        if len(non_nan_X) == 0:
            warnings.warn("All data points are NaN. Density estimation cannot be computed. Using empty density.")
            return self.copy_with(_bin_edges=np.array([]), _density=np.array([]), _nan_probability=nan_probability)

        iqr = np.subtract(*np.percentile(non_nan_X, [75, 25]))
        min_x, max_x = (np.min(non_nan_X) - self.range_padding * iqr, np.max(non_nan_X) + self.range_padding * iqr)
        bin_edges = np.linspace(min_x, max_x, self.num_points + 1)
        midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        if self.batch_size is not None:
            density = None
            for batch in np.array_split(midpoints, math.ceil(midpoints.shape[0] / self.batch_size)):
                distances = (batch[:, np.newaxis] - non_nan_X[np.newaxis, :]) / bandwidth
                kernel_values = self._kernel(distances)
                batch_density = np.sum(kernel_values, axis=1) / (non_nan_X.shape[0] * bandwidth)
                if density is None:
                    density = batch_density
                else:
                    density = np.concatenate([density, batch_density])

        else:
            distances = (midpoints[:, np.newaxis] - non_nan_X[np.newaxis, :]) / bandwidth
            kernel_values = self._kernel(distances)
            density = np.sum(kernel_values, axis=1) / (non_nan_X.shape[0] * bandwidth)

        return self.copy_with(_bin_edges=bin_edges, _density=density, _nan_probability=nan_probability)

    def predict(self, X: Array1DFloat) -> Array1DFloat:
        """Compute the probability estimations for the data X."""
        nan_mask = np.isnan(X)
        likelihoods = np.zeros_like(X)
        likelihoods[nan_mask] = self._nan_probability

        bin_indices = np.digitize(X[~nan_mask], self._bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(self._density) - 1)
        likelihoods[~nan_mask] = self._density[bin_indices] * (1 - self._nan_probability)
        return likelihoods


from .gaussian_kernel import (
    EagerGaussianKDEstimator,
    GaussianKDEstimator,
    RobustEagerGaussianKDEstimator,
    RobustGaussianKDEstimator,
)
