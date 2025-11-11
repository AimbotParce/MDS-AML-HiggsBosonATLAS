from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

from .. import Array1DFloat, FittedProbabilityEstimator, ProbabilityEstimator


class KernelMixinBase(ABC):
    """
    Mixin base class for kernel functions.
    """

    @abstractmethod
    def _kernel(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Kernel function."""
        pass


class FittedKDEstimatorMixin(FittedProbabilityEstimator):
    """
    Fitted kernel density estimator for a single feature.

    This estimator does not support NaN values in the input data.
    """

    def __init__(self, X: Array1DFloat, bandwidth: float):
        self.X = X
        self.bandwidth = bandwidth

    def predict(self, X: Array1DFloat) -> Array1DFloat:
        """Compute the probability estimations for the data X."""
        if np.isnan(X).any():
            raise ValueError("FittedKDEstimator does not support NaN values in the input data.")
        distances = (X[:, np.newaxis] - self.X[np.newaxis, :]) / self.bandwidth
        kernel_values = self._kernel(distances)
        return np.sum(kernel_values, axis=1) / (self.X.shape[0] * self.bandwidth)


class KDEstimatorBase(ProbabilityEstimator, ABC):
    """
    Kernel density estimator base class for a single feature.

    This estimator does not support NaN values in the input data.
    """

    def __init__(self, bandwidth: Optional[float] = None):
        """
        Initialize the KDEstimatorBase.
        """
        self.bandwidth = bandwidth

    @abstractmethod
    def _create_fitted_estimator(self, non_nan_X: Array1DFloat, bandwidth: float) -> FittedKDEstimatorMixin:
        pass

    def fit(self, X: Array1DFloat):
        """Fit the histogram to the data X."""
        if np.isnan(X).any():
            raise ValueError("GaussianKDEstimator does not support NaN values in the input data.")
        if self.bandwidth is None:
            # Silverman's rule of thumb for bandwidth selection
            std_dev = np.std(X, ddof=1)
            n = len(X)
            self.bandwidth = 1.06 * std_dev * n ** (-1 / 5)
        return self._create_fitted_estimator(X, self.bandwidth)


class FittedRobustKDEstimatorMixin(FittedProbabilityEstimator):
    """
    Fitted robust kernel density estimator for a single feature.

    This implementation is designed to be completely robust to missing values
    both in the training and prediction phases. If a feature is missing in the
    training stage, its missingness is considered as a separate "component" in the
    density estimation. This allows the model to learn from the absence of data as well.
    """

    def __init__(self, non_nan_X: Array1DFloat, bandwidth: float, nan_probability: float):
        self.non_nan_X = non_nan_X
        self.bandwidth = bandwidth
        self.nan_probability = nan_probability

    def predict(self, X: Array1DFloat) -> Array1DFloat:
        """Compute the probability estimations for the data X."""
        nan_mask = np.isnan(X)
        likelihoods = np.zeros_like(X)
        likelihoods[nan_mask] = self.nan_probability
        distances = (X[:, np.newaxis] - self.non_nan_X[np.newaxis, :]) / self.bandwidth
        kernel_values = self._kernel(distances)
        likelihoods[~nan_mask] = (
            np.sum(kernel_values, axis=1) * (1 - self.nan_probability) / (self.non_nan_X.shape[0] * self.bandwidth)
        )
        return likelihoods


class RobustKDEstimatorBase(ProbabilityEstimator, ABC):
    """
    Robust kernel density estimator base class for a single feature.

    This implementation is designed to be completely robust to missing values
    both in the training and prediction phases. If a feature is missing in the
    training stage, its missingness is considered as a separate "component" in the
    density estimation. This allows the model to learn from the absence of data as well.
    """

    def __init__(self, bandwidth: Optional[float] = None):
        """
        Initialize the RobustKDEstimatorBase.
        """
        self.bandwidth = bandwidth

    @abstractmethod
    def _create_fitted_estimator(
        self, non_nan_X: Array1DFloat, bandwidth: float, nan_probability: float
    ) -> FittedRobustKDEstimatorMixin:
        pass

    def fit(self, X: Array1DFloat):
        """Fit the histogram to the data X."""
        nan_mask = np.isnan(X)
        nan_probability = (np.mean(nan_mask, axis=0) + self.laplace_smoothing) / (1 + 2 * self.laplace_smoothing)
        non_nan_X = X[~nan_mask]
        if self.bandwidth is None:
            # Silverman's rule of thumb for bandwidth selection
            std_dev = np.std(non_nan_X, ddof=1)
            n = len(non_nan_X)
            self.bandwidth = 1.06 * std_dev * n ** (-1 / 5)
        return self._create_fitted_estimator(non_nan_X, self.bandwidth, nan_probability)


from .gaussian_kernel import GaussianKDEstimator, RobustGaussianKDEstimator

__all__ = ["GaussianKDEstimator", "RobustGaussianKDEstimator"]
