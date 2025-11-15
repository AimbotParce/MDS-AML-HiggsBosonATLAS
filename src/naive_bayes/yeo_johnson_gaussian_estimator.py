from typing import Generic, List, Optional, Self, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from . import Array1DFloat, ProbabilityEstimator


def _gaussian_pdf(mean, std, x: Array1DFloat) -> Array1DFloat:
    """Compute the Gaussian probability density function."""
    coeff = 1 / (std * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mean) / std) ** 2
    return coeff * np.exp(exponent)


class YeoJohnsonGaussianEstimator(ProbabilityEstimator):
    """
    Yeo-Johnson Gaussian estimator for a single feature. This estimator assumes that,
    under a yeo-johnson transformation, the data follows a Gaussian
    distribution and estimates the parameters (lambda, mean and standard deviation)
    from the data.

    This estimator does not support NaN values in the input data.
    """

    _mean: float
    _std: float
    _lambda: float

    def __init__(self):
        """
        Initialize the GaussianEstimator.
        """
        pass

    def fit(self, X: Array1DFloat) -> Self:
        """Fit the histogram to the data X."""
        if np.isnan(X).any():
            raise ValueError("GaussianEstimator does not support NaN values in the input data.")

        transformed_X, lambda_ = stats.yeojohnson(X + 1e-6)  # small constant to avoid zero values
        mean = np.mean(transformed_X)
        std = np.std(transformed_X)
        return self.copy_with(_mean=mean, _std=std, _lambda=lambda_)

    def predict(self, X: Array1DFloat) -> Array1DFloat:
        """Compute the probability estimations for the data X."""
        if np.isnan(X).any():
            raise ValueError("GaussianEstimator does not support NaN values in the input data.")
        transformed_X = stats.yeojohnson(X + 1e-6, lmbda=self._lambda)
        return _gaussian_pdf(self._mean, self._std, transformed_X)


class RobustYeoJohnsonGaussianEstimator(ProbabilityEstimator):
    """
    Robust Yeo-Johnson Gaussian estimator for a single feature. This estimator assumes that,
    under a yeo-johnson transformation, the data follows a Gaussian
    distribution and estimates the parameters (lambda, mean and standard deviation)
    from the data.

    This implementation is designed to be completely robust to missing values
    both in the training and prediction phases. If a feature is missing in the
    training stage, its missingness is considered as a separate "bin" in the
    histogram. This allows the model to learn from the absence of data as well.
    """

    _mean: float
    _std: float
    _lambda: float
    _nan_probability: float

    def __init__(self, laplace_smoothing: float = 1e-9):
        self.laplace_smoothing = laplace_smoothing

    def fit(self, X: Array1DFloat) -> Self:
        """Fit the histogram to the data X."""
        nan_mask = np.isnan(X)
        nan_probability = (np.mean(nan_mask, axis=0) + self.laplace_smoothing) / (1 + 2 * self.laplace_smoothing)
        non_nan_X = X[~nan_mask]
        transformed_X, lambda_ = stats.yeojohnson(non_nan_X + 1e-6)  # small constant to avoid zero values
        mean = np.mean(transformed_X)
        std = np.std(transformed_X)
        return self.copy_with(_mean=mean, _std=std, _nan_probability=nan_probability, _lambda=lambda_)

    def predict(self, X: Array1DFloat) -> Array1DFloat:
        """Compute the probability estimations for the data X."""
        nan_mask = np.isnan(X)
        likelihoods = np.zeros_like(X)
        likelihoods[nan_mask] = self._nan_probability
        transformed_X = stats.yeojohnson(X[~nan_mask] + 1e-6, lmbda=self._lambda)
        likelihoods[~nan_mask] = _gaussian_pdf(self._mean, self._std, transformed_X)
        return likelihoods
