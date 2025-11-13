import numpy as np
from numpy.typing import NDArray

from . import (
    EagerKDEstimatorBase,
    KDEstimatorBase,
    RobustEagerKDEstimatorBase,
    RobustKDEstimatorBase,
)


def _gaussian_kernel(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Gaussian kernel function."""
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)


class GaussianKDEstimator(KDEstimatorBase):
    """
    Gaussian kernel density estimator for a single feature.

    This estimator does not support NaN values in the input data.

    Note: This is a Lazy implementation, which computes the density estimation
    on-the-fly during prediction.
    """

    _kernel = staticmethod(_gaussian_kernel)


class RobustGaussianKDEstimator(RobustKDEstimatorBase):
    """
    Robust Gaussian kernel density estimator for a single feature.

    This estimator does support NaN values in the input data.

    Note: This is a Lazy implementation, which computes the density estimation
    on-the-fly during prediction.
    """

    _kernel = staticmethod(_gaussian_kernel)


class EagerGaussianKDEstimator(EagerKDEstimatorBase):
    """
    Eager Gaussian kernel density estimator for a single feature.

    This estimator does not support NaN values in the input data.

    Note: This is an Eager implementation, which precomputes the density estimation
    during the fitting phase.
    """

    _kernel = staticmethod(_gaussian_kernel)


class RobustEagerGaussianKDEstimator(RobustEagerKDEstimatorBase):
    """
    Robust Eager Gaussian kernel density estimator for a single feature.

    This estimator does support NaN values in the input data.

    Note: This is an Eager implementation, which precomputes the density estimation
    during the fitting phase.
    """

    _kernel = staticmethod(_gaussian_kernel)
