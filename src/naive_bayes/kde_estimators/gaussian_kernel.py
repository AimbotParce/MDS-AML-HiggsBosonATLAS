import numpy as np
from numpy.typing import NDArray

from . import FittedKDEstimatorMixin, KDEstimatorBase, KernelMixinBase


class _GaussianKernel(KernelMixinBase):
    """
    Fitted gaussian kernel density estimator for a single feature.

    This estimator does not support NaN values in the input data.
    """

    def _kernel(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Gaussian kernel function."""
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)


class FittedGaussianKDEstimator(FittedKDEstimatorMixin, _GaussianKernel):
    """
    Fitted gaussian kernel density estimator for a single feature.

    This estimator does not support NaN values in the input data.
    """

    pass


class GaussianKDEstimator(KDEstimatorBase):
    """
    Gaussian kernel density estimator for a single feature.

    This estimator does not support NaN values in the input data.
    """

    def _create_fitted_estimator(self, non_nan_X, bandwidth):
        return FittedGaussianKDEstimator(non_nan_X, bandwidth)
