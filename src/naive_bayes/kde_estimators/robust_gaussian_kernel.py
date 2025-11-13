from . import FittedRobustKDEstimatorMixin, RobustKDEstimatorBase
from .gaussian_kernel import _GaussianKernel


class FittedRobustGaussianKDEstimator(FittedRobustKDEstimatorMixin, _GaussianKernel):
    """
    Fitted robust gaussian kernel density estimator for a single feature.

    This implementation is designed to be completely robust to missing values
    both in the training and prediction phases. If a feature is missing in the
    training stage, its missingness is considered as a separate "bin" in the
    histogram. This allows the model to learn from the absence of data as well.
    """

    pass


class RobustGaussianKDEstimator(RobustKDEstimatorBase):
    """
    Robust gaussian kernel density estimator for a single feature.

    This implementation is designed to be completely robust to missing values
    both in the training and prediction phases. If a feature is missing in the
    training stage, its missingness is considered as a separate "bin" in the
    histogram. This allows the model to learn from the absence of data as well.
    """

    def __init__(self, bandwidth: float = None, laplace_smoothing: float = 1e-9):
        self.bandwidth = bandwidth
        self.laplace_smoothing = laplace_smoothing

    def _create_fitted_estimator(self, non_nan_X, bandwidth, nan_probability):
        return FittedRobustGaussianKDEstimator(non_nan_X, bandwidth, nan_probability)
