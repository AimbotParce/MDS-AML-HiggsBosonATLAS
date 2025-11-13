from abc import ABC, abstractmethod
from typing import Annotated, Dict, Generic, List, Literal, Self, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray
from sklearn.naive_bayes import _BaseNB
from sklearn.utils.validation import validate_data

YType = TypeVar("YType", bound=np.generic)

Array1DFloat = Annotated[NDArray[np.float64], Literal["N"]]
Array1DInt = Annotated[NDArray[np.int64], Literal["N"]]


class ProbabilityEstimator(ABC):
    """
    Abstract base class for fitting probability distribution (or mass) function
    estimators to data.
    """

    def copy_with(self, **kwargs) -> "Self":
        """Create a copy of the estimator."""
        init_kwargs = {}
        for key in self.__class__.__init__.__code__.co_varnames[1:]:
            init_kwargs[key] = getattr(self, key)
        res = self.__class__(**init_kwargs)
        for key, value in kwargs.items():
            setattr(res, key, value)
        return res

    @abstractmethod
    def fit(self, X: Array1DFloat) -> "Self":
        """Fit the distribution to the data X. Returns a copy of self fitted to X."""
        pass

    @abstractmethod
    def predict(self, X: Array1DFloat) -> Array1DFloat:
        """Compute the probability estimations for the data X."""
        pass


class BespokeNB(_BaseNB, Generic[YType]):
    """
    Bespoke Naive Bayes classifier that allows the user to specify each feature's
    probability distribution (or mass) function fitting.
    """

    _priors: NDArray[np.float64]
    _fitted_estimators: List[List[ProbabilityEstimator]]  # [class][feature]
    num_features: int
    num_classes: int
    classes_: NDArray[YType]

    def __init__(self, estimators: Dict[int, ProbabilityEstimator]):
        super().__init__()
        self.estimators = estimators

    def fit(self, X: NDArray[np.float64], y: NDArray[YType]):
        """Fit Histogram Naive Bayes according to X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features. It cannot contain N/A
            values.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = validate_data(self, X, y, reset=True, ensure_all_finite="allow-nan")
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.num_features = X.shape[1]
        class_counts = np.zeros(self.num_classes, dtype=np.float64)
        self._fitted_estimators = [[None] * self.num_features for _ in range(self.num_classes)]

        for class_idx, y_i in enumerate(self.classes_):
            X_i = X[y == y_i, :]
            class_counts[class_idx] += X_i.shape[0]
            for feature_idx in range(self.num_features):
                self._fitted_estimators[class_idx][feature_idx] = self.estimators[feature_idx].fit(X_i[:, feature_idx])

        self._priors = class_counts / class_counts.sum()

        return self

    def _joint_log_likelihood(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the unnormalized posterior log probability of X."""
        X = self._check_X(X)
        joint_log_likelihood = []
        for class_index in range(self.num_classes):
            log_likelihood = np.log(self._priors[class_index]) * np.ones(X.shape[0])
            for feature_idx in range(self.num_features):
                likelihoods = self._fitted_estimators[class_index][feature_idx].predict(X[:, feature_idx])
                likelihoods[likelihoods == 0] += 1e-9  # Avoid log(0)
                log_likelihood += np.log(likelihoods)
            joint_log_likelihood.append(log_likelihood)
        return np.array(joint_log_likelihood).T

    def _check_X(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Validate X against the training data."""
        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
        if X.shape[1] != self.num_features:
            raise ValueError(f"X has {X.shape[1]} features, but BespokeNB is expecting {self.num_features} features.")
        return X


class CategoricalAwareBespokeNB(_BaseNB, Generic[YType]):
    """
    Bespoke Naive Bayes classifier that allows the user to specify each feature's
    probability distribution (or mass) function fitting.

    This implementation allows for dependencies between the continuous features
    and a set of categorical features by creating separate probability
    distribution estimations conditioned on the values of said categorical
    features. Note that this increases the model complexity a lot, and can incur
    in sparsity issues if the categorical features have many distinct values.
    One should only use this model when there are a small (VERY small) number of
    categorical feature combinations, all their combinations appear in the data,
    and they have sufficient samples to estimate reliable distributions.

    To make it more robust, Laplace smoothing is also applied to the categorical
    feature combinations.
    """

    _priors: NDArray[np.float64]
    _fitted_estimators: List[List[List[ProbabilityEstimator]]]  # [class][categorical_combination][feature]
    num_features: int
    num_classes: int
    classes_: NDArray[YType]
    _categorical_combinations: NDArray[np.float64]
    _categorical_combination_probabilities: NDArray[np.float64]

    def __init__(
        self,
        estimators: Dict[int, ProbabilityEstimator],
        categorical_features: List[int],
        laplace_smoothing: float = 1e-9,
    ):
        super().__init__()
        self.estimators = estimators
        self.categorical_features = categorical_features
        self.laplace_smoothing = laplace_smoothing

    def fit(self, X: NDArray[np.float64], y: NDArray[YType]):
        """Fit Histogram Naive Bayes according to X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features. It cannot contain N/A
            values.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = validate_data(self, X, y, reset=True, ensure_all_finite="allow-nan")
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.num_features = X.shape[1]
        class_counts = np.zeros(self.num_classes, dtype=np.float64)

        cat_combinations = X[:, self.categorical_features]
        self._categorical_combinations = np.unique(cat_combinations, axis=0)
        self._fitted_estimators = [  # [class][cat_comb][feature] (there will be None in categorical features)
            [[None] * self.num_features for _ in self._categorical_combinations] for _ in range(self.num_classes)
        ]
        self._categorical_combination_probabilities = np.zeros(
            (self.num_classes, len(self._categorical_combinations)), dtype=np.float64
        )

        for class_idx, y_i in enumerate(self.classes_):
            X_i = X[y == y_i, :]
            class_counts[class_idx] += X_i.shape[0]
            for combination_idx, combination in enumerate(self._categorical_combinations):
                combination_mask = np.all(X_i[:, self.categorical_features] == combination, axis=1)
                self._categorical_combination_probabilities[class_idx, combination_idx] = (
                    np.mean(combination_mask, axis=0) + self.laplace_smoothing
                ) / (1 + len(self._categorical_combinations) * self.laplace_smoothing)
                # Fit a separate estimator for each combination of categorical features
                X_i_combination = X_i[combination_mask, :]
                for feature_idx in range(self.num_features):
                    if feature_idx in self.categorical_features:
                        continue  # Skip categorical features
                    self._fitted_estimators[class_idx][combination_idx][feature_idx] = self.estimators[feature_idx].fit(
                        X_i_combination[:, feature_idx]
                    )

        self._priors = class_counts / class_counts.sum()

        return self

    def _joint_log_likelihood(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the unnormalized posterior log probability of X."""
        X = self._check_X(X)
        joint_log_likelihood = []
        for class_index in range(self.num_classes):
            log_likelihood = np.log(self._priors[class_index]) * np.ones(X.shape[0])
            for combination_idx, combination in enumerate(self._categorical_combinations):
                combination_mask = np.all(X[:, self.categorical_features] == combination, axis=1)
                log_likelihood[combination_mask] += np.log(
                    self._categorical_combination_probabilities[class_index, combination_idx]
                )
                for feature_idx in range(self.num_features):
                    if feature_idx in self.categorical_features:
                        continue  # Skip categorical features
                    likelihoods = self._fitted_estimators[class_index][combination_idx][feature_idx].predict(
                        X[combination_mask, feature_idx]
                    )
                    likelihoods[likelihoods == 0] += 1e-9  # Avoid log(0)
                    log_likelihood[combination_mask] += np.log(likelihoods)
            joint_log_likelihood.append(log_likelihood)
        return np.array(joint_log_likelihood).T

    def _check_X(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Validate X against the training data."""
        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
        if X.shape[1] != self.num_features:
            raise ValueError(f"X has {X.shape[1]} features, but BespokeNB is expecting {self.num_features} features.")
        return X


from .box_cox_gaussian_estimator import (
    BoxCoxGaussianEstimator,
    RobustBoxCoxGaussianEstimator,
)
from .categorical_estimator import CategoricalEstimator, RobustCategoricalEstimator
from .gaussian_estimator import GaussianEstimator, RobustGaussianEstimator
from .histogram_estimator import HistogramEstimator, RobustHistogramEstimator
from .kde_estimators import (
    EagerGaussianKDEstimator,
    GaussianKDEstimator,
    RobustEagerGaussianKDEstimator,
    RobustGaussianKDEstimator,
)

__all__ = [
    "BespokeNB",
    "CategoricalAwareBespokeNB",
    "ProbabilityEstimator",
    "CategoricalEstimator",
    "HistogramEstimator",
    "GaussianKDEstimator",
    "EagerGaussianKDEstimator",
    "RobustHistogramEstimator",
    "RobustGaussianKDEstimator",
    "RobustEagerGaussianKDEstimator",
    "RobustCategoricalEstimator",
    "GaussianEstimator",
    "RobustGaussianEstimator",
    "BoxCoxGaussianEstimator",
    "RobustBoxCoxGaussianEstimator",
]
