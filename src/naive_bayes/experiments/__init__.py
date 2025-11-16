import os
import warnings
from typing import Dict, List, Literal, Optional, Self, Set, Tuple, Type, TypeVar, Union

from numpy.typing import NDArray
from pydantic import BaseModel, model_validator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from tqdm import tqdm

from ... import naive_bayes
from ...evaluate import ams_score


class ExperimentResult(BaseModel):
    accuracy: float
    b_recall: float
    b_precision: float
    b_f1_score: float
    s_recall: float
    s_precision: float
    s_f1_score: float
    ams_score: float


class ExperimentBase(BaseModel):
    model_class: Literal["BespokeNB", "CategoricalAwareBespokeNB"]
    categorical_estimator_class: Literal["CategoricalEstimator", "RobustCategoricalEstimator"]
    continuous_estimator_class: Literal[
        "GaussianEstimator",
        "RobustGaussianEstimator",
        "HistogramEstimator",
        "RobustHistogramEstimator",
        "EagerGaussianKDEstimator",
        "RobustEagerGaussianKDEstimator",
        "YeoJohnsonGaussianEstimator",
        "RobustYeoJohnsonGaussianEstimator",
    ]
    dataset: str
    categorical_estimator_params: Dict[str, Union[Optional[float], Optional[int]]] = {}
    continuous_estimator_params: Dict[str, Union[Optional[float], Optional[int]]] = {}

    result: Optional[ExperimentResult] = None

    def __hash__(self):
        return hash(
            (
                self.model_class,
                self.categorical_estimator_class,
                self.continuous_estimator_class,
                frozenset(self.categorical_estimator_params.items()),
                frozenset(self.continuous_estimator_params.items()),
                self.dataset,
                *self.custom_hash_items(),
            )
        )

    def custom_hash_items(self) -> Tuple:
        """Override this method in subclasses to add additional hash items."""
        return ()

    def __eq__(self, other):
        if not isinstance(other, ExperimentBase):
            return NotImplemented
        return (
            self.model_class == other.model_class
            and self.categorical_estimator_class == other.categorical_estimator_class
            and self.continuous_estimator_class == other.continuous_estimator_class
            and self.categorical_estimator_params == other.categorical_estimator_params
            and self.continuous_estimator_params == other.continuous_estimator_params
            and self.dataset == other.dataset
            and self.custom_equals(other)
        )

    def custom_equals(self, other: "Self") -> bool:
        """Override this method in subclasses to add additional equality checks."""
        return True


def _instantiate_estimator(
    estimator_cls: Type[naive_bayes.ProbabilityEstimator],
    estimator_params: Dict[str, Union[Optional[float], Optional[int]]],
) -> naive_bayes.ProbabilityEstimator:
    init_kwargs = {}
    for key in estimator_cls.__init__.__code__.co_varnames[1:]:
        if not key in estimator_params:
            raise ValueError(f"Missing value for estimator parameter: {key}")
        init_kwargs[key] = estimator_params[key]
    return estimator_cls(**init_kwargs)


def _get_estimator_instances(
    experiment: ExperimentBase, num_features: int, categorical_features: list[int]
) -> Dict[int, naive_bayes.ProbabilityEstimator]:
    categorical_estimator_cls = getattr(naive_bayes, experiment.categorical_estimator_class)
    continuous_estimator_cls = getattr(naive_bayes, experiment.continuous_estimator_class)
    instances = {}
    for feature in range(num_features):
        if feature in categorical_features:
            instances[feature] = _instantiate_estimator(
                categorical_estimator_cls, experiment.categorical_estimator_params
            )
        else:
            instances[feature] = _instantiate_estimator(
                continuous_estimator_cls, experiment.continuous_estimator_params
            )
    return instances


def _get_model_instance(
    experiment: ExperimentBase, num_features: int, categorical_features: list[int]
) -> naive_bayes.BespokeNB | naive_bayes.CategoricalAwareBespokeNB:
    model_cls = getattr(naive_bayes, experiment.model_class)
    estimators = _get_estimator_instances(
        experiment, num_features=num_features, categorical_features=categorical_features
    )
    if model_cls == naive_bayes.BespokeNB:
        return model_cls(estimators=estimators)
    elif model_cls == naive_bayes.CategoricalAwareBespokeNB:
        return model_cls(
            estimators=estimators,
            categorical_features=categorical_features,
        )
    else:
        raise ValueError(f"Unknown model class: {experiment.model_class}")


def store_experiment_set(experiments: Set[ExperimentBase], filename: str) -> None:
    """Store a set of Experiments to a JSONL file."""
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, "w") as f:
        for er in experiments:
            if not isinstance(er, (ExperimentBase)):
                raise ValueError(f"Invalid type in experiments set: {type(er)}")
            f.write(er.model_dump_json() + "\n")


def append_to_experiment_set(experiment: ExperimentBase, filename: str) -> None:
    """Append an Experiment to a JSONL file."""
    if not isinstance(experiment, (ExperimentBase)):
        raise ValueError(f"Invalid type in experiments set: {type(experiment)}")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, "a") as f:
        f.write(experiment.model_dump_json() + "\n")


E = TypeVar("E", bound=ExperimentBase)


def load_experiment_set(cls: Type[E], filename: str) -> Set[E]:
    """Load a set of Experiments from a JSONL file."""
    if not os.path.exists(filename):
        return set()
    experiments: Set[E] = set()
    with open(filename, "r") as f:
        for line in f:
            experiments.add(cls.model_validate_json(line.strip()))
    return experiments


def filter_out_of_experiment_set(cls: Type[E], filename: str, experiment_to_remove: E) -> None:
    """Remove an Experiments from a JSONL file."""
    if not os.path.exists(filename):
        return
    # Instead of reading everything, and rewriting everything, we can do it in a single pass.
    temp_filename = filename + ".tmp"
    with open(filename, "r") as f_in, open(temp_filename, "w") as f_out:
        for line in f_in:
            experiment = cls.model_validate_json(line.strip())
            if experiment != experiment_to_remove:
                f_out.write(line)
    os.replace(temp_filename, filename)


def run_experiment(
    experiment: ExperimentBase, datasets: Dict[str, Tuple[NDArray, NDArray, NDArray, List[int]]]
) -> None:
    X_train, y_train, weights_train, X_test, y_test, weights_test, categorical_features = (
        experiment._get_train_test_data(datasets)
    )

    model = _get_model_instance(experiment, num_features=X_train.shape[1], categorical_features=categorical_features)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    (b_precision, s_precision), (b_recall, s_recall), (b_f1_score, s_f1_score), _ = precision_recall_fscore_support(
        y_test, y_pred, labels=["b", "s"], average=None, zero_division=0
    )
    ams = ams_score(y_true=y_test, y_pred=y_pred, weights=weights_test)
    experiment.result = ExperimentResult(
        accuracy=accuracy,
        b_recall=b_recall,
        b_precision=b_precision,
        b_f1_score=b_f1_score,
        s_recall=s_recall,
        s_precision=s_precision,
        s_f1_score=s_f1_score,
        ams_score=ams,
    )


def run_all_experiments(
    experiments: Set[E],
    datasets: Dict[str, Tuple[NDArray, NDArray, NDArray, List[int]]],
    *,
    verbose: bool = True,
    results_file: Optional[os.PathLike] = None,
    failed_file: Optional[os.PathLike] = None,
) -> Tuple[Set[E], Set[E]]:
    results: Set[E] = set()
    failed: Set[E] = set()
    experiment_type = type(next(iter(experiments))) if experiments else ExperimentBase
    if results_file is not None:
        already_ran: Set[E] = load_experiment_set(experiment_type, results_file)
    if failed_file is not None:
        already_failed: Set[E] = load_experiment_set(experiment_type, failed_file)
    missing_experiments = experiments - already_ran  # Small optimization: If it was ran successfully already, skip it.
    bar = tqdm(total=len(missing_experiments), desc="Running experiments", disable=not verbose)
    try:
        for experiment in missing_experiments:
            if not isinstance(experiment, experiment_type):
                raise TypeError(f"Invalid type in experiments set: {type(experiment)}")
            # Skip experiments that have already been run
            if experiment in results:
                bar.update(1)
                continue
            bar.set_postfix_str(
                f"{experiment.model_class} with {experiment.continuous_estimator_class} on {experiment.dataset}"
            )
            bar.refresh()
            try:
                run_experiment(experiment, datasets)
            except Exception as e:
                import traceback

                traceback.print_exc()
                warnings.warn(f"Experiment failed: {e}")
                failed.add(experiment)
                if failed_file is not None and experiment not in already_failed:
                    append_to_experiment_set(experiment, failed_file)
            else:
                results.add(experiment)
                if results_file is not None:
                    append_to_experiment_set(experiment, results_file)
                if failed_file is not None and experiment in already_failed:
                    # If it was previously marked as failed, but now succeeded, we can remove it from the failed file.
                    already_failed.remove(experiment)
                    filter_out_of_experiment_set(experiment_type, failed_file, experiment)
            finally:
                bar.update(1)
    finally:
        bar.close()
    return results, failed
