"""Evaluation helpers (printing metrics and AMS used in the notebook)."""

from typing import Sequence, Union

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def ams_score(
    y_true: Sequence[Union[int, str]], y_pred: Sequence[Union[int, str]], weights: Sequence[float], br: float = 10.0
) -> float:
    """
    Compute the Approximate Median Significance (AMS) metric.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels, where 1 or 's' indicates signal, 0 or 'b' indicates background.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (same encoding as y_true).
    weights : array-like of shape (n_samples,)
        Event weights for each observation.
    br : float, default=10.0
        Regularization term (background regularization constant).

    Returns
    -------
    ams : float
        The AMS metric value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)

    # normalize labels to 's'/'b'
    def to_sb(a):
        if np.issubdtype(a.dtype, np.number):
            return np.where(a == 1, "s", "b")
        return a

    y_true_sb = to_sb(y_true)
    y_pred_sb = to_sb(y_pred)

    s = np.sum(weights[(y_true_sb == "s") & (y_pred_sb == "s")])
    b = np.sum(weights[(y_true_sb == "b") & (y_pred_sb == "s")])

    if b + br <= 0:
        return 0.0
    rad = 2 * ((s + b + br) * np.log(1.0 + s / (b + br)) - s)
    return np.sqrt(rad) if rad > 0 else 0.0


def report_metrics(y_true_num, y_pred_num, y_proba, weights=None):
    """Print common metrics. y_true_num / y_pred_num should be 0/1.
    y_proba is probability for class 1.
    """
    print("Accuracy:", accuracy_score(y_true_num, y_pred_num))
    print(
        "Classification report:\n", classification_report(y_true_num, y_pred_num, target_names=["background", "signal"])
    )
    try:
        print("ROC AUC:", roc_auc_score(y_true_num, y_proba))
    except Exception:
        pass
    if weights is not None:
        print("AMS:", ams_score(y_true_num, y_pred_num, weights))


if __name__ == "__main__":
    # tiny smoke test
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 1, 0]
    proba = [0.9, 0.1, 0.8, 0.6, 0.4]
    w = [1.0] * len(y_true)
    report_metrics(y_true, y_pred, proba, w)
    w = [1.0] * len(y_true)
    report_metrics(y_true, y_pred, proba, w)
    w = [1.0] * len(y_true)
    report_metrics(y_true, y_pred, proba, w)
