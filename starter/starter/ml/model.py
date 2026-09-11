import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def compute_slice_metrics(
    df: pd.DataFrame,
    feature: str,
    y: np.ndarray,
    preds: np.ndarray,
) -> dict[str, dict]:
    """Compute model metrics for each unique value of a categorical feature.

    For every unique value of ``feature``, the rows belonging to that value are
    selected and precision, recall, and F-beta are computed via the existing
    ``compute_model_metrics`` function, along with the number of samples in the
    slice.

    Inputs
    ------
    df : pd.DataFrame
        DataFrame aligned by row index with ``y`` and ``preds``.
    feature : str
        Name of the categorical column in ``df`` to slice on.
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    metrics : dict[str, dict]
        Mapping of each unique feature value to a dict with keys
        ``feature``, ``value``, ``count``, ``precision``, ``recall``, and
        ``fbeta``.
    """
    y = np.asarray(y)
    preds = np.asarray(preds)
    values = df[feature].to_numpy()

    results: dict[str, dict] = {}
    for value in pd.unique(df[feature]):
        mask = values == value
        precision, recall, fbeta = compute_model_metrics(y[mask], preds[mask])
        results[str(value)] = {
            "feature": feature,
            "value": value,
            "count": int(mask.sum()),
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
        }
    return results
