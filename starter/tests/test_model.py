"""Unit tests for the model layer functions in ``ml/model.py``.

Covers ``train_model`` (returns a fitted classifier), ``inference`` (returns
predictions of the correct length and values), and ``compute_model_metrics``
(returns three floats and is correct on a known input).

Validates: Requirements 5.1, 5.5
"""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

# Path to the cleaned census data, resolved relative to this test file.
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "census_clean.csv")

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def data_sample():
    """Load a small subset of ``census_clean.csv`` for fast model tests.

    A stratified-ish sample is taken by combining the head and tail of the
    dataset so both salary classes are represented, keeping the fixture quick
    while still exercising the real processing pipeline.
    """
    df = pd.read_csv(DATA_PATH)
    # Take rows from both ends so both label classes are present.
    subset = pd.concat([df.head(100), df.tail(100)]).reset_index(drop=True)
    return subset


@pytest.fixture(scope="module")
def processed_data(data_sample):
    """Process the sample into (X, y) arrays via the shared ``process_data``."""
    X, y, _encoder, _lb = process_data(
        data_sample,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    return X, y


def test_train_model_returns_fitted_classifier(processed_data):
    """``train_model`` returns a fitted ``RandomForestClassifier``."""
    X, y = processed_data
    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)
    # A fitted estimator exposes predict and learned attributes.
    assert hasattr(model, "predict")
    assert hasattr(model, "classes_")


def test_inference_length_and_values(processed_data):
    """``inference`` returns predictions matching row count with values in {0, 1}."""
    X, y = processed_data
    model = train_model(X, y)

    preds = inference(model, X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_returns_three_floats(processed_data):
    """``compute_model_metrics`` returns three floats for real predictions."""
    X, y = processed_data
    model = train_model(X, y)
    preds = inference(model, X)

    precision, recall, fbeta = compute_model_metrics(y, preds)

    for metric in (precision, recall, fbeta):
        assert isinstance(metric, float)
        assert 0.0 <= metric <= 1.0


def test_compute_model_metrics_perfect_predictions():
    """Perfect predictions yield precision, recall, and F-beta of 1.0."""
    y = np.array([0, 1, 1, 0, 1])
    preds = np.array([0, 1, 1, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    assert fbeta == pytest.approx(1.0)
