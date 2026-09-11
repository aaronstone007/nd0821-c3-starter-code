# Script to train machine learning model.

import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    train_model,
)

# Resolve paths relative to this module so the script runs from any CWD.
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "census_clean.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
SLICE_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "slice_output.txt")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def main() -> None:
    """Train the model, persist artifacts, and report overall metrics."""
    # Load in the cleaned data.
    data = pd.read_csv(DATA_PATH)

    # Optional enhancement: use K-fold cross validation instead of a split.
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    # Process the training data (fits the encoder and label binarizer).
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Process the test data, reusing the fitted encoder and label binarizer.
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Train the model.
    model = train_model(X_train, y_train)

    # Persist the model and preprocessing artifacts.
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)
    with open(os.path.join(MODEL_DIR, "lb.pkl"), "wb") as f:
        pickle.dump(lb, f)

    # Evaluate overall performance on the test set.
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print("Overall test set metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F-beta:    {fbeta:.4f}")

    # Generate the slice performance report on the test set.
    write_slice_output(test, y_test, preds)
    print(f"Slice metrics written to {SLICE_OUTPUT_PATH}")


def write_slice_output(test: pd.DataFrame, y_test, preds) -> None:
    """Compute per-slice metrics for every categorical feature and write them.

    ``test`` retains the same row order as ``y_test`` and ``preds`` because
    ``process_data`` does not reorder rows, so positional masks stay aligned.
    Each line identifies the feature name, the value, and the sample count
    along with precision, recall, and F-beta for that slice.
    """
    with open(SLICE_OUTPUT_PATH, "w") as f:
        for feature in cat_features:
            f.write(f"Feature: {feature}\n")
            slice_metrics = compute_slice_metrics(test, feature, y_test, preds)
            for metrics in slice_metrics.values():
                f.write(
                    f"  {feature} = {metrics['value']} "
                    f"| count: {metrics['count']} "
                    f"| precision: {metrics['precision']:.4f} "
                    f"| recall: {metrics['recall']:.4f} "
                    f"| fbeta: {metrics['fbeta']:.4f}\n"
                )
            f.write("\n")


if __name__ == "__main__":
    main()
