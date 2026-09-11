# Put the code for your API here.
import os
import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference


# directory present, pull the versioned data/artifacts before serving.
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Resolve artifact paths relative to this module so the app is portable
# regardless of the current working directory (local runs vs. Heroku).
MODULE_DIR = Path(__file__).parent
MODEL_DIR = MODULE_DIR / "model"

# Categorical features, in the same order used during training.
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


def _load_artifact(name: str):
    """Load a pickled artifact from the model directory.

    Inputs
    ------
    name : str
        File name of the artifact (e.g. ``"model.pkl"``).
    Returns
    -------
    object
        The unpickled artifact.
    """
    path = MODEL_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Required model artifact not found: {path}. "
            "Train the model or pull artifacts before starting the API."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# Load the model, encoder, and label binarizer once at module load so they are
# reused across requests instead of being read from disk on every call.
model = _load_artifact("model.pkl")
encoder = _load_artifact("encoder.pkl")
lb = _load_artifact("lb.pkl")


class CensusRecord(BaseModel):
    """A single census record used as the inference request body.

    Hyphenated CSV column names are exposed through Pydantic aliases so the
    JSON contract keeps the original names while the Python attributes remain
    valid identifiers. ``populate_by_name`` also allows population via the
    underscore attribute names.
    """

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        },
    )


app = FastAPI(
    title="Census Income Classification API",
    description="Predicts whether income exceeds $50K/yr from census data.",
    version="1.0.0",
)

# Full training column order (original hyphenated CSV names). A prediction
# request is turned into a single-row DataFrame with these columns so the
# loaded encoder sees features in exactly the layout it was fitted on.
TRAINING_COLUMNS = [
    "age",
    "workclass",
    "fnlgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]


@app.get("/")
async def welcome() -> dict[str, str]:
    """Return a welcome message for the API root.

    Returns
    -------
    dict[str, str]
        A JSON greeting served with HTTP 200.
    """
    return {
        "message": "Welcome to the Census Income Classification API. "
        "POST a record to /predict to get an income prediction."
    }


@app.post("/predict")
async def predict(record: CensusRecord) -> dict[str, str]:
    """Run model inference on a single census record.

    The validated request is converted back to the original hyphenated column
    names, assembled into a single-row DataFrame in training column order, and
    processed with the fitted encoder/label binarizer before inference.

    Inputs
    ------
    record : CensusRecord
        A single census record supplied in the request body.
    Returns
    -------
    dict[str, str]
        The predicted salary bracket, either ``"<=50K"`` or ``">50K"``.
    """
    # Convert back to the original hyphenated keys used during training.
    row = record.model_dump(by_alias=True)
    df = pd.DataFrame([row], columns=TRAINING_COLUMNS)

    X, _, _, _ = process_data(
        df,
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(model, X)
    prediction = lb.inverse_transform(preds)[0]
    return {"prediction": prediction}
