"""API tests for the FastAPI census income service in ``main.py``.

Uses FastAPI's ``TestClient`` to exercise the live app: one ``GET /`` test that
asserts the status code and welcome body, and two ``POST /predict`` tests that
each assert the status code and JSON body for a distinct prediction outcome
(one low-income profile expected to yield ``<=50K`` and one high-income profile
expected to yield ``>50K``).

Validates: Requirements 5.2, 5.3, 5.4, 5.5
"""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# A low-income profile: young, never-married, low education/hours, no capital
# gains. Reliably classified as ``<=50K`` by the trained model.
LOW_INCOME_RECORD = {
    "age": 25,
    "workclass": "Private",
    "fnlgt": 226802,
    "education": "11th",
    "education-num": 7,
    "marital-status": "Never-married",
    "occupation": "Handlers-cleaners",
    "relationship": "Own-child",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 20,
    "native-country": "United-States",
}

# A high-income profile: advanced degree, professional occupation, substantial
# capital gains, and long hours. Reliably classified as ``>50K``.
HIGH_INCOME_RECORD = {
    "age": 42,
    "workclass": "Private",
    "fnlgt": 159449,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 14084,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States",
}


def test_get_root_returns_welcome_message():
    """``GET /`` responds with HTTP 200 and a JSON welcome message."""
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert "message" in body
    assert "Welcome" in body["message"]


def test_post_predict_low_income():
    """``POST /predict`` on a low-income profile returns HTTP 200 and ``<=50K``."""
    response = client.post("/predict", json=LOW_INCOME_RECORD)

    assert response.status_code == 200
    body = response.json()
    assert body == {"prediction": "<=50K"}


def test_post_predict_high_income():
    """``POST /predict`` on a high-income profile returns HTTP 200 and ``>50K``."""
    response = client.post("/predict", json=HIGH_INCOME_RECORD)

    assert response.status_code == 200
    body = response.json()
    assert body == {"prediction": ">50K"}
