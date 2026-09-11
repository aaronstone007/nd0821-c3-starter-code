"""Query the live Census Income Classification API.

POSTs a single census record to the deployed ``/predict`` endpoint and prints
the HTTP status code and the model's prediction.

The base URL defaults to the deployed Render service but can be overridden with
the ``LIVE_API_URL`` environment variable or a command-line argument:

    python query_live_api.py
    python query_live_api.py https://my-service.onrender.com
    LIVE_API_URL=https://my-service.onrender.com python query_live_api.py
"""

import os
import sys

import requests

DEFAULT_URL = "https://census-income-ml-api.onrender.com"

# Sample record using the API's original hyphenated JSON keys.
SAMPLE_RECORD = {
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


def main() -> int:
    base_url = (
        sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LIVE_API_URL", DEFAULT_URL)
    ).rstrip("/")
    predict_url = f"{base_url}/predict"

    print(f"POST {predict_url}")
    print(f"Payload: {SAMPLE_RECORD}")
    print("(a free Render instance may cold-start; allowing up to 60s...)")

    try:
        response = requests.post(predict_url, json=SAMPLE_RECORD, timeout=60)
    except requests.exceptions.RequestException as exc:
        print(f"Request failed: {exc}")
        return 1

    print(f"Status code: {response.status_code}")
    try:
        print(f"Response JSON: {response.json()}")
    except ValueError:
        print(f"Response body: {response.text}")

    return 0 if response.status_code == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
