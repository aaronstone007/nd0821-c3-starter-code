# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- **Developed by:** Santhoshkumar Kotteeswaran.
- **Model type:** Supervised binary classifier.
- **Algorithm:** scikit-learn `RandomForestClassifier` trained with default hyperparameters (`random_state=42`).
- **Inputs:** A single U.S. Census record with continuous features (`age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`) and categorical features (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`). Categorical features are one-hot encoded with a fitted `OneHotEncoder` (`handle_unknown="ignore"`); the label is binarized with a `LabelBinarizer`.
- **Output:** A predicted income class, either `<=50K` or `>50K`.
- **Artifacts:** `model.pkl` (classifier), `encoder.pkl` (one-hot encoder), and `lb.pkl` (label binarizer), persisted under `starter/model/`.
- **Serving:** Exposed through a FastAPI service (`GET /` welcome, `POST /predict` inference) that loads the artifacts once at startup.

## Intended Use

- **Primary use:** Predict whether an individual's annual income exceeds $50K based on demographic and employment attributes from the Census dataset.
- **Intended users:** Learners and engineers exploring an end-to-end ML pipeline, and API consumers integrating the classifier into demonstration applications.
- **Out of scope:** This model is a teaching artifact. It must not be used for real-world decisions affecting individuals such as credit, hiring, lending, insurance, housing, or any determination of economic opportunity or eligibility.

## Training Data

- **Source:** The publicly available U.S. Census Income dataset (the "Adult" dataset), provided as `starter/data/census.csv`.
- **Cleaning:** Leading and trailing whitespace is stripped from column headers and all string cells; the cleaned data is written to `starter/data/census_clean.csv` without modifying the raw file.
- **Split:** The cleaned data is split into training and test sets with `train_test_split(test_size=0.20, random_state=42)`; the training set is 80% of the rows.
- **Processing:** The training split is processed with `process_data(..., training=True)`, which fits the `OneHotEncoder` on the eight categorical features and the `LabelBinarizer` on the `salary` label. The label `<=50K` maps to 0 and `>50K` maps to 1.

## Evaluation Data

- **Source:** The held-out 20% test split from the same cleaned Census dataset (6,513 records).
- **Processing:** The test split is processed with `process_data(..., training=False)`, reusing the encoder and label binarizer fitted on the training data (no refitting), so the evaluation mirrors production inference.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

Performance is measured with precision, recall, and F-beta (beta = 1, i.e. the F1 score), computed via `compute_model_metrics` with `zero_division=1`.

On the held-out test set the model achieves:

| Metric    | Value  |
|-----------|--------|
| Precision | 0.7391 |
| Recall    | 0.6384 |
| F-beta    | 0.6851 |

These values were produced by running `python starter/starter/train_model.py` (which uses `random_state=42` for reproducibility), whose output reports the overall test-set metrics.

Per-slice performance across every value of each categorical feature is written to `starter/slice_output.txt`. Those slices reveal notable variation. For example, on the `sex` feature the model reaches an F-beta of 0.6985 for `Male` versus 0.5995 for `Female`, driven largely by lower recall for the `Female` slice (0.5107 vs 0.6607). Recall is also weak for lower-education slices (e.g. `HS-grad` recall 0.4232) and small-count slices show unstable metrics.

## Ethical Considerations

- **Protected attributes:** The features include `race`, `sex`, and `native-country`. Training a model directly on these attributes risks learning and reproducing historical societal biases present in the data.
- **Disparate performance:** The slice metrics show uneven performance across subgroups (for example, lower recall for the `Female` slice than the `Male` slice), meaning the model's errors are not evenly distributed across groups.
- **Representativeness:** The Census "Adult" dataset reflects a specific population and time period and is not representative of all populations; predictions on other populations may be unreliable.
- **Potential harm:** Using this model for consequential decisions could unfairly disadvantage individuals in underperforming or underrepresented groups. It is intended for educational demonstration only.

## Caveats and Recommendations

- **Not for production decisions:** This is a course/demonstration model and should not inform real decisions about individuals.
- **Small slices are noisy:** Several categorical values have very few samples (some fewer than ten), so their precision/recall/fbeta figures are statistically unreliable and can appear as 0.0 or 1.0.
- **Default hyperparameters:** The `RandomForestClassifier` uses scikit-learn defaults; hyperparameter tuning, class balancing, or threshold adjustment could improve recall and reduce subgroup disparities.
- **Fairness auditing:** Before any broader use, conduct a fairness assessment, monitor slice-level metrics over time, and consider mitigation techniques for the observed performance gaps.
- **Reproducibility:** A fixed `random_state=42` is used for the split and the classifier so results are reproducible; changing it will shift the reported metrics.
