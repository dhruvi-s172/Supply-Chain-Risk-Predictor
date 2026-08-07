# Supply Chain Risk Prediction

A production-ready Streamlit application that predicts the **risk classification** of a supply-chain event from logistics and operational signals.

## Problem statement

Supply-chain teams need an early, consistent view of delivery and operating risk. This project classifies an event as **High Risk**, **Moderate Risk**, or **Low Risk** using historical logistics data.

## Dataset description

`dataset.csv` contains 32,065 records and 26 original fields, including vehicle location, fuel use, congestion, inventory, timing, supplier reliability, environmental conditions, and the target `risk_classification`.

The notebook's final preprocessing removes `disruption_likelihood_score` and `delay_probability` from the model inputs because they are high-risk leakage candidates. It converts `timestamp` to Unix seconds. The target is encoded with `LabelEncoder` in alphabetical class order: `High Risk`, `Low Risk`, `Moderate Risk`.

## Project workflow

1. Load and inspect the supplied logistics dataset.
2. Remove duplicate rows.
3. Separate the target and leakage-prone fields.
4. Convert `timestamp` to Unix seconds.
5. Label-encode the target and make a stratified train/test split.
6. Train candidate classifiers and select the deployment Random Forest workflow.
7. Serialize `model.pkl` and metadata, then serve predictions in Streamlit.

## Technologies used

- Python, pandas, scikit-learn, joblib
- Streamlit and Plotly
- Jupyter Notebook

## Machine-learning models explored

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Linear Support Vector Machine (the notebook uses `LinearSVC`)

`model.pkl` is a reproducibly trained `RandomForestClassifier(n_estimators=100, random_state=42)`. Its exact hold-out score is saved in `model_metadata.json` when the artifact is built.

## Installation

```bash
git clone <your-repository-url>
cd Supply-Chain-Risk-Prediction
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Local execution

The original saved artifacts were not present with the supplied files. Build the compatible deployment artifact once from the included dataset:

```bash
python train_model.py
streamlit run app.py
```

## Streamlit Community Cloud deployment

1. Push this entire folder to a GitHub repository.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**.
3. Select the repository, branch, and `app.py` as the main file.
4. Deploy. Community Cloud installs `requirements.txt` automatically.

No secrets or environment variables are required.

## Repository structure

```text
Supply-Chain-Risk-Prediction/
├── app.py                  # Streamlit user interface
├── model.pkl               # Generated Random Forest classifier (after training)
├── model_metadata.json     # Generated feature order, encodings, statistics, model details
├── train_model.py          # Reproducible model-build script
├── requirements.txt
├── README.md
├── .gitignore
├── dataset.csv
├── notebook.ipynb
├── assets/
└── images/
```

## Screenshots

Add application screenshots to `images/` after launching the app, then reference them here:

```markdown
![Risk prediction page](images/risk-prediction.png)
```

## Future improvements

- Validate the risk label's business definition and remove any remaining data leakage.
- Add automated data validation, model monitoring, and unit tests.
- Compare calibrated models and report macro-F1 alongside accuracy for the imbalanced classes.
- Connect the app to live fleet, warehouse, and weather feeds.

## Artifact note

The supplied notebook did **not** contain the claimed `model.pkl` or `scaler.pkl`. No scaler is needed because the notebook does not use `StandardScaler`. `train_model.py` creates the compatible `model.pkl` and `model_metadata.json` using the notebook's documented final preprocessing and Random Forest workflow. If you have an original saved model, replace `model.pkl` only together with compatible preprocessing metadata (feature order, timestamp conversion, target mapping, and any scaler/encoders), or rerun the model-generation workflow so all artifacts remain aligned.
