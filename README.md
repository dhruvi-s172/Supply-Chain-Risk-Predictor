# 🚚 Supply Chain Risk Predictor

A production-ready **Streamlit Machine Learning application** that predicts the **risk classification** of supply-chain events using logistics and operational indicators. The model analyzes shipment, warehouse, transportation, environmental, and supplier-related information to classify shipments into **High Risk**, **Moderate Risk**, or **Low Risk**.

---

## 🌐 Live Demo

**Streamlit App**

https://supply-chain-risk-predictor-d9fhpjnrslfjhnbtlqyf5f.streamlit.app/

---

## 📂 GitHub Repository

https://github.com/dhruvi-s172/Supply-Chain-Risk-Predictor

---

## 📖 Project Overview

Efficient supply-chain management requires identifying risky shipments before delays and disruptions occur. This project uses a **Random Forest Classifier** trained on historical logistics data to estimate shipment risk based on operational conditions.

The application provides an intuitive Streamlit interface where users can enter shipment details and instantly receive:

- Predicted Risk Classification
- Prediction Confidence
- Probability Distribution across all risk classes
- Dataset Overview
- Model Information

---

# ✨ Features

- Interactive Streamlit UI
- Real-time shipment risk prediction
- Random Forest Machine Learning model
- Prediction confidence visualization
- Probability chart using Plotly
- Dataset overview page
- Model information page
- Input validation
- Reproducible training workflow
- Clean and responsive interface

---

# 🗂 Dataset

**Source**

Kaggle Logistics & Supply Chain Dataset

https://www.kaggle.com/datasets/datasetengineer/logistics-and-supply-chain-dataset

### Dataset Information

- **Records:** 32,065
- **Original Features:** 26
- **Target Column:** `risk_classification`

The dataset includes operational logistics information such as:

- Vehicle GPS Location
- ETA Variation
- Traffic Congestion
- Fuel Consumption
- Warehouse Inventory
- Weather Severity
- Supplier Reliability
- Lead Time
- Shipping Costs
- Historical Demand
- Route Risk
- Driver Behavior
- Customs Clearance Time
- Cargo Condition
- Delivery Time Deviation
- and several additional logistics features.

---

# 🧹 Data Preprocessing

The preprocessing pipeline follows the notebook used for model training.

Steps include:

- Load dataset
- Remove duplicate rows
- Separate target variable
- Remove leakage-prone columns
  - `delay_probability`
  - `disruption_likelihood_score`
- Convert timestamp into Unix seconds
- Encode target labels using LabelEncoder
- Perform stratified train-test split
- Train Random Forest classifier
- Save trained model using Joblib

Target encoding order:

```
High Risk
Low Risk
Moderate Risk
```

---

# 🤖 Machine Learning Models Explored

The following algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Linear Support Vector Machine (LinearSVC)

The deployed model is:

```
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

---

# 📁 Project Structure

```
Supply-Chain-Risk-Predictor/
│
├── app.py
├── train_model.py
├── notebook.ipynb
├── dataset.csv
├── model.pkl
├── model_metadata.json
├── requirements.txt
├── README.md
├── assets/
└── images/
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/dhruvi-s172/Supply-Chain-Risk-Predictor.git

cd Supply-Chain-Risk-Predictor
```

Create virtual environment

```bash
python -m venv .venv
```

Activate environment

### Windows

```bash
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Project

Launch the Streamlit application

```bash
streamlit run app.py
```

If the model artifacts are missing, regenerate them:

```bash
python train_model.py
```

---

# 🖥 Application Pages

### Risk Prediction

- Enter shipment details
- Predict shipment risk
- View confidence scores
- Probability distribution

### Dataset Overview

Displays dataset statistics and feature information.

### Model Information

Displays:

- Model type
- Features used
- Target classes
- Training metadata

---
---

## 📷 Screenshots

### Home Page

![Home Page](images/home.png)

---

### Prediction Result

![Prediction Result](images/prediction.png)

---
# 📊 Prediction Output

The application predicts one of three classes:

- 🔴 High Risk
- 🟡 Moderate Risk
- 🟢 Low Risk

Along with prediction probabilities visualized using Plotly.

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Plotly
- Jupyter Notebook

---

# 🚀 Future Improvements

- Hyperparameter tuning
- Cross-validation reporting
- SHAP feature importance
- Model monitoring
- Batch prediction using CSV upload
- REST API deployment
- Docker support
- CI/CD pipeline
- Automated unit testing

---

# 👩‍💻 Author

**Dhruvi Ghodasara**

GitHub:

https://github.com/dhruvi-s172

---

# 🙏 Acknowledgements

- Kaggle Logistics & Supply Chain Dataset
- Scikit-learn
- Streamlit
- Plotly

---

## ⭐ If you found this project useful, consider giving it a star on GitHub!
