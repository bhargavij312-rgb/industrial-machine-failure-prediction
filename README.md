# Industrial Machine Failure Prediction

**Learn Depth Academy Track 1 Capstone — Problem 21**

## Problem

Predict whether an industrial machine is likely to experience a failure within a defined operating window using machine operating and maintenance data.

The capstone focuses on:

* **Domain:** Industrial Systems
* **ML Task:** Binary Classification
* **Expected Inputs:** Temperature, vibration, pressure, runtime, maintenance gap
* **Models:** Logistic Regression, KNN, Decision Tree
* **Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix and ROC-AUC where appropriate
* **Application:** Streamlit-based prediction prototype

## Objective

Unexpected machine failures can result in production downtime, maintenance costs and operational losses.

This project explores how machine-learning classification models can be used to identify patterns associated with machine failures and provide an early prediction of failure risk.

## Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset** from the **UCI Machine Learning Repository**.

**Dataset:**
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

The dataset contains **10,000 observations** with machine-related variables including:

* Air temperature
* Process temperature
* Rotational speed
* Torque
* Tool wear
* Machine failure information

The dataset is synthetic, but it is designed to represent predictive-maintenance scenarios found in industrial environments.

### Dataset Handling

The dataset is **not included directly in this repository**. It is retrieved through the `ucimlrepo` package during training.

This keeps the repository lightweight while maintaining the original dataset source and attribution.

## Machine Learning Workflow

The project follows the workflow below:

```text
Dataset
   ↓
Data Loading
   ↓
Data Cleaning & Preprocessing
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Streamlit Prediction App
```

## Models

The project focuses on foundational machine-learning classification models:

### Logistic Regression

Used as a baseline binary classification model for predicting whether machine failure occurs.

### K-Nearest Neighbors (KNN)

Classifies observations based on the similarity between their feature values and neighboring training samples.

### Decision Tree

Uses decision rules based on feature values to classify machine observations into failure or non-failure categories.

## Evaluation Metrics

Model performance is evaluated using multiple metrics:

* **Accuracy** — Overall proportion of correct predictions
* **Precision** — Proportion of predicted failures that were actually failures
* **Recall** — Proportion of actual failures correctly identified
* **F1-score** — Balance between precision and recall
* **Confusion Matrix** — Breakdown of correct and incorrect predictions
* **ROC-AUC** — Measures the model's ability to distinguish between the two classes where applicable

Accuracy is not considered alone because machine-failure datasets may contain fewer failure cases than normal-operation cases.

## Project Structure

```text
Industrial-Machine-Failure-Prediction/
│
├── train.py
├── app.py
├── capstone_notebook.ipynb
├── requirements.txt
├── README.md
├── PROJECT_REPORT.pdf
│
├── data/
│   └── Created automatically after training
│
└── artifacts/
    └── Trained model and evaluation outputs
```

### File Description

| File                      | Description                                                                   |
| ------------------------- | ----------------------------------------------------------------------------- |
| `train.py`                | Loads data, preprocesses features, trains the model and evaluates performance |
| `app.py`                  | Streamlit application for machine-failure prediction                          |
| `capstone_notebook.ipynb` | Notebook containing the ML workflow and analysis                              |
| `requirements.txt`        | Required Python packages                                                      |
| `README.md`               | Project documentation                                                         |
| `PROJECT_REPORT.pdf`      | Technical report and research document                                        |
| `data/`                   | Dataset files generated during execution                                      |
| `artifacts/`              | Saved model and evaluation outputs                                            |

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run the training script:

```bash
python train.py
```

The script will:

1. Retrieve the dataset.
2. Prepare the data.
3. Perform preprocessing.
4. Train the classification model.
5. Evaluate the model.
6. Save the trained model and evaluation outputs.

## Running the Streamlit App

After training the model, run:

```bash
streamlit run app.py
```

The Streamlit application provides a simple interface for entering machine parameters and generating a predicted machine-failure outcome.

## Viva Points

**1. Why predictive maintenance?**
Predictive maintenance aims to identify potential machine failures before they occur, helping reduce unexpected downtime and maintenance costs.

**2. Why classification?**
The target represents whether a machine failure occurs, making this a binary classification problem.

**3. Why not use only accuracy?**
Failure cases may be less frequent than normal cases. Precision, recall and F1-score provide additional information about failure detection performance.

**4. Why preprocessing?**
Different features may have different scales, and categorical variables may require encoding. Preprocessing prepares the data for effective model training.

**5. Why avoid data leakage?**
Using information that would only be available after a failure can make model performance appear better than it actually is.

**6. Why Streamlit?**
Streamlit provides a simple way to convert a machine-learning model into an interactive prototype without building a separate frontend.

## Future Improvements

Possible improvements include:

* Hyperparameter tuning
* Cross-validation
* Handling class imbalance
* Feature importance analysis
* Model comparison
* Model explainability
* Improved Streamlit interface
* Cloud deployment

## Academic Integrity

This project is developed as part of the **Learn Depth Academy Track 1 Capstone**.

AI tools may be used for learning and development support, but the final implementation, results and methodology should be reproducible and explainable by the student.

External datasets, libraries, research papers and other resources should be appropriately credited.

## Project Status

**Status:** In Development

Built as part of the Learn Depth Academy Track 1 Capstone.
