# Industrial Machine Failure Prediction

Learn Depth Academy Track 1 Capstone — Problem 21

## Problem
Predict whether an industrial machine is likely to experience a failure within a defined operating window.

The capstone brief specifies:
- Domain: Industrial Systems
- ML task: Binary Classification
- Expected inputs: temperature, vibration, pressure, runtime, maintenance gap
- Foundational models such as Logistic Regression, KNN or Decision Tree
- Evaluation using Accuracy, Precision, Recall, F1-score, Confusion Matrix and ROC-AUC where appropriate
- A simple Streamlit prototype
- Technical paper, notebook, README, requirements, trained model and presentation

## Dataset
This project is designed for the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

UCI dataset page:
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

The dataset contains 10,000 observations and includes temperature, rotational speed, torque, tool wear and machine-failure information. It is synthetic but intended to reflect predictive-maintenance data encountered in industry.

### Important
The dataset is NOT bundled into this ZIP because the capstone requires students to credit external datasets and the UCI dataset has its own license/source. Run:

    pip install -r requirements.txt
    python train.py

The training script downloads the UCI dataset through `ucimlrepo`, prepares the data, trains a model, evaluates it and saves the model.

## Project structure

- `train.py` — data loading, preprocessing, training, evaluation and model saving
- `app.py` — Streamlit prediction application
- `capstone_notebook.ipynb` — notebook version of the workflow
- `requirements.txt` — Python dependencies
- `README.md` — setup and usage
- `PROJECT_REPORT.pdf` — preliminary technical report/research document
- `data/` — created automatically after running training
- `artifacts/` — trained model and evaluation outputs

## Run

    python -m venv venv
    # Windows:
    venv\Scripts\activate
    pip install -r requirements.txt
    python train.py
    streamlit run app.py

## Viva points
1. Why predictive maintenance? It aims to identify failure risk before failure occurs.
2. Why classification? The target is whether machine failure occurs.
3. Why not use only accuracy? Failure cases can be rare, so precision, recall and F1 are important.
4. Why preprocessing? Categorical variables need encoding and numerical features may require scaling depending on the selected model.
5. Why avoid leakage? Features that reveal the target after the event can make evaluation unrealistically optimistic.
6. Why Streamlit? The capstone explicitly asks for a lightweight application.

## Academic integrity
The capstone document states that AI tools may be used for learning/support, but submitted results must be reproducible and explainable by the student. External datasets, papers, libraries and references must be credited.
