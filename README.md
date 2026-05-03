# EndoAware

**EndoAware** is a Flask-based web application that estimates the probability of endometriosis using multiple pre-trained machine learning models.

It provides an interactive questionnaire, model comparison capabilities, and explainable AI insights to demonstrate how different classifiers interpret symptom data.

As the dataset used contains only binary-encoded features no scaling or further encoding was used. It is also important to note that initial EDA was not included in this repository.

---

## Overview

EndoAware simulates a lightweight clinical decision-support tool by:

- Collecting user-reported symptoms via a structured questionnaire  
- Applying machine learning models to estimate risk probability  
- Providing interpretable outputs using SHAP feature importance  
- Generating a downloadable report for further review  

This application is intended for **demonstration purposes only** and is **not a medical diagnostic tool**.

---

## Features

- **Symptom Questionnaire**  
  30 binary symptom inputs grouped by category  

- **Multiple ML Models**  
  Compare predictions across:
  - Support Vector Machine (SVM)  
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
  - Multi-Layer Perceptron (MLP)  

- **Probability-Based Output**  
  Returns a percentage likelihood rather than a binary prediction  

- **Explainable AI (SHAP)**  
  Visualises feature contributions for supported models  

- **PDF Report Generation**  
  Exports:
  - Prediction result  
  - User responses  
  - Key contributing features  

---

### Key Components

- **`app.py`**  
  Main Flask application handling routing, inference, SHAP explanations, and PDF generation  

- **`train_models.py`**  
  Script used to train and export machine learning models  

- **`model_training/`**  
  Serialized trained models  

- **`data/`**  
  Preprocessed dataset used for training and SHAP background sampling  

---

## Getting Started

### 1. Install Dependencies
```bash
pip install flask pandas numpy joblib openpyxl pypdf shap scikit-learn xgboost
```

### 2. Run the Application
```bash
python3 app.py
```

### 3. Open in Browser
```bash
http://127.0.0.1:5000
```