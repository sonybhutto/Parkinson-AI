# Parkinson's Disease Prediction Using Machine Learning and Explainable AI

## Overview

This project presents a machine learning-based approach for predicting Parkinson's disease from biomedical voice measurements. The project focuses on developing a complete machine learning pipeline, evaluating multiple classification models, addressing subject-level data leakage, and interpreting model predictions using Explainable AI techniques.

An interactive Streamlit application is also included to demonstrate the trained prediction model.

## Objectives

The main objectives of this project are to:

- Develop a machine learning pipeline for Parkinson's disease prediction.
- Prevent subject-level data leakage during model training and evaluation.
- Compare multiple machine learning classification algorithms.
- Evaluate model performance using classification metrics and ROC-AUC.
- Apply SHAP-based Explainable AI to identify important predictive features.
- Develop an interactive Streamlit application for model demonstration.

## Dataset

The project uses the UCI Parkinson's dataset, which contains biomedical voice measurements collected from individuals with and without Parkinson's disease.

A key characteristic of this dataset is that it contains multiple voice recordings from the same subjects. Therefore, randomly splitting individual recordings into training and testing sets can result in data leakage, where recordings from the same subject appear in both sets.

To address this issue, subject IDs were extracted from the recording names and used for subject-level data splitting.

### Dataset Statistics

- 195 recordings
- 32 unique subjects
- 22 voice-based predictive features
- 152 recordings from 25 subjects in the training set
- 43 recordings from 7 subjects in the test set

## Methodology

The project follows the following machine learning workflow:

1. Load the Parkinson's dataset.
2. Extract subject IDs from the recording names.
3. Separate input features and target labels.
4. Perform subject-level train/test splitting using `GroupShuffleSplit`.
5. Preprocess and scale the feature data where required.
6. Train multiple machine learning classification models.
7. Evaluate models using subject-aware cross-validation with `GroupKFold`.
8. Select the best-performing model.
9. Evaluate the selected model on the held-out test subjects.
10. Apply SHAP for model explainability.
11. Visualize model performance and feature importance.
12. Integrate the trained model into a Streamlit application.

## Machine Learning Models

The following machine learning algorithms were evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Naive Bayes
- XGBoost
- LightGBM

Subject-aware cross-validation was performed using `GroupKFold` to ensure that recordings from the same subject were not distributed across different validation folds.

## Results

The best-performing model during subject-aware cross-validation was K-Nearest Neighbors (KNN).

### Best Model

- Model: K-Nearest Neighbors
- Cross-validation F1-score: 0.8493

### Held-Out Test Set

The final model was evaluated on 43 recordings belonging to subjects that were not included in the training set.

- Accuracy: 0.74
- ROC-AUC: 0.628

The results demonstrate the importance of subject-level evaluation when working with biomedical datasets containing multiple observations from the same individual.

## Model Evaluation

The project includes several evaluation visualizations:

- Confusion Matrix
- ROC Curve
- Model Comparison
- Subject Distribution

These visualizations are available in the `figures/` directory.

## Explainable AI

SHAP (SHapley Additive exPlanations) was used to investigate the features contributing to the model's predictions.

The project includes:

- Global feature importance analysis
- SHAP summary plot
- SHAP waterfall plot

Several voice-related features were identified as influential during the explainability analysis, including:

- RPDE
- D2
- MDVP:APQ
- PPE
- DFA
- spread1
- spread2
- MDVP:Shimmer(dB)
- MDVP:PPQ
- HNR

Explainable AI provides additional insight into model behavior and helps reduce the black-box nature of machine learning predictions.

## Streamlit Application

🚀 **Live Demo:** [Open Parkinson's AI Web App](https://parkinson-ai-92jynkkiefrqbwcvjcms6o.streamlit.app/)
The project includes an interactive Streamlit application located at:

`app/streamlit_app.py`

The application provides a user interface for interacting with the trained Parkinson's disease prediction model.

To run the application locally, use:

`streamlit run app/streamlit_app.py`
