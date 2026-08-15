"""
Machine Learning Models for Parkinson's Disease Detection
Subject-Aware Cross-Validation and Hyperparameter Tuning

Research Project:
Explainable AI for Early Parkinson's Disease Detection
using Machine Learning

Author: Gul Jan
Date: August 2026
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score
)

import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# Create models directory
os.makedirs("models", exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_processed_data():

    print("📂 Loading processed data...")

    X_train = pd.read_csv(
        "data/processed/X_train.csv"
    )

    X_test = pd.read_csv(
        "data/processed/X_test.csv"
    )

    y_train = pd.read_csv(
        "data/processed/y_train.csv"
    ).values.ravel()

    y_test = pd.read_csv(
        "data/processed/y_test.csv"
    ).values.ravel()

    subject_train = pd.read_csv(
        "data/processed/subject_train.csv"
    ).values.ravel()

    subject_test = pd.read_csv(
        "data/processed/subject_test.csv"
    ).values.ravel()

    print(
        f"   ✅ Training: {X_train.shape[0]} samples "
        f"from {len(np.unique(subject_train))} subjects"
    )

    print(
        f"   ✅ Test: {X_test.shape[0]} samples "
        f"from {len(np.unique(subject_test))} subjects"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        subject_train,
        subject_test
    )


# ============================================================
# MODELS
# ============================================================

def get_models():

    return {

        "Logistic Regression":
            LogisticRegression(
                random_state=42,
                max_iter=2000
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                random_state=42
            ),

        "SVM":
            SVC(
                random_state=42,
                probability=True
            ),

        "KNN":
            KNeighborsClassifier(),

        "Naive Bayes":
            GaussianNB(),

        "XGBoost":
            XGBClassifier(
                random_state=42,
                eval_metric="logloss"
            ),

        "LightGBM":
            LGBMClassifier(
                random_state=42,
                verbose=-1
            )
    }


# ============================================================
# HYPERPARAMETER GRIDS
# ============================================================

def get_param_grids():

    return {

        "Logistic Regression": {

            "classifier__C":
                [0.01, 0.1, 1, 10, 100],

            "classifier__solver":
                ["liblinear", "lbfgs"]
        },

        "Decision Tree": {

            "classifier__max_depth":
                [3, 5, 7, 10, None],

            "classifier__min_samples_split":
                [2, 5, 10],

            "classifier__min_samples_leaf":
                [1, 2, 4]
        },

        "Random Forest": {

            "classifier__n_estimators":
                [50, 100, 200],

            "classifier__max_depth":
                [3, 5, 7, 10, None],

            "classifier__min_samples_split":
                [2, 5],

            "classifier__min_samples_leaf":
                [1, 2, 4]
        },

        "SVM": {

            "classifier__C":
                [0.01, 0.1, 1, 10, 100],

            "classifier__kernel":
                ["linear", "rbf"],

            "classifier__gamma":
                ["scale", "auto"]
        },

        "KNN": {

            "classifier__n_neighbors":
                [3, 5, 7, 9, 11],

            "classifier__weights":
                ["uniform", "distance"]
        },

        "Naive Bayes": {},

        "XGBoost": {

            "classifier__n_estimators":
                [50, 100, 200],

            "classifier__max_depth":
                [3, 5, 7],

            "classifier__learning_rate":
                [0.01, 0.1, 0.3],

            "classifier__subsample":
                [0.8, 1.0]
        },

        "LightGBM": {

            "classifier__n_estimators":
                [50, 100, 200],

            "classifier__max_depth":
                [3, 5, 7, -1],

            "classifier__learning_rate":
                [0.01, 0.1, 0.3]
        }
    }


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
    subject_train,
    model_name,
    model
):

    print(f"\n🔹 {model_name}...")

    # Scaling is INSIDE the pipeline.
    # This prevents scaling leakage during CV.
    pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            model
        )
    ])

    param_grids = get_param_grids()

    param_grid = param_grids.get(
        model_name,
        {}
    )

    # --------------------------------------------------------
    # Naive Bayes: no hyperparameter search
    # --------------------------------------------------------

    if not param_grid:

        print("   ℹ️ No hyperparameter grid.")
        print("   Training default configuration...")

        pipeline.fit(
            X_train,
            y_train
        )

        return (
            pipeline,
            None,
            None
        )

    # --------------------------------------------------------
    # GroupKFold
    # --------------------------------------------------------

    gkf = GroupKFold(
        n_splits=5
    )

    grid_search = GridSearchCV(

        estimator=pipeline,

        param_grid=param_grid,

        cv=gkf,

        # Model selection happens ONLY using
        # training data / CV folds.
        scoring="f1_macro",

        n_jobs=-1,

        return_train_score=False,

        verbose=0
    )

    grid_search.fit(
        X_train,
        y_train,
        groups=subject_train
    )

    print(
        f"   ✅ Best params: "
        f"{grid_search.best_params_}"
    )

    print(
        f"   📊 Mean CV F1-macro: "
        f"{grid_search.best_score_:.4f}"
    )

    return (
        grid_search.best_estimator_,
        grid_search.best_score_,
        grid_search.best_params_
    )


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def train_all_models():

    print("=" * 60)
    print(
        "🤖 TRAINING ML MODELS "
        "(SUBJECT-AWARE)"
    )
    print("=" * 60)

    print()
    print(
        "⚠️ IMPORTANT:"
    )
    print(
        "   Test set is NOT used for model selection."
    )
    print(
        "   GroupKFold is used only on training subjects."
    )
    print(
        "   Scaling occurs inside each pipeline."
    )
    print()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        subject_train,
        subject_test
    ) = load_processed_data()

    models = get_models()

    trained_models = {}

    results = []

    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    print(
        "📊 Performing subject-aware "
        "cross-validation...\n"
    )

    for name, model in models.items():

        try:

            (
                trained_model,
                cv_f1,
                best_params
            ) = train_model(

                X_train,
                y_train,
                subject_train,
                name,
                model
            )

            # If no CV score exists (Naive Bayes),
            # train model and calculate CV performance
            # separately using GroupKFold.

            if cv_f1 is None:

                gkf = GroupKFold(
                    n_splits=5
                )

                cv_scores = []

                for train_idx, val_idx in gkf.split(
                    X_train,
                    y_train,
                    groups=subject_train
                ):

                    temp_model = Pipeline([
                        (
                            "scaler",
                            StandardScaler()
                        ),
                        (
                            "classifier",
                            GaussianNB()
                        )
                    ])

                    temp_model.fit(
                        X_train.iloc[train_idx],
                        y_train[train_idx]
                    )

                    pred = temp_model.predict(
                        X_train.iloc[val_idx]
                    )

                    score = f1_score(
                        y_train[val_idx],
                        pred,
                        average="macro"
                    )

                    cv_scores.append(score)

                cv_f1 = np.mean(cv_scores)

                print(
                    f"   📊 Mean CV F1-macro: "
                    f"{cv_f1:.4f}"
                )

            trained_models[name] = trained_model

            results.append({

                "Model": name,

                "CV_F1_Macro":
                    cv_f1,

                "Best_Params":
                    str(best_params)
            })

        except Exception as e:

            print(
                f"   ❌ Error in {name}: {e}"
            )


    # ========================================================
    # SELECT BEST MODEL USING CV ONLY
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "CV_F1_Macro",
        ascending=False
    )

    best_model_name = (
        results_df.iloc[0]["Model"]
    )

    best_model = (
        trained_models[best_model_name]
    )

    print("\n" + "=" * 60)
    print(
        "🏆 MODEL SELECTION "
        "(TRAINING CV ONLY)"
    )
    print("=" * 60)

    print(
        results_df[
            [
                "Model",
                "CV_F1_Macro"
            ]
        ].to_string(index=False)
    )

    print()
    print(
        f"🏆 SELECTED MODEL: "
        f"{best_model_name}"
    )

    print(
        f"   Mean CV F1-macro: "
        f"{results_df.iloc[0]['CV_F1_Macro']:.4f}"
    )

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    print("\n" + "=" * 60)
    print(
        "🧪 FINAL TEST SET EVALUATION"
    )
    print("=" * 60)

    test_results = []

    for name, model in trained_models.items():

        y_pred = model.predict(
            X_test
        )

        if hasattr(
            model,
            "predict_proba"
        ):

            y_proba = model.predict_proba(
                X_test
            )[:, 1]

        else:

            y_proba = None

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1_macro = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        balanced_acc = balanced_accuracy_score(
            y_test,
            y_pred
        )

        if y_proba is not None:

            roc_auc = roc_auc_score(
                y_test,
                y_proba
            )

        else:

            roc_auc = np.nan

        test_results.append({

            "Model": name,

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1-Score":
                f1,

            "F1-Macro":
                f1_macro,

            "Balanced-Accuracy":
                balanced_acc,

            "ROC-AUC":
                roc_auc
        })


    test_results_df = pd.DataFrame(
        test_results
    )

    print(
        test_results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    joblib.dump(
        best_model,
        "models/best_model.pkl"
    )

    print()
    print(
        f"💾 Selected model saved:"
    )

    print(
        "   models/best_model.pkl"
    )


    # Save CV results
    results_df.to_csv(
        "models/cv_model_results.csv",
        index=False
    )

    # Save final test results
    test_results_df.to_csv(
        "models/model_results.csv",
        index=False
    )

    print(
        "💾 CV results saved:"
    )

    print(
        "   models/cv_model_results.csv"
    )

    print(
        "💾 Test results saved:"
    )

    print(
        "   models/model_results.csv"
    )

    print("\n" + "=" * 60)
    print(
        "✅ MODEL TRAINING COMPLETE!"
    )
    print("=" * 60)

    return (
        best_model,
        results_df,
        test_results_df
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    train_all_models()