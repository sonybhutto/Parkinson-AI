"""
Explainable AI using SHAP for Parkinson's Disease Classification

This module explains the selected leakage-free SVM model.
The trained model is a Pipeline:

    StandardScaler -> SVC

SHAP explanations are generated on the correctly scaled feature data.

Author: Gul Jan
Date: August 2026
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs("figures", exist_ok=True)


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

def load_model_and_data():
    """
    Load the trained Pipeline and test data.

    The Pipeline contains:
        StandardScaler -> SVC

    Returns:
        model
        scaler
        classifier
        X_test_raw
        X_test_scaled
        feature_names
    """

    print("📂 Loading model and data...")

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model_path = "models/best_model.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find {model_path}. "
            "Run python src\\models.py first."
        )

    model = joblib.load(model_path)

    print(f"   ✅ Model loaded: {type(model).__name__}")

    # --------------------------------------------------------
    # Verify Pipeline
    # --------------------------------------------------------

    if not hasattr(model, "named_steps"):
        raise TypeError(
            "best_model.pkl is not a Pipeline. "
            "Expected StandardScaler -> SVC."
        )

    if "scaler" not in model.named_steps:
        raise KeyError(
            "The trained Pipeline does not contain a 'scaler' step."
        )

    if "classifier" not in model.named_steps:
        raise KeyError(
            "The trained Pipeline does not contain a 'classifier' step."
        )

    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]

    print(f"   ✅ Scaler: {type(scaler).__name__}")
    print(f"   ✅ Classifier: {type(classifier).__name__}")

    # --------------------------------------------------------
    # Verify SVM
    # --------------------------------------------------------

    if type(classifier).__name__ != "SVC":
        print(
            f"   ⚠️ Current classifier is {type(classifier).__name__}, "
            "not SVC."
        )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    test_path = "data/processed/X_test.csv"

    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Could not find {test_path}. "
            "Run python src\\preprocessing.py first."
        )

    X_test_raw = pd.read_csv(test_path)

    print(f"   ✅ Test data loaded: {X_test_raw.shape[0]} samples")
    print(f"   ✅ Number of features: {X_test_raw.shape[1]}")

    # --------------------------------------------------------
    # Load original feature names
    # --------------------------------------------------------

    raw_path = "data/raw/parkinsons.csv"

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Could not find {raw_path}."
        )

    df_raw = pd.read_csv(raw_path)

    feature_names = [
        column
        for column in df_raw.columns
        if column not in ["name", "status"]
    ]

    # Make sure X_test contains the expected features
    missing_features = [
        feature for feature in feature_names
        if feature not in X_test_raw.columns
    ]

    if missing_features:
        raise ValueError(
            "The following expected features are missing from X_test.csv:\n"
            + "\n".join(missing_features)
        )

    # Ensure exact feature order
    X_test_raw = X_test_raw[feature_names]

    print(f"   ✅ Original feature names loaded: {len(feature_names)}")

    # --------------------------------------------------------
    # IMPORTANT:
    # Transform test data using the SAME scaler from Pipeline
    # --------------------------------------------------------

    X_test_scaled_array = scaler.transform(X_test_raw)

    X_test_scaled = pd.DataFrame(
        X_test_scaled_array,
        columns=feature_names,
        index=X_test_raw.index
    )

    print("   ✅ Test data transformed using trained StandardScaler")

    return (
        model,
        scaler,
        classifier,
        X_test_raw,
        X_test_scaled,
        feature_names
    )


# ============================================================
# GENERATE SHAP VALUES
# ============================================================

def explain_model(classifier, X_test_scaled):
    """
    Generate SHAP explanations for the trained SVM.

    For a linear SVC, SHAP LinearExplainer is used to explain
    the decision function.

    Returns:
        explainer
        shap_values
    """

    print("\n🔍 Generating SHAP explanations...")

    # --------------------------------------------------------
    # Make sure classifier is linear SVC
    # --------------------------------------------------------

    if not hasattr(classifier, "decision_function"):
        raise TypeError(
            "The classifier does not provide decision_function(). "
            "Cannot explain it as an SVM decision function."
        )

    if hasattr(classifier, "kernel"):
        print(f"   SVM kernel: {classifier.kernel}")

        if classifier.kernel != "linear":
            print(
                "   ⚠️ SVM kernel is not linear. "
                "LinearExplainer is intended for linear models."
            )

    # --------------------------------------------------------
    # Use SHAP LinearExplainer
    # --------------------------------------------------------

    print("   Using SHAP LinearExplainer on scaled SVM input...")

    try:
        explainer = shap.LinearExplainer(
            classifier,
            X_test_scaled
        )

        shap_values = explainer(X_test_scaled)

        # Convert Explanation object to ndarray
        if hasattr(shap_values, "values"):
            shap_array = shap_values.values
        else:
            shap_array = np.asarray(shap_values)

    except Exception as e:
        print(f"   ⚠️ LinearExplainer failed: {e}")
        print("   Trying a model-agnostic SHAP fallback...")

        # Fallback that directly explains the SVM decision function.
        background = shap.sample(
            X_test_scaled,
            min(20, len(X_test_scaled)),
            random_state=42
        )

        def decision_function(data):
            return classifier.decision_function(data)

        explainer = shap.KernelExplainer(
            decision_function,
            background
        )

        shap_array = explainer.shap_values(
            X_test_scaled,
            nsamples=100
        )

        shap_array = np.asarray(shap_array)

    # --------------------------------------------------------
    # Normalize SHAP output shape
    # --------------------------------------------------------

    if shap_array.ndim == 3:
        # Some SHAP versions can return:
        # samples x features x outputs
        shap_array = shap_array[:, :, 0]

    if shap_array.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP array shape: {shap_array.shape}"
        )

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if shap_array.shape[0] != X_test_scaled.shape[0]:
        raise ValueError(
            "Number of SHAP samples does not match X_test."
        )

    if shap_array.shape[1] != X_test_scaled.shape[1]:
        raise ValueError(
            "Number of SHAP features does not match X_test."
        )

    # --------------------------------------------------------
    # Critical diagnostic
    # --------------------------------------------------------

    max_abs_value = np.max(np.abs(shap_array))

    if max_abs_value == 0:
        raise ValueError(
            "SHAP values are all exactly zero. "
            "The explanation is invalid and will not be treated "
            "as a successful result."
        )

    print(
        f"   ✅ SHAP values calculated: {shap_array.shape}"
    )

    print(
        f"   ✅ Maximum absolute SHAP value: "
        f"{max_abs_value:.6f}"
    )

    return explainer, shap_array


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

def plot_global_importance(
    shap_values,
    X_test_scaled,
    feature_names
):
    """
    Create global mean absolute SHAP feature importance.
    """

    print("\n📊 Creating global feature importance plot...")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean Absolute SHAP": mean_abs_shap
    })

    importance_df = importance_df.sort_values(
        "Mean Absolute SHAP",
        ascending=True
    )

    # Keep top 15 for readability
    plot_df = importance_df.tail(15)

    plt.figure(figsize=(10, 8))

    plt.barh(
        plot_df["Feature"],
        plot_df["Mean Absolute SHAP"],
        color="#1F7A8C"
    )

    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.title(
        "Global SHAP Feature Importance\n"
        "SVM — Subject-Aware Evaluation",
        fontsize=14,
        fontweight="bold"
    )

    plt.grid(
        axis="x",
        alpha=0.25
    )

    plt.tight_layout()

    output_path = "figures/shap_global_importance.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"   ✅ Saved: {output_path}")


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

def plot_summary(
    shap_values,
    X_test_scaled,
    feature_names
):
    """
    Create SHAP beeswarm summary plot.
    """

    print("\n📊 Creating SHAP summary plot...")

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        X_test_scaled,
        feature_names=feature_names,
        show=False,
        max_display=15
    )

    plt.title(
        "SHAP Summary Plot\n"
        "SVM — Subject-Aware Evaluation",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    output_path = "figures/shap_summary_plot.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"   ✅ Saved: {output_path}")


# ============================================================
# SHAP WATERFALL
# ============================================================

def plot_waterfall(
    explainer,
    shap_values,
    X_test_scaled,
    feature_names,
    sample_index=0
):
    """
    Create SHAP waterfall plot for one test prediction.
    """

    print(
        "\n📊 Creating SHAP waterfall plot "
        "for a single prediction..."
    )

    try:

        # ----------------------------------------------------
        # Determine base value
        # ----------------------------------------------------

        if hasattr(explainer, "expected_value"):
            base_value = explainer.expected_value

            if isinstance(base_value, np.ndarray):
                base_value = base_value.flatten()[0]

            base_value = float(base_value)

        else:
            base_value = 0.0

        # ----------------------------------------------------
        # Create Explanation object
        # ----------------------------------------------------

        explanation = shap.Explanation(
            values=shap_values[sample_index],
            base_values=base_value,
            data=X_test_scaled.iloc[sample_index].values,
            feature_names=feature_names
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        plt.figure(figsize=(12, 8))

        shap.plots.waterfall(
            explanation,
            max_display=10,
            show=False
        )

        plt.tight_layout()

        output_path = "figures/shap_waterfall.png"

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"   ✅ Saved: {output_path}")

    except Exception as e:
        print(
            f"   ⚠️ Waterfall plot could not be created: {e}"
        )


# ============================================================
# TOP FEATURES
# ============================================================

def print_top_features(
    feature_names,
    shap_values
):
    """
    Print top 10 features by mean absolute SHAP value.
    """

    print(
        "\n🔝 Top 10 Features Influencing "
        "SVM Decision Function:"
    )

    print("=" * 65)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean Absolute SHAP": mean_abs_shap
    })

    importance_df = importance_df.sort_values(
        "Mean Absolute SHAP",
        ascending=False
    )

    for rank, (_, row) in enumerate(
        importance_df.head(10).iterrows(),
        start=1
    ):
        print(
            f"   {rank:2d}. "
            f"{row['Feature']:<25} "
            f"{row['Mean Absolute SHAP']:.6f}"
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_explainability():
    """
    Run complete SHAP explainability pipeline.
    """

    print("=" * 60)
    print(
        "🔬 EXPLAINABLE AI WITH SHAP "
        "(SUBJECT-AWARE SVM)"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load model and correctly scaled data
    # --------------------------------------------------------

    (
        model,
        scaler,
        classifier,
        X_test_raw,
        X_test_scaled,
        feature_names
    ) = load_model_and_data()

    # --------------------------------------------------------
    # Generate SHAP values
    # --------------------------------------------------------

    explainer, shap_values = explain_model(
        classifier,
        X_test_scaled
    )

    # --------------------------------------------------------
    # Generate visualizations
    # --------------------------------------------------------

    plot_global_importance(
        shap_values,
        X_test_scaled,
        feature_names
    )

    plot_summary(
        shap_values,
        X_test_scaled,
        feature_names
    )

    plot_waterfall(
        explainer,
        shap_values,
        X_test_scaled,
        feature_names,
        sample_index=0
    )

    # --------------------------------------------------------
    # Print top features
    # --------------------------------------------------------

    print_top_features(
        feature_names,
        shap_values
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("✅ EXPLAINABILITY COMPLETE!")
    print("=" * 60)

    print("\n📁 Figures saved in 'figures/' directory:")
    print("   - shap_global_importance.png")
    print("   - shap_summary_plot.png")
    print("   - shap_waterfall.png")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_explainability()