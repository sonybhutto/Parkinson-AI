"""
Parkinson's Disease Classification
Explainable Machine Learning for Voice Biomarker Analysis

Research-oriented Streamlit application.

Author: Gul Jan
Date: August 2026
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA = os.path.join(DATA_DIR, "raw", "parkinsons.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Parkinson's AI Research",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL COLOR THEME
# ============================================================

NAVY = "#123B5D"
DARK_NAVY = "#0B263D"
TEAL = "#1F7A8C"
LIGHT_TEAL = "#4A9BAA"
PALE_TEAL = "#EAF6F7"
GREEN = "#2E8B57"
RED = "#C94C4C"
GOLD = "#D89B35"
TEXT = "#203040"
MUTED = "#5B6B7A"
WHITE = "#FFFFFF"
LIGHT_BG = "#F6F9FB"
BORDER = "#D9E4EA"


# ============================================================
# EXACT UCI PARKINSON'S FEATURE ORDER
# ============================================================

FEATURE_NAMES = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE"
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    f"""
<style>

    .stApp {{
        background-color: {LIGHT_BG};
        color: {TEXT};
    }}

    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {DARK_NAVY};
        border-right: 1px solid #173E58;
    }}

    section[data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
    }}

    section[data-testid="stSidebar"] .stRadio label {{
        color: #FFFFFF !important;
    }}

    section[data-testid="stSidebar"] .stRadio label:hover {{
        color: #8ED8E2 !important;
        background-color: rgba(255,255,255,0.08) !important;
        border-radius: 8px;
    }}

    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        color: #FFFFFF !important;
        background-color: transparent !important;
    }}

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        color: #8ED8E2 !important;
        background-color: rgba(31,122,140,0.25) !important;
    }}

    section[data-testid="stSidebar"] div[role="radiogroup"] label p {{
        color: #FFFFFF !important;
    }}

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover p {{
        color: #8ED8E2 !important;
    }}

    h1, h2, h3 {{
        color: {NAVY} !important;
    }}

    /* Keep sidebar headings white */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: #FFFFFF !important;
    }}

    section[data-testid="stSidebar"] .stCaption {{
        color: #9FD7DE !important;
    }}

    .page-subtitle {{
        color: {MUTED};
        font-size: 16px;
        margin-top: -10px;
        margin-bottom: 25px;
    }}

    .research-card {{
        background-color: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(18,59,93,0.05);
    }}

    .metric-card {{
        background-color: {WHITE};
        border: 1px solid {BORDER};
        border-top: 4px solid {TEAL};
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        min-height: 125px;
    }}

    .metric-number {{
        color: {NAVY};
        font-size: 28px;
        font-weight: 700;
    }}

    .metric-label {{
        color: {MUTED};
        font-size: 14px;
        margin-top: 5px;
    }}

    .success-box {{
        background-color: #EAF7EF;
        border-left: 5px solid {GREEN};
        padding: 16px;
        border-radius: 8px;
        color: #245B3A;
        margin: 15px 0;
    }}

    .warning-box {{
        background-color: #FFF8E8;
        border-left: 5px solid {GOLD};
        padding: 16px;
        border-radius: 8px;
        color: #6D531B;
        margin: 15px 0;
    }}

    .danger-box {{
        background-color: #FCEEEE;
        border-left: 5px solid {RED};
        padding: 16px;
        border-radius: 8px;
        color: #7B2D2D;
        margin: 15px 0;
    }}

    .info-box {{
        background-color: {PALE_TEAL};
        border-left: 5px solid {TEAL};
        padding: 16px;
        border-radius: 8px;
        color: #245563;
        margin: 15px 0;
    }}

    .stButton > button {{
        background-color: {TEAL};
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
    }}

    .stButton > button:hover {{
        background-color: {NAVY};
        color: white !important;
        border: none;
    }}

    div[data-baseweb="input"] {{
        border-radius: 7px;
    }}

    .footer {{
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid {BORDER};
        text-align: center;
        color: {MUTED};
        font-size: 13px;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA / MODEL
# ============================================================

@st.cache_data
def load_raw_dataset():
    """Load original UCI Parkinson's dataset."""

    if not os.path.exists(RAW_DATA):
        return None

    return pd.read_csv(RAW_DATA)


@st.cache_resource
def load_best_model():
    """
    Load the COMPLETE trained ML Pipeline.

    The saved pipeline contains:
        StandardScaler
        +
        SVC(kernel='linear')

    RAW feature values must be passed directly to the pipeline.
    """

    path = os.path.join(
        MODEL_DIR,
        "best_model.pkl"
    )

    if not os.path.exists(path):
        return None

    return joblib.load(path)


@st.cache_data
def load_model_results():

    path = os.path.join(
        MODEL_DIR,
        "model_results.csv"
    )

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    for col in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


# ============================================================
# DATASET INFORMATION
# ============================================================

def get_dataset_stats(df):

    if df is None:
        return None

    working = df.copy()

    if "name" in working.columns:

        working["subject_id"] = (
            working["name"]
            .str.extract(r"S(\d+)")
            .astype(int)
        )

    return {
        "recordings": len(working),

        "subjects": (
            working["subject_id"].nunique()
            if "subject_id" in working.columns
            else None
        ),

        "features": len(
            [
                c for c in working.columns
                if c not in [
                    "name",
                    "status",
                    "subject_id"
                ]
            ]
        ),

        "parkinson": int(
            (working["status"] == 1).sum()
        ),

        "healthy": int(
            (working["status"] == 0).sum()
        )
    }


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    with st.sidebar:

        # ====================================================
        # SIDEBAR HEADER
        # ====================================================

        st.markdown(
            "<div style='text-align:center; font-size:48px;'>🧠</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "## Parkinson's AI"
        )

        st.caption(
            "Explainable ML Research"
        )

        st.markdown("---")

        st.markdown(
            "NAVIGATION"
        )

        page = st.radio(
            "Navigation",
            [
                "Prediction",
                "Data Exploration",
                "Model Comparison",
                "Explainable AI",
                "Experiments",
                "Research",
                "About"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # ====================================================
        # RESEARCH PROTOTYPE BOX
        # ====================================================

        st.markdown(
            """
            **Research Prototype**

            Subject-aware evaluation

            GroupKFold cross-validation

            Hyperparameter tuning

            SHAP explainability
            """
        )

        st.markdown("---")

        st.caption(
            "Academic research only"
        )

    return page


# ============================================================
# PAGE HEADER
# ============================================================

def page_header(title, subtitle):

    st.markdown(f"# {title}")

    st.markdown(
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

def page_prediction():

    page_header(
        "Parkinson's Disease Prediction",
        "Research prototype for voice-biomarker classification"
    )

    model = load_best_model()
    df = load_raw_dataset()

    if model is None:

        st.error(
            "The trained model was not found. "
            "Please run models.py first."
        )

        return

    if df is None:

        st.error(
            "The Parkinson's dataset was not found at "
            "`data/raw/parkinsons.csv`."
        )

        return

    stats = get_dataset_stats(df)

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    try:

        if hasattr(model, "named_steps"):

            classifier = model.named_steps["classifier"]

            model_name = type(classifier).__name__

        else:

            model_name = type(model).__name__

    except Exception:

        model_name = "Trained Model"

    st.markdown(
        f"""
        <div class="info-box">
            <b>Current trained model:</b> {model_name}<br>
            The complete trained pipeline handles preprocessing
            internally before generating predictions.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DATASET METRICS
    # --------------------------------------------------------

    if stats:

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-number">
                        {stats['recordings']}
                    </div>
                    <div class="metric-label">
                        Voice recordings
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-number">
                        {stats['subjects']}
                    </div>
                    <div class="metric-label">
                        Independent subjects
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-number">
                        {stats['features']}
                    </div>
                    <div class="metric-label">
                        Voice features
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:

            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-number">
                        SHAP
                    </div>
                    <div class="metric-label">
                        Explainability
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    st.markdown("### Voice Biomarker Input")

    st.markdown(
        """
        Enter values for the 22 voice biomarkers used by the UCI dataset.
        These values are intended for research experimentation only.
        """
    )

    missing_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in df.columns
    ]

    if missing_features:

        st.error(
            "The dataset is missing expected UCI features: "
            + ", ".join(missing_features)
        )

        return

    col1, col2 = st.columns(2)

    input_values = {}

    midpoint = len(FEATURE_NAMES) // 2

    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

    with col1:

        for feature in FEATURE_NAMES[:midpoint]:

            series = pd.to_numeric(
                df[feature],
                errors="coerce"
            )

            default_value = float(
                series.median()
            )

            min_value = float(
                series.min()
            )

            max_value = float(
                series.max()
            )

            input_values[feature] = st.number_input(
                feature,
                min_value=min_value,
                max_value=max_value,
                value=default_value,
                format="%.6f",
                help=(
                    f"Typical dataset range: "
                    f"{min_value:.6f} – {max_value:.6f}"
                )
            )

    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with col2:

        for feature in FEATURE_NAMES[midpoint:]:

            series = pd.to_numeric(
                df[feature],
                errors="coerce"
            )

            default_value = float(
                series.median()
            )

            min_value = float(
                series.min()
            )

            max_value = float(
                series.max()
            )

            input_values[feature] = st.number_input(
                feature,
                min_value=min_value,
                max_value=max_value,
                value=default_value,
                format="%.6f",
                help=(
                    f"Typical dataset range: "
                    f"{min_value:.6f} – {max_value:.6f}"
                )
            )

    st.markdown("---")

    predict_col, reset_col = st.columns([3, 1])

    with predict_col:

        predict_button = st.button(
            "🔬 Predict",
            use_container_width=True
        )

    with reset_col:

        if st.button(
            "Reset",
            use_container_width=True
        ):

            st.rerun()

    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        try:

            # The saved model is a COMPLETE Pipeline:
            #
            # StandardScaler -> SVC
            #
            # Therefore raw feature values are passed directly
            # into the pipeline.
            #
            # NO scaler.pkl is loaded.
            # NO manual scaling is performed.

            input_df = pd.DataFrame(
                [
                    [
                        input_values[feature]
                        for feature in FEATURE_NAMES
                    ]
                ],
                columns=FEATURE_NAMES
            )

            # ------------------------------------------------
            # RAW DATA -> COMPLETE PIPELINE
            # ------------------------------------------------

            prediction = model.predict(
                input_df
            )[0]

            # ------------------------------------------------
            # OPTIONAL PROBABILITY
            # ------------------------------------------------

            probability = None

            if hasattr(model, "predict_proba"):

                try:

                    probability = float(
                        model.predict_proba(
                            input_df
                        )[0][1]
                    )

                except Exception:

                    probability = None

            # ------------------------------------------------
            # SVM DECISION FUNCTION
            # ------------------------------------------------

            decision_score = None

            if hasattr(model, "decision_function"):

                try:

                    decision_score = float(
                        np.asarray(
                            model.decision_function(
                                input_df
                            )
                        ).ravel()[0]
                    )

                except Exception:

                    decision_score = None

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.markdown(
                "### Prediction Result"
            )

            if prediction == 1:

                st.markdown(
                    """
                    <div class="danger-box">
                        <h3 style="color:#7B2D2D !important;">
                            Model Output: Parkinson's Class
                        </h3>

                        The model classified this input as belonging
                        to the Parkinson's class.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="success-box">
                        <h3 style="color:#245B3A !important;">
                            Model Output: Healthy Class
                        </h3>

                        The model classified this input as belonging
                        to the healthy class.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ------------------------------------------------
            # PROBABILITY IF AVAILABLE
            # ------------------------------------------------

            if probability is not None:

                st.metric(
                    "Estimated model probability for Parkinson's class",
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(probability)
                )

            # ------------------------------------------------
            # SVM DECISION SCORE
            # ------------------------------------------------

            elif decision_score is not None:

                st.metric(
                    "SVM decision score",
                    f"{decision_score:.4f}"
                )

                st.markdown(
                    """
                    <div class="info-box">
                        <b>Interpretation:</b>
                        This is the SVM decision-function score, not a
                        probability. Values on opposite sides of the
                        classifier's decision boundary correspond to
                        different predicted classes.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ------------------------------------------------
            # SCIENTIFIC DISCLAIMER
            # ------------------------------------------------

            st.markdown(
                """
                <div class="warning-box">
                    <b>Important:</b> This is a machine-learning
                    research output, not a medical diagnosis.
                    The model has not been clinically validated and
                    must not be used for diagnosis, treatment,
                    screening, or patient risk assessment.
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(
                "Prediction could not be completed."
            )

            st.exception(e)


# ============================================================
# DATA EXPLORATION PAGE
# ============================================================

def page_data_exploration():

    page_header(
        "Data Exploration",
        "Understanding the UCI Parkinson's voice-biomarker dataset"
    )

    df = load_raw_dataset()

    if df is None:

        st.error("Dataset not found.")

        return

    st.markdown(
        "### Dataset Overview"
    )

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        ("195", "Total recordings"),
        ("32", "Independent subjects"),
        ("22", "Voice features"),
        ("2", "Classes")
    ]

    for col, (value, label) in zip(
        [c1, c2, c3, c4],
        metrics
    ):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-number">
                        {value}
                    </div>
                    <div class="metric-label">
                        {label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        "### Class Distribution"
    )

    class_counts = (
        df["status"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    labels = [
        "Healthy",
        "Parkinson's"
    ]

    values = [
        class_counts.get(0, 0),
        class_counts.get(1, 0)
    ]

    bars = ax.bar(
        labels,
        values,
        color=[GREEN, RED],
        width=0.55
    )

    ax.set_ylabel(
        "Number of recordings"
    )

    ax.set_title(
        "Recording-Level Class Distribution"
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    for bar, value in zip(
        bars,
        values
    ):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            str(value),
            ha="center",
            fontweight="bold"
        )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.markdown(
        "### Subject-Level Structure"
    )

    st.markdown(
        """
        <div class="info-box">
            <b>Why subject-level splitting matters:</b><br>
            The dataset contains multiple recordings from the same person.
            Randomly splitting individual recordings can place recordings
            from one person into both training and test sets, producing
            overly optimistic results. This project therefore separates
            subjects rather than individual recordings.
        </div>
        """,
        unsafe_allow_html=True
    )

    if "name" in df.columns:

        subject_df = df.copy()

        subject_df["subject_id"] = (
            subject_df["name"]
            .str.extract(r"S(\d+)")
            .astype(int)
        )

        recordings = (
            subject_df
            .groupby("subject_id")
            .size()
            .reset_index(
                name="recordings"
            )
        )

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        ax.hist(
            recordings["recordings"],
            bins=np.arange(
                recordings["recordings"].min() - 0.5,
                recordings["recordings"].max() + 1.5,
                1
            ),
            color=TEAL,
            edgecolor="white"
        )

        ax.set_xlabel(
            "Recordings per subject"
        )

        ax.set_ylabel(
            "Number of subjects"
        )

        ax.set_title(
            "Recordings per Independent Subject"
        )

        ax.grid(
            axis="y",
            alpha=0.2
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    st.markdown(
        "### Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        "### Feature List"
    )

    feature_table = pd.DataFrame(
        {
            "Feature": FEATURE_NAMES,
            "Data Type": [
                str(df[c].dtype)
                for c in FEATURE_NAMES
            ]
        }
    )

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

def page_model_comparison():

    page_header(
        "Model Comparison",
        "Leakage-aware comparison of machine-learning algorithms"
    )

    results = load_model_results()

    if results is None:

        st.error(
            "Model results were not found. Run models.py first."
        )

        return

    st.markdown(
        """
        <div class="info-box">
            Models were evaluated using a subject-aware training strategy.
            Hyperparameter tuning uses GroupKFold so recordings from the
            same subject do not appear across training and validation folds.
        </div>
        """,
        unsafe_allow_html=True
    )

    best_row = results.loc[
        results["F1-Score"].idxmax()
    ]

    st.markdown(
        "### Best Model"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {best_row['Model']}
                </div>

                <div class="metric-label">
                    Highest F1-Score
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {best_row['F1-Score']:.2%}
                </div>

                <div class="metric-label">
                    F1-Score
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {best_row['Recall']:.2%}
                </div>

                <div class="metric-label">
                    Recall
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "### Performance Table"
    )

    display_df = results.copy()

    for col in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ]:

        if col in display_df.columns:

            display_df[col] = display_df[col].apply(
                lambda x:
                    f"{x:.3f}"
                    if pd.notna(x)
                    else "N/A"
            )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        "### Performance Visualization"
    )

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]

    plot_df = results.copy()

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    x = np.arange(
        len(plot_df)
    )

    width = 0.18

    colors = [
        NAVY,
        TEAL,
        LIGHT_TEAL,
        "#7BC0C9"
    ]

    for i, metric in enumerate(metrics):

        ax.bar(
            x + (i - 1.5) * width,
            plot_df[metric],
            width,
            label=metric,
            color=colors[i]
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        plot_df["Model"],
        rotation=20,
        ha="right"
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Leakage-Aware Model Performance"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.markdown(
        "### Evaluation Metrics"
    )

    st.markdown(
        """
        **Accuracy:** Overall proportion of correct predictions.

        **Precision:** Proportion of predicted Parkinson's cases that
        actually belong to the Parkinson's class.

        **Recall:** Proportion of Parkinson's cases correctly identified
        by the model.

        **F1-Score:** Harmonic mean of precision and recall.

        **ROC-AUC:** Measures ranking ability across classification
        thresholds when probability estimates are available.
        """
    )


# ============================================================
# EXPLAINABLE AI
# ============================================================

def page_explainable_ai():

    page_header(
        "Explainable AI",
        "Understanding which voice biomarkers influence model predictions"
    )

    st.markdown(
        """
        <div class="info-box">
            This project uses <b>SHAP (SHapley Additive exPlanations)</b>
            to investigate model behavior and feature importance.
            Explainability helps researchers understand which voice
            biomarkers contribute most strongly to model predictions.
        </div>
        """,
        unsafe_allow_html=True
    )

    shap_global = os.path.join(
        FIGURES_DIR,
        "shap_global_importance.png"
    )

    shap_summary = os.path.join(
        FIGURES_DIR,
        "shap_summary_plot.png"
    )

    shap_waterfall = os.path.join(
        FIGURES_DIR,
        "shap_waterfall.png"
    )

    if os.path.exists(shap_global):

        st.markdown(
            "### Global Feature Importance"
        )

        st.image(
            shap_global,
            use_container_width=True
        )

    else:

        st.warning(
            "Global SHAP figure not found. "
            "Run explainability.py first."
        )

    if os.path.exists(shap_summary):

        st.markdown(
            "### SHAP Summary Plot"
        )

        st.image(
            shap_summary,
            use_container_width=True
        )

    if os.path.exists(shap_waterfall):

        st.markdown(
            "### Individual Prediction Explanation"
        )

        st.image(
            shap_waterfall,
            use_container_width=True
        )

    st.markdown(
        "### How to Interpret SHAP"
    )

    st.markdown(
        """
        SHAP values estimate how individual features contribute to a
        model output.

        - **Positive SHAP contribution:** pushes the model toward the
          Parkinson's class.
        - **Negative SHAP contribution:** pushes the model away from the
          Parkinson's class.
        - **Larger absolute SHAP value:** stronger influence on the
          prediction.

        These explanations describe **model behavior**, not biological
        or clinical causation.
        """
    )

    st.markdown(
        """
        <div class="warning-box">
            <b>Important scientific distinction:</b>
            Feature importance does not prove that a voice feature causes
            Parkinson's disease. SHAP explains associations learned by
            the machine-learning model.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# EXPERIMENTS
# ============================================================

def page_experiments():

    page_header(
        "Experiments",
        "Evaluation methodology and reproducibility"
    )

    st.markdown(
        "### Experimental Design"
    )

    experiment_data = [
        ["Dataset", "UCI Parkinson's Disease Dataset"],
        ["Recordings", "195"],
        ["Independent subjects", "32"],
        ["Features", "22 voice biomarkers"],
        ["Train/Test Strategy", "Subject-aware GroupShuffleSplit"],
        ["Cross-validation", "5-fold GroupKFold"],
        ["Hyperparameter tuning", "GridSearchCV"],
        ["Preprocessing", "StandardScaler inside Pipeline"],
        ["Selected model", "Linear SVM"],
        ["Explainability", "SHAP"],
        ["Primary selection metric", "F1-Macro"]
    ]

    experiment_df = pd.DataFrame(
        experiment_data,
        columns=[
            "Component",
            "Method"
        ]
    )

    st.dataframe(
        experiment_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        "### Subject-Level Leakage Prevention"
    )

    st.markdown(
        """
        The original dataset contains repeated voice recordings for the
        same subjects.

        A conventional random sample split could place recordings from
        one subject in both training and test sets.

        This project instead extracts the subject identifier and uses
        **GroupShuffleSplit** to create the final train/test partition.

        During model selection, **GroupKFold** ensures that the subject
        groups remain separated across cross-validation folds.
        """
    )

    try:

        train_subject_path = os.path.join(
            PROCESSED_DIR,
            "subject_train.csv"
        )

        test_subject_path = os.path.join(
            PROCESSED_DIR,
            "subject_test.csv"
        )

        if (
            os.path.exists(train_subject_path)
            and os.path.exists(test_subject_path)
        ):

            train_subjects = set(
                pd.read_csv(
                    train_subject_path
                ).iloc[:, 0]
            )

            test_subjects = set(
                pd.read_csv(
                    test_subject_path
                ).iloc[:, 0]
            )

            overlap = train_subjects.intersection(
                test_subjects
            )

            if len(overlap) == 0:

                st.markdown(
                    """
                    <div class="success-box">
                        <b>✓ Subject leakage check passed</b><br>
                        No subject appears in both the training and test sets.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="danger-box">
                        <b>⚠ Subject overlap detected</b><br>
                        {len(overlap)} subject(s) appear in both sets.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    except Exception as e:

        st.warning(
            f"Could not perform subject overlap check: {e}"
        )

    st.markdown(
        "### Generated Evaluation Figures"
    )

    figure_files = [
        (
            "Model Comparison",
            "model_comparison.png"
        ),
        (
            "Confusion Matrix",
            "confusion_matrix.png"
        ),
        (
            "ROC Curve",
            "roc_curve.png"
        ),
        (
            "Subject Distribution",
            "subject_distribution.png"
        )
    ]

    for title, filename in figure_files:

        path = os.path.join(
            FIGURES_DIR,
            filename
        )

        if os.path.exists(path):

            st.markdown(
                f"#### {title}"
            )

            st.image(
                path,
                use_container_width=True
            )


# ============================================================
# RESEARCH PAGE
# ============================================================

def page_research():

    page_header(
        "Research Methodology",
        "Scientific design, evaluation strategy, and interpretation"
    )

    st.markdown(
        "### Research Question"
    )

    st.markdown(
        """
        **Can machine-learning models classify Parkinson's disease from
        voice biomarkers while maintaining subject-independent evaluation
        and interpretable model behavior?**
        """
    )

    st.markdown(
        "### Research Pipeline"
    )

    pipeline = [
        (
            "01",
            "Dataset",
            "UCI Parkinson's voice recordings"
        ),
        (
            "02",
            "Subject Identification",
            "Extract independent subject IDs"
        ),
        (
            "03",
            "Data Splitting",
            "GroupShuffleSplit"
        ),
        (
            "04",
            "Model Selection",
            "GroupKFold cross-validation"
        ),
        (
            "05",
            "Optimization",
            "GridSearchCV"
        ),
        (
            "06",
            "Evaluation",
            "Accuracy, precision, recall, F1, ROC-AUC"
        ),
        (
            "07",
            "Explainability",
            "SHAP feature analysis"
        ),
        (
            "08",
            "Interpretation",
            "Research-oriented conclusions"
        )
    ]

    for number, title, description in pipeline:

        st.markdown(
            f"""
            <div class="research-card">

                <span style="
                    color:{TEAL};
                    font-weight:700;
                    font-size:14px;
                ">
                    {number}
                </span>

                <span style="
                    color:{NAVY};
                    font-size:18px;
                    font-weight:700;
                    margin-left:12px;
                ">
                    {title}
                </span>

                <br>

                <span style="
                    color:{MUTED};
                    margin-left:45px;
                ">
                    {description}
                </span>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "### Scientific Contribution"
    )

    st.markdown(
        """
        An important part of this project is demonstrating how evaluation
        methodology can substantially affect reported machine-learning
        performance.

        Because multiple recordings belong to the same individual,
        subject-aware evaluation provides a more realistic estimate of
        how the model may behave on previously unseen subjects.

        The project therefore emphasizes **methodological rigor over
        maximizing a performance number**.
        """
    )

    st.markdown(
        "### Limitations"
    )

    limitations = [
        "Small number of independent subjects (32)",
        "Multiple recordings per subject",
        "Dataset-specific characteristics",
        "Limited demographic diversity",
        "No external clinical validation",
        "Voice measurements may be influenced by factors unrelated to Parkinson's disease",
        "Research prototype only — not clinically validated"
    ]

    for limitation in limitations:

        st.markdown(
            f"- {limitation}"
        )

    st.markdown(
        f"""
        <div style="
            background-color:#FFF8E1;
            border-left:4px solid {GOLD};
            padding:14px 18px;
            border-radius:6px;
            margin-top:20px;
        ">
            <b>Scientific transparency:</b>
            These limitations are intentionally documented because
            reproducibility and appropriate interpretation are central
            to responsible machine-learning research.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ABOUT PAGE
# ============================================================

def page_about():

    page_header(
        "About",
        "Project overview and technical context"
    )

    st.markdown(
        """
        <div class="research-card">

        <h3>What is this project?</h3>

        A research-oriented machine-learning project investigating
        Parkinson's disease classification from voice biomarkers using
        subject-aware evaluation and Explainable AI.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### Dataset"
    )

    st.markdown(
        """
        **UCI Parkinson's Disease Dataset**

        - 195 voice recordings
        - 32 independent subjects
        - 22 voice biomarker features
        - Binary classification target
        """
    )

    st.markdown(
        "### Machine Learning"
    )

    st.markdown(
        """
        The project compares multiple algorithms:

        - Logistic Regression
        - Decision Tree
        - Random Forest
        - Support Vector Machine
        - K-Nearest Neighbors
        - Naive Bayes
        - XGBoost
        - LightGBM

        Hyperparameters are optimized using **GridSearchCV with
        GroupKFold cross-validation**.
        """
    )

    st.markdown(
        "### Explainability"
    )

    st.markdown(
        """
        **SHAP (SHapley Additive exPlanations)** is used to investigate
        which features influence model predictions.
        """
    )

    st.markdown(
        "### Technical Stack"
    )

    tech = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "Pandas / NumPy",
                "Scikit-learn",
                "XGBoost",
                "LightGBM",
                "SHAP",
                "Streamlit",
                "Matplotlib / Seaborn"
            ],
            "Purpose": [
                "Programming",
                "Data processing",
                "Machine learning",
                "Gradient boosting",
                "Gradient boosting",
                "Explainable AI",
                "Interactive research interface",
                "Visualization"
            ]
        }
    )

    st.dataframe(
        tech,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        "### Research Disclaimer"
    )

    st.markdown(
        """
        <div class="danger-box">
            <b>This model is a research prototype.</b><br><br>

            It is not clinically validated and must not be used for
            medical diagnosis, treatment decisions, clinical screening,
            or patient risk assessment.

            All predictions and explanations are research outputs and
            should be interpreted with caution.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    selected_page = render_sidebar()

    pages = {
        "Prediction": page_prediction,
        "Data Exploration": page_data_exploration,
        "Model Comparison": page_model_comparison,
        "Explainable AI": page_explainable_ai,
        "Experiments": page_experiments,
        "Research": page_research,
        "About": page_about
    }

    page_function = pages.get(
        selected_page,
        page_prediction
    )

    page_function()

    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown(
        "---"
    )

    st.markdown(
        """
        <div class="footer">
            <strong>Parkinson's AI Research Prototype</strong><br><br>

            Built with Streamlit, Scikit-learn, SHAP, XGBoost
            and LightGBM<br><br>

            <span style="font-size:11px;">
                Data source: UCI Machine Learning Repository —
                Parkinson's Disease Dataset<br>
                Research prototype — Not clinically validated
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()