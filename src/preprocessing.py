"""
Data Preprocessing Pipeline for Parkinson's Disease Detection
Subject-Aware Splitting to Prevent Data Leakage

Important:
- Raw features are saved after subject-aware splitting.
- Feature scaling is NOT performed here.
- Scaling is handled inside the ML pipeline during cross-validation.
- This prevents preprocessing leakage between CV folds.

Author: Gul Jan
Date: August 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import os
import warnings

warnings.filterwarnings('ignore')


def load_data_with_subjects():
    """
    Load the Parkinson's dataset and extract subject IDs.

    Each subject has multiple voice recordings.
    All recordings from one subject must remain in the same
    train/test split.
    """

    print("📂 Loading dataset with subject IDs...")

    df = pd.read_csv('data/raw/parkinsons.csv')

    # Extract subject ID from name
    # Example:
    # phon_R01_S01_1 -> subject_id = 1
    df['subject_id'] = df['name'].str.extract(
        r'S(\d+)'
    ).astype(int)

    print(
        f"   ✅ Loaded {df.shape[0]} samples "
        f"from {df['subject_id'].nunique()} unique subjects"
    )

    print("   📊 Samples per subject:")
    print(f"      Min: {df['subject_id'].value_counts().min()}")
    print(f"      Max: {df['subject_id'].value_counts().max()}")
    print(f"      Avg: {df['subject_id'].value_counts().mean():.1f}")

    return df


def clean_data(df):
    """
    Basic dataset cleaning.

    Missing values are NOT filled here because that operation
    can also cause data leakage if calculated before splitting.

    The current UCI dataset contains no missing values.
    """

    print("🧹 Cleaning data...")

    # Remove identifier column
    if 'name' in df.columns:
        df = df.drop('name', axis=1)
        print("   - Removed 'name' column")

    # Check missing values
    missing = df.isnull().sum().sum()

    if missing == 0:
        print("   ✅ No missing values found")
    else:
        print(
            f"   ⚠️ Found {missing} missing values."
        )
        print(
            "   Missing-value handling will be performed "
            "inside the ML pipeline."
        )

    return df


def split_by_subject(df, test_size=0.2, random_state=42):
    """
    Split data by SUBJECT instead of individual recordings.

    This guarantees that recordings from the same subject
    cannot appear in both training and test sets.
    """

    print("✂️ Splitting data by SUBJECT (no leakage!)...")

    # Features
    X = df.drop(
        ['status', 'subject_id'],
        axis=1
    )

    # Target
    y = df['status']

    # Subject groups
    groups = df['subject_id']

    # Subject-aware split
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, test_idx = next(
        gss.split(
            X,
            y,
            groups=groups
        )
    )

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    subject_train = groups.iloc[train_idx].copy()
    subject_test = groups.iloc[test_idx].copy()

    # Verify no subject overlap
    train_subjects = set(subject_train.unique())
    test_subjects = set(subject_test.unique())

    overlap = train_subjects.intersection(test_subjects)

    print(
        f"   ✅ Training set: {X_train.shape[0]} samples "
        f"from {subject_train.nunique()} subjects"
    )

    print(
        f"   ✅ Test set: {X_test.shape[0]} samples "
        f"from {subject_test.nunique()} subjects"
    )

    if len(overlap) == 0:
        print("   ✅ No subject overlap between train and test!")
    else:
        raise ValueError(
            f"❌ Subject leakage detected! "
            f"Overlapping subjects: {overlap}"
        )

    # Target distribution
    print("\n   📊 Training target distribution:")
    print(y_train.value_counts().to_string())

    print("\n   📊 Test target distribution:")
    print(y_test.value_counts().to_string())

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        subject_train,
        subject_test
    )


def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test,
    subject_train,
    subject_test
):
    """
    Save RAW feature values.

    IMPORTANT:
    We intentionally do NOT scale here.

    Scaling will happen inside the ML Pipeline so that
    each GroupKFold validation fold gets its scaler fitted
    only on its own training portion.
    """

    print("\n💾 Saving processed data...")

    os.makedirs(
        'data/processed',
        exist_ok=True
    )

    # Save features with their original column names
    X_train.to_csv(
        'data/processed/X_train.csv',
        index=False
    )

    X_test.to_csv(
        'data/processed/X_test.csv',
        index=False
    )

    # Save target
    pd.DataFrame({
        'status': y_train.values
    }).to_csv(
        'data/processed/y_train.csv',
        index=False
    )

    pd.DataFrame({
        'status': y_test.values
    }).to_csv(
        'data/processed/y_test.csv',
        index=False
    )

    # Save subject IDs
    pd.DataFrame({
        'subject_id': subject_train.values
    }).to_csv(
        'data/processed/subject_train.csv',
        index=False
    )

    pd.DataFrame({
        'subject_id': subject_test.values
    }).to_csv(
        'data/processed/subject_test.csv',
        index=False
    )

    print(
        "   ✅ Raw feature values saved "
        "with original feature names"
    )

    print(
        "   ✅ Scaling will be handled inside ML pipelines"
    )


def run_preprocessing():
    """
    Run complete preprocessing pipeline.
    """

    print("=" * 60)
    print(
        "🔬 DATA PREPROCESSING PIPELINE "
        "(Leakage-Free)"
    )
    print("=" * 60)

    # Load
    df = load_data_with_subjects()

    # Clean
    df = clean_data(df)

    # Subject-aware split
    (
        X_train,
        X_test,
        y_train,
        y_test,
        subject_train,
        subject_test
    ) = split_by_subject(df)

    # Save raw split data
    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test,
        subject_train,
        subject_test
    )

    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 60)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        subject_train,
        subject_test
    )


if __name__ == "__main__":
    run_preprocessing()