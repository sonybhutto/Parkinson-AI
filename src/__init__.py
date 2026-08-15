"""
Parkinson's Disease Classification - Source Package
Explainable Machine Learning for Voice Biomarker Analysis

This package contains all core functionality for:
- Data preprocessing with subject-aware splitting
- Model training with GroupKFold cross-validation
- Model evaluation and comparison
- SHAP explainability

Author: Gul Jan
Date: August 2026
"""

# Import from preprocessing
from .preprocessing import (
    run_preprocessing,
    load_data_with_subjects,
    split_by_subject,
    clean_data,
    scale_features,
    save_processed_data
)

# Import from models
from .models import (
    train_all_models,
    load_processed_data,
    get_models,
    evaluate_model_leakage_free
)

# Import from evaluation
from .evaluation import (
    run_evaluation,
    load_results,
    get_subject_stats,
    create_confusion_matrix,
    create_roc_curve,
    create_model_comparison_chart
)

# Import from explainability
from .explainability import (
    run_explainability,
    load_model_and_data,
    explain_model,
    plot_global_importance,
    plot_summary
)

__version__ = "1.0.0"
__author__ = "Gul Jan"

__all__ = [
    # Preprocessing
    'run_preprocessing',
    'load_data_with_subjects',
    'split_by_subject',
    'clean_data',
    'scale_features',
    'save_processed_data',
    
    # Models
    'train_all_models',
    'load_processed_data',
    'get_models',
    'evaluate_model_leakage_free',
    
    # Evaluation
    'run_evaluation',
    'load_results',
    'get_subject_stats',
    'create_confusion_matrix',
    'create_roc_curve',
    'create_model_comparison_chart',
    
    # Explainability
    'run_explainability',
    'load_model_and_data',
    'explain_model',
    'plot_global_importance',
    'plot_summary',
    
    # Metadata
    '__version__',
    '__author__'
]