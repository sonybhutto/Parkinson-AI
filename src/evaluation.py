"""
Evaluation utilities for Parkinson's Disease Classification
Generates metrics, visualizations, and comparison tables
Author: Gul Jan
Date: August 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

def load_results():
    """Load model comparison results"""
    try:
        results_df = pd.read_csv('models/model_results.csv')
        return results_df
    except FileNotFoundError:
        print("⚠️ No results found. Run models.py first.")
        return None

def load_best_model_info():
    """Load best model information"""
    try:
        best_model = joblib.load('models/best_model.pkl')
        # Extract model name from pipeline
        if hasattr(best_model, 'named_steps'):
            model_type = type(best_model.named_steps['classifier']).__name__
        else:
            model_type = type(best_model).__name__
        return best_model, model_type
    except FileNotFoundError:
        print("⚠️ Best model not found. Run models.py first.")
        return None, None

def load_test_data():
    """Load test data for evaluation"""
    try:
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        subject_test = pd.read_csv('data/processed/subject_test.csv').values.ravel()
        return X_test, y_test, subject_test
    except FileNotFoundError:
        print("⚠️ Test data not found. Run preprocessing.py first.")
        return None, None, None

def get_subject_stats():
    """Get subject-level statistics"""
    try:
        df = pd.read_csv('data/raw/parkinsons.csv')
        df['subject_id'] = df['name'].str.extract(r'S(\d+)').astype(int)
        
        total_subjects = df['subject_id'].nunique()
        recordings_per_subject = df['subject_id'].value_counts()
        
        # Get class distribution by subject
        subject_status = df.groupby('subject_id')['status'].first()
        parkinson_subjects = (subject_status == 1).sum()
        healthy_subjects = (subject_status == 0).sum()
        
        return {
            'total_subjects': total_subjects,
            'recordings_per_subject': recordings_per_subject,
            'parkinson_subjects': parkinson_subjects,
            'healthy_subjects': healthy_subjects,
            'total_recordings': len(df)
        }
    except FileNotFoundError:
        return None

def create_confusion_matrix(y_test, y_pred, model_name="Model"):
    """Create and save confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Parkinson\'s'],
                yticklabels=['Healthy', 'Parkinson\'s'],
                ax=ax, cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    return cm

def create_roc_curve(y_test, y_proba, model_name="Model"):
    """Create and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})', color='#1F7A8C', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    return auc

def create_model_comparison_chart(results_df):
    """Create model comparison bar chart"""
    if results_df is None:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(results_df['Model']))
    width = 0.2
    
    colors = ['#123B5D', '#1F7A8C', '#4A9BAA', '#7BC0C9']
    
    for i, metric in enumerate(metrics):
        values = results_df[metric].astype(float)
        ax.bar(x + i*width, values, width, label=metric, color=colors[i])
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_subject_distribution():
    """Create subject distribution visualization"""
    stats = get_subject_stats()
    if stats is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Recordings per subject
    recordings = stats['recordings_per_subject']
    axes[0].hist(recordings, bins=range(4, 10), color='#1F7A8C', edgecolor='white')
    axes[0].set_xlabel('Recordings per Subject')
    axes[0].set_ylabel('Number of Subjects')
    axes[0].set_title('Recordings per Subject', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Class distribution by subject
    subject_counts = [stats['healthy_subjects'], stats['parkinson_subjects']]
    axes[1].bar(['Healthy', 'Parkinson\'s'], subject_counts, color=['#2E8B57', '#C94C4C'])
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Number of Subjects')
    axes[1].set_title('Class Distribution by Subject', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Subject-Level Dataset Statistics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/subject_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_evaluation():
    """Run complete evaluation pipeline"""
    print("=" * 60)
    print("📊 EVALUATION PIPELINE")
    print("=" * 60)
    
    # Load results
    results_df = load_results()
    if results_df is not None:
        print("\n📊 Model Performance Summary:")
        print(results_df.to_string(index=False))
        create_model_comparison_chart(results_df)
        print("   ✅ Model comparison chart saved")
    
    # Load best model
    best_model, model_type = load_best_model_info()
    if best_model is not None:
        print(f"\n🏆 Best Model: {model_type}")
    
    # Load test data and evaluate
    X_test, y_test, subject_test = load_test_data()
    if X_test is not None and best_model is not None:
        try:
            y_pred = best_model.predict(X_test)
            
            # Get probabilities if available
            try:
                y_proba = best_model.predict_proba(X_test)[:, 1]
                auc = create_roc_curve(y_test, y_proba, model_type)
                print(f"   ✅ ROC curve saved (AUC: {auc:.3f})")
            except:
                print("   ⚠️ ROC not available for this model")
            
            cm = create_confusion_matrix(y_test, y_pred, model_type)
            print("   ✅ Confusion matrix saved")
            
            # Classification report
            report = classification_report(y_test, y_pred, 
                                          target_names=['Healthy', 'Parkinson\'s'])
            print("\n📋 Classification Report:")
            print(report)
            
        except Exception as e:
            print(f"   ⚠️ Evaluation error: {e}")
    
    # Create subject distribution
    create_subject_distribution()
    print("   ✅ Subject distribution saved")
    
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()