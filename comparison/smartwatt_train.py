"""
Smart-Watt Model Training Module
Adapted from vindhya/DVFS_F repository

Trains a Random Forest Classifier for CPU frequency prediction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


def train_smartwatt_classifier(X, y, model_name="smartwatt", save_path=None):
    """
    Train Random Forest classifier using Smart-Watt parameters.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Binary labels (n_samples,)
        model_name: Name for the model (for printing)
        save_path: Path to save trained model (optional)
    
    Returns:
        model: Trained classifier
        y_prob: Prediction probabilities for all samples
        metrics: Dictionary with accuracy and other metrics
    """
    print(f"\n{'='*60}")
    print(f"TRAINING {model_name.upper()} MODEL")
    print(f"{'='*60}")
    
    # Time-aware split (NO SHUFFLE - preserve temporal order)
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nDataset split:")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Train HIGH ratio: {y_train.mean():.1%}")
    print(f"  Test HIGH ratio: {y_test.mean():.1%}")
    
    # Smart-Watt model configuration
    print(f"\nModel configuration:")
    print(f"  Algorithm: Random Forest")
    print(f"  Trees: 400")
    print(f"  Max depth: 14")
    print(f"  Class weight: balanced")
    
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        class_weight="balanced",  # Handle class imbalance
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    # Train
    print(f"\n🔧 Training model...")
    model.fit(X_train, y_train)
    print(f"✅ Training complete!")
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_prob = model.predict_proba(X)[:, 1]  # Probability of HIGH class
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    print(f"\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, digits=3, 
                                target_names=['LOW', 'HIGH']))
    
    # Feature importance
    feature_names = [
        'CPU_t-5', 'CPU_t-4', 'CPU_t-3', 'CPU_t-2', 'CPU_t-1',
        'Delta_1', 'Delta_2', 'Delta_3', 'Delta_4',
        'Mean', 'Std'
    ]
    
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🔝 Feature Importance (Top 5):")
    for idx, row in importances.head().iterrows():
        print(f"  {row['Feature']:12s}: {row['Importance']:.4f}")
    
    # Save model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)
        print(f"\n💾 Model saved: {save_path}")
        
        # Save probabilities
        prob_path = save_path.replace('.pkl', '_probs.npy')
        np.save(prob_path, y_prob)
        print(f"💾 Probabilities saved: {prob_path}")
    
    metrics = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'confusion_matrix': cm,
        'feature_importance': importances
    }
    
    return model, y_prob, metrics


def load_smartwatt_model(model_path):
    """
    Load a trained Smart-Watt model.
    
    Args:
        model_path: Path to saved model (.pkl file)
    
    Returns:
        model: Loaded classifier
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"✅ Model loaded from: {model_path}")
    
    return model


def predict_cpu_frequency(model, X, return_probs=True):
    """
    Predict CPU frequency needs using trained model.
    
    Args:
        model: Trained classifier
        X: Feature matrix
        return_probs: Whether to return probabilities (default: True)
    
    Returns:
        predictions: Binary predictions (0=LOW, 1=HIGH)
        probabilities: Probability of HIGH class (if return_probs=True)
    """
    predictions = model.predict(X)
    
    if return_probs:
        probabilities = model.predict_proba(X)[:, 1]
        return predictions, probabilities
    
    return predictions


if __name__ == "__main__":
    # Test the module
    print("Testing Smart-Watt Training Module\n")
    
    # Generate synthetic data
    from smartwatt_features import build_features_smartwatt, build_labels_horizon
    
    np.random.seed(42)
    cpu_test = np.random.rand(1000)
    
    X_test = build_features_smartwatt(cpu_test, window=5)
    y_test = build_labels_horizon(cpu_test, window=5, horizon=5)
    
    # Align
    min_len = min(len(X_test), len(y_test))
    X_test = X_test[:min_len]
    y_test = y_test[:min_len]
    
    # Train
    model, y_prob, metrics = train_smartwatt_classifier(
        X_test, y_test, 
        model_name="Test Model",
        save_path="models/test_model.pkl"
    )
    
    print(f"\n✅ Test complete!")
    print(f"   Accuracy: {metrics['test_accuracy']*100:.2f}%")
