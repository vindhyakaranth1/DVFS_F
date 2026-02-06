"""
Smart-Watt Feature Engineering Module
Adapted from vindhya/DVFS_F repository

This module provides feature engineering functions for CPU utilization data
using temporal windowing and statistical features.
"""

import numpy as np
import pandas as pd


def build_features_smartwatt(cpu_values, window=5):
    """
    Build Smart-Watt style features from CPU utilization.
    
    Features created:
    - Raw window values (5 features)
    - Deltas/differences (4 features)
    - Mean and Std (2 features)
    Total: 11 features per sample
    
    Args:
        cpu_values: Array of CPU utilization values (0-1 scale)
        window: Number of past samples to use (default: 5)
    
    Returns:
        X: Feature matrix of shape (n_samples - window, 11)
    
    Example:
        >>> cpu_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        >>> X = build_features_smartwatt(cpu_data, window=3)
        >>> X.shape
        (3, 7)  # 3 raw + 2 deltas + 2 stats
    """
    X = []
    
    for i in range(window, len(cpu_values)):
        window_data = cpu_values[i - window:i]
        
        features = []
        
        # 1. Raw window values
        features.extend(window_data)
        
        # 2. Deltas - rate of change between consecutive samples
        features.extend(np.diff(window_data))
        
        # 3. Statistics
        features.append(np.mean(window_data))
        features.append(np.std(window_data))
        
        X.append(features)
    
    return np.array(X)


def build_labels_horizon(cpu_values, window=5, horizon=5, threshold=0.30):
    """
    Build horizon-based binary labels for classification.
    
    Instead of predicting current CPU state, we predict the average
    CPU utilization over the next 'horizon' samples. This enables
    predictive DVFS that can scale frequency BEFORE load increases.
    
    Args:
        cpu_values: Array of CPU utilization (0-1 scale)
        window: Feature window size (must match build_features_smartwatt)
        horizon: How many samples ahead to predict (default: 5)
        threshold: CPU threshold for HIGH frequency (default: 30%)
    
    Returns:
        y: Binary labels (1 = HIGH freq needed, 0 = LOW freq)
    
    Example:
        >>> cpu_data = np.array([0.1, 0.2, 0.8, 0.9, 0.8, 0.7, 0.2])
        >>> y = build_labels_horizon(cpu_data, window=2, horizon=3, threshold=0.5)
        >>> y
        array([1, 1])  # Both predict HIGH because future avg > 0.5
    """
    y = []
    
    # Skip first 'window' samples (used for features)
    for i in range(window, len(cpu_values) - horizon):
        # Look ahead 'horizon' samples and take average
        future_avg = np.mean(cpu_values[i:i + horizon])
        
        # Binary classification: HIGH (1) or LOW (0)
        y.append(1 if future_avg > threshold else 0)
    
    return np.array(y)


def prepare_dataset(csv_path, cpu_column, window=5, horizon=5, 
                   threshold=0.30, normalize=True):
    """
    Complete pipeline: Load CSV, build features, create labels.
    
    Args:
        csv_path: Path to CSV file with CPU data
        cpu_column: Name of CPU utilization column
        window: Feature window size
        horizon: Prediction horizon
        threshold: CPU threshold for HIGH frequency
        normalize: Whether to normalize CPU values to 0-1 (if in 0-100 range)
    
    Returns:
        X: Feature matrix
        y: Labels
        df: Original dataframe (for reference)
    
    Example:
        >>> X, y, df = prepare_dataset('cpu_log.csv', 'cpu_util')
        >>> print(f"Features: {X.shape}, Labels: {y.shape}")
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Get CPU values
    cpu_vals = df[cpu_column].values
    
    # Normalize if needed (detect 0-100 range)
    if normalize and cpu_vals.max() > 1.0:
        print(f"Normalizing CPU values (max: {cpu_vals.max():.2f}) to 0-1 scale")
        cpu_vals = cpu_vals / 100.0
    
    # Build features
    X = build_features_smartwatt(cpu_vals, window=window)
    
    # Build labels
    y = build_labels_horizon(cpu_vals, window=window, horizon=horizon, 
                            threshold=threshold)
    
    # Align X and y (y might be shorter due to horizon)
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]
    
    print(f"\n✅ Dataset prepared:")
    print(f"  Features: {X.shape}")
    print(f"  Labels: {y.shape}")
    print(f"  Class distribution: HIGH={y.sum()} ({y.mean():.1%}), "
          f"LOW={len(y)-y.sum()} ({1-y.mean():.1%})")
    
    return X, y, df


def get_feature_names(window=5):
    """
    Get descriptive names for all features.
    
    Args:
        window: Window size used in feature engineering
    
    Returns:
        List of feature names
    """
    names = []
    
    # Raw values
    for i in range(window, 0, -1):
        names.append(f'CPU_t-{i}')
    
    # Deltas
    for i in range(1, window):
        names.append(f'Delta_{i}')
    
    # Statistics
    names.extend(['Mean', 'Std'])
    
    return names


if __name__ == "__main__":
    # Test the module
    print("Testing Smart-Watt Feature Engineering Module\n")
    
    # Generate synthetic CPU data
    np.random.seed(42)
    cpu_test = np.random.rand(100)
    
    # Build features
    X_test = build_features_smartwatt(cpu_test, window=5)
    y_test = build_labels_horizon(cpu_test, window=5, horizon=5)
    
    print(f"Test data: {len(cpu_test)} samples")
    print(f"Features: {X_test.shape}")
    print(f"Labels: {y_test.shape}")
    print(f"\nFeature names: {get_feature_names(window=5)}")
    print(f"\nSample feature vector:")
    print(X_test[0])
    print(f"\nLabel: {'HIGH' if y_test[0] == 1 else 'LOW'}")
