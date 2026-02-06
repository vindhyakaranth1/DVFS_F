# Smart-Watt DVFS: ML-Based CPU Frequency Optimization

**Predictive Power Management using Machine Learning**


---

## 🎯 Project Overview

**Smart-Watt** is a machine learning-based Dynamic Voltage and Frequency Scaling (DVFS) system that predicts future CPU utilization and proactively adjusts processor frequency to minimize energy consumption while maintaining performance.

### Performance Achievements

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Model Accuracy | >90% | **96.73%** | ✅ |
| Energy Savings | 5-20% | **18.15%** | ✅ |
| Transition Reduction | >50% | **68.1%** | ✅ |
| Inference Time | <5ms | **0.87ms** | ✅ |

---

## 🏗️ System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Smart-Watt DVFS System                       │
└─────────────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┴──────────────────────┐
        │                                             │
┌───────▼────────┐                          ┌─────────▼────────┐
│  Data Pipeline │                          │  DVFS Governor   │
└───────┬────────┘                          └───────┬──────────┘
        │                                           │
  ┌─────┴─────┐                               ┌─────┴──────┐
  │           │                               │            │
┌─▼─┐    ┌────▼────┐   ┌──────────┐    ┌──────▼────┐   ┌───▼─────┐
│CPU│──▶│Features │──▶│ML Model  │──▶│Hysteresis │──▶│Frequency│
│Mon│    │Engineer │   │(Random   │    │State      │   │Actuator │
│   │    │         │   │ Forest)  │    │Machine    │   │         │
└───┘    └─────────┘   └──────────┘    └───────────┘   └─────────┘
                           │
                ┌──────────▼────────┐
                │  Energy Model     │
                │  E = f² + α·Δf·f  │
                └───────────────────┘
```

### Key Features Implemented

1. ✅ **Temporal Windowing** - 5-sample windows → 11 features
2. ✅ **Horizon Prediction** - Predicts 1 second ahead
3. ✅ **Random Forest Classifier** - 400 trees, 96.73% accuracy
4. ✅ **Probability-Aware DVFS** - Uses prediction confidence
5. ✅ **Hysteresis** - Reduces transitions by 68.1%
6. ✅ **Multi-Level Frequencies** - LOW/MID/HIGH (1520/2000/2400 MHz)
7. ✅ **Physics-Based Energy** - E = f² + α·|Δf|·f
8. ✅ **Cross-OS Analysis** - Windows vs Ubuntu validation

---

## 📊 Results Summary

### Synthetic Data (24 hours, 86,400 samples)

**Model Performance:**
```
Testing Accuracy:    96.73%
Precision (HIGH):    0.954
Recall (HIGH):       0.958
F1-Score (HIGH):     0.956
Inference Time:      0.87ms
```

**Energy Savings:**
```
Baseline Energy:     315,584,806,400
Smart-Watt Energy:   258,301,609,600
Energy Savings:      18.15%
Transition Reduction: 68.1% (18,440 → 5,878)
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd comparison/

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn, matplotlib; print('✅ Ready!')"
```

### Run Demo

```bash
# Option 1: Jupyter Notebook (Recommended)
jupyter notebook SmartWatt_DVFS_Synthetic_Data.ipynb

# Option 2: Python Script
python run_comparison.py --data data/synthetic_data.csv
```

---

## 📁 Repository Structure

```
comparison/
│
├── README.md                                    # This file
├── PROJECT_REPORT.md                            # Comprehensive report
├── requirements.txt                             # Dependencies
│
├── data/                                        # Datasets
│   ├── synthetic_data.csv                       # 24-hour synthetic data
│   ├── cpu_log_prepared.csv                     # Windows real-world
│   └── ubuntu_laptop_data.csv                   # Ubuntu real-world
│
├── models/                                      # Trained models
│   └── smartwatt_synthetic_model.pkl            # RF model (96.73%)
│
├── results/                                     # Outputs
│   ├── synthetic_data_results/
│   │   ├── 01_raw_data_analysis.png
│   │   ├── 02_temporal_features.png
│   │   ├── 03_model_performance.png
│   │   ├── 04_baseline_comparison.png
│   │   └── dvfs_comparison_results.csv
│   └── os_comparison.csv
│
├── smartwatt_features.py                        # Feature engineering
├── smartwatt_train.py                           # Model training
├── smartwatt_dvfs.py                            # DVFS governor
├── run_comparison.py                            # End-to-end pipeline
│
├── SmartWatt_DVFS_Synthetic_Data.ipynb         # Main notebook
└── Windows_vs_Ubuntu_SmartWatt_DVFS.ipynb      # Cross-OS analysis
```

---

## 🔬 Technical Implementation

### Feature Engineering

**Transform 1D time-series → 2D feature matrix:**

```python
Input:  [cpu[t-5], cpu[t-4], cpu[t-3], cpu[t-2], cpu[t-1]]

Output: 11 features
  ├── Raw values (5): cpu[t-5] ... cpu[t-1]
  ├── Deltas (4):     Δ₁, Δ₂, Δ₃, Δ₄
  └── Statistics (2): mean, std
```

**Feature Importance:**
- Mean (window average): **48.21%** 
- CPU_t-1 (most recent): **21.56%**
- Others: **30.23%**

### Horizon-Based Prediction

**Traditional (Reactive):** Predict current state  
**Smart-Watt (Predictive):** Predict average CPU 5 seconds ahead

```python
y[t] = 1 if mean(cpu[t:t+5]) > 30% else 0
# Advantage: Scale UP before load increases
```

### DVFS Governor

**Decision Logic:**
```python
if P(HIGH) > 0.85 and recent_cpu > 70:  → HIGH (2400 MHz)
elif P(HIGH) > 0.55:                    → MID  (2000 MHz)
else:                                   → LOW  (1520 MHz)
```

**Hysteresis:**
- HOLD_HIGH = 5 samples (maintain HIGH for 5 seconds)
- HOLD_LOW = 3 samples (maintain LOW for 3 seconds)
- Result: 68.1% fewer transitions

### Energy Model

```python
E(t) = [f(t)² + α·|Δf(t)|·f(t)] · (active_cores / total_cores)

Components:
  ├── f²:        Base power (CMOS law)
  ├── α·|Δf|·f:  Transition penalty (α = 0.5)
  └── Core scaling: Energy ∝ active cores
```

---

## 📈 Key Findings

### 1. Predictive DVFS Outperforms Reactive

- **Energy savings:** 18.15% vs baseline threshold governor
- **Fewer transitions:** 68.1% reduction (18,440 → 5,878)
- **Multi-level frequencies:** MID freq used 18.7% of time

### 2. Temporal Features Capture CPU Patterns

- Mean (window average) is 48% of importance
- Simple features > complex features
- 5-second lookback sufficient

### 3. Hysteresis Critical for Stability

- Without: 21.3% of samples involve transitions
- With: 6.8% of samples involve transitions
- No accuracy loss from hysteresis

### 4. Fixed Thresholds Don't Generalize

- Windows (11.65% avg CPU): 97.05% accuracy ✅
- Ubuntu (9.57% avg CPU): 49.53% accuracy ❌
- Solution: Adaptive percentile-based thresholds

---

## 🧪 Example Usage

### 1. Feature Engineering

```python
from smartwatt_features import build_temporal_features, build_horizon_labels
import numpy as np

# Sample CPU data
cpu = np.array([10, 15, 20, 35, 40, 30, 25, 20, 15, 10])

# Build features
X = build_temporal_features(cpu, window=5)
print(f"Features shape: {X.shape}")  # (5, 11)

# Build labels
y = build_horizon_labels(cpu, window=5, horizon=3, threshold=25)
print(f"Labels: {y}")  # [1, 0] (high then low)
```

### 2. Model Training

```python
from smartwatt_train import train_smartwatt_classifier
from sklearn.metrics import accuracy_score

# Train
model = train_smartwatt_classifier(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

# Save
import joblib
joblib.dump(model, 'my_model.pkl')
```

### 3. DVFS Simulation

```python
from smartwatt_dvfs import simulate_smartwatt_dvfs
import pandas as pd
import joblib

# Load data and model
df = pd.read_csv('data/synthetic_data.csv')
model = joblib.load('models/smartwatt_synthetic_model.pkl')

# Simulate
results = simulate_smartwatt_dvfs(df, model)

print(f"Total Energy: {results['total_energy']:,.0f}")
print(f"Transitions: {results['transitions']}")
print(f"Frequency Usage: {results['freq_usage']}")
```

---

## 🔍 Troubleshooting

### Low Model Accuracy (<70%)

**Possible Causes:**
1. Threshold too high/low for workload
2. Insufficient training data
3. Class imbalance

**Solutions:**
```python
# Check class distribution
print(f"HIGH: {y.sum() / len(y) * 100:.1f}%")

# Use adaptive threshold
threshold = np.percentile(cpu_values, 60)

# Ensure >1000 samples per class
```

### Excessive Transitions

**Solution:**
```python
# Increase hold times
governor = SmartWattGovernor(
    hold_high=10,  # Increase from 5
    hold_low=7     # Increase from 3
)
```

### Model Predicts Only One Class

**Root Cause:** Threshold inappropriate for workload

**Solution:**
```python
# Analyze CPU distribution
print(f"25th percentile: {np.percentile(cpu, 25):.1f}%")
print(f"50th percentile: {np.percentile(cpu, 50):.1f}%")
print(f"75th percentile: {np.percentile(cpu, 75):.1f}%")

# Choose threshold between 50th and 75th
threshold = np.percentile(cpu, 60)
```

---

## 📚 References

### Academic Papers

1. Dhiman, G., & Rosing, T. S. (2009). "Dynamic voltage frequency scaling for multi-tasking systems using online learning". *IEEE ISLPED*.

2. Le Sueur, E., & Heiser, G. (2010). "Dynamic voltage and frequency scaling: The laws of diminishing returns". *HotPower*.

3. Karanth, V. (2020). "Smart-Watt DVFS Framework". DVFS_F Repository.

### Documentation

- Linux Kernel DVFS: https://www.kernel.org/doc/html/latest/admin-guide/pm/cpufreq.html
- Intel DVFS: https://www.intel.com/content/www/us/en/developer/articles/technical/power-management-states-p-states-c-states-and-package-c-states.html
- Scikit-learn: https://scikit-learn.org/stable/modules/ensemble.html#forest

---

## 🌟 Future Work

1. **Adaptive Threshold Selection** - Percentile-based auto-calibration
2. **Online Learning** - Incremental model updates
3. **Deep Learning** - LSTM for sequence modeling
4. **Real-Time Deployment** - Linux kernel integration
5. **Hardware Validation** - Power meter measurements

