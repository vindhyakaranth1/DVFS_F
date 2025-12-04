# Machine Learning–Based Predictive CPU Scheduling Simulator
### A Machine Learning Approach to Shortest Job First (SJF) Scheduling

## 📖 Overview
This project addresses a fundamental problem in Operating Systems: **The Infeasibility of Shortest Job First (SJF).**

While SJF is the mathematically optimal scheduling algorithm for minimizing average waiting time, it cannot be implemented in standard systems because the OS does not know the future CPU burst time of a process. This project solves this limitation using **Machine Learning (Linear Regression)** to accurately predict future burst times based on historical execution patterns.

## 🚀 Key Features
* **Predictive Scheduling:** Uses AI to forecast CPU burst times.
* **Dynamic Adaptation:** Implements a "Sliding Window" mechanism to adapt to changing process behaviors in real-time.
* **Pattern Recognition:** Distinguishes between **CPU-Bound** (Heavy) and **I/O-Bound** (Light) processes.
* **Lightweight Implementation:** Built with Scikit-Learn and Python, optimized for low-latency decision-making.

---

## ⚙️ Technical Implementation

The project is divided into three distinct phases:

### Phase 1: Synthetic Data Generation (`generate_data.py`)
Since real-world CPU traces are often strictly proprietary or too large for mini-projects, we developed a **Patterned Synthetic Generator**.
* **Logic:** The script simulates process "stickiness." If a process had a long burst previously, it is statistically likely to have a long burst again.
* **Profiles:** It generates two distinct process profiles:
    * *Type A (CPU Bound):* Bursts between 50ms–150ms (e.g., Video Rendering).
    * *Type B (I/O Bound):* Bursts between 1ms–10ms (e.g., Text Editors).
* **Output:** `cpu_burst_dataset.csv` (Features: Previous 3 Bursts | Target: Next Burst).

### Phase 2: Model Training (`train_model.py`)
We utilize **Linear Regression** for the prediction engine.
* **Why Linear Regression?** In OS scheduling, decision speed is critical. Deep Learning models are too slow (high inference latency). Linear Regression offers $O(1)$ complexity for predictions, making it ideal for real-time systems.
* **Input Features:** A sliding window of the last 3 CPU burst times $[t_{n-3}, t_{n-2}, t_{n-1}]$.
* **Target:** The next CPU burst time $t_n$.
* **Artifact:** The trained model is serialized and saved as `burst_predictor.pkl`.

### Phase 3: The Simulator (`main_scheduler.py`)
This script acts as the Operating System Kernel.
1.  **Process Control Block (PCB):** Each process maintains a `history` list (a sliding window of its last 3 bursts).
2.  **The Scheduler Loop:**
    * **Prediction:** Before execution, the Scheduler feeds the process `history` into the loaded `.pkl` model.
    * **Sorting:** The Ready Queue is sorted based on these *predicted* times (SJF Logic).
    * **Execution:** The process runs for a simulated "Actual Time" (generated with random noise).
    * **Dynamic Update:** The "Actual Time" is pushed into the process's `history`, removing the oldest entry. This allows the AI to adapt immediately if a process changes behavior.

---

## 🛠️ Tech Stack
* **Language:** Python 3.12+
* **Machine Learning:** Scikit-Learn
* **Data Handling:** Pandas
* **Serialization:** Joblib

