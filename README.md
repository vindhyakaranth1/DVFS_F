# Machine Learning–Based Predictive CPU Scheduling Simulator
### ML-enabled SJF & SRTF using Google Borg trace training dataset

## **Overview**
This repository demonstrates a practical bridge between the theoretical optimality of Shortest Job First (SJF) scheduling and real-world kernels by using a lightweight Machine Learning model to predict future CPU bursts. With accurate predictions we implement:
- Non-Preemptive AI-SJF with an Aging mechanism to prevent starvation.
- Preemptive AI-SRTF (Shortest Remaining Time First) that context-switches in real time when a shorter job arrives.

The model is trained from Google Borg cluster traces (via the included Colab pipeline) and used by the simulator to make scheduling decisions with O(1) inference latency.



## **Key Features**
- **Google-trained brain:** Linear Regression model trained from processed Borg trace data (`ColabNtbk.ipynb`) and saved as `burst_predictor.pkl`.
- **Dual-mode scheduling:** `preemptive_scheduling.py` implements both Non-Preemptive AI-SJF (with Aging) and Preemptive AI-SRTF for apples-to-apples comparison.
- **Convoy mitigation:** Preemptive SRTF reduces convoy effect by interrupting long tasks when short ones arrive.
- **Comparative arena:** Runs Standard FCFS vs Non-Preemptive AI-SJF vs Preemptive AI-SRTF on identical workloads.
- **Visualization:** Saves `scheduling_comparison.png` illustrating both modes side-by-side.


## **Files**
- `ColabNtbk.ipynb`: Data engineering pipeline that processes Google Borg traces, extracts burst durations, normalizes them (log + scaling), creates sliding-window training sequences ([T-3,T-2,T-1] -> T), trains a `LinearRegression` model, and saves `burst_predictor.pkl` and `model_metadata.txt`.
- `preemptive_scheduling.py`: Primary simulator. Loads `burst_predictor.pkl`, generates synthetic cloud-like processes, runs FCFS, Non-Preemptive AI (SJF+Aging), and Preemptive AI (SRTF), prints average waiting times, and saves `scheduling_comparison.png`.
- `train_model.py`: (If present) alternative local script to train the same linear model on `cpu_burst_dataset.csv` or generated data.
- `generate_data.py`: (If present) creates `cpu_burst_dataset.csv` used for training.



## **Data & Model (Colab pipeline)**

 - **Source:** Google Borg Cluster trace (instance events). The notebook extracts activity segments and computes durations. The CSV data used in the Colab pipeline for this project was taken from the Kaggle sample: https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample
- **Transform:** `np.log1p` is applied to durations to compress variance; durations are then normalized to a 1–100 range (simulating milliseconds for the simulator).
- **Training:** Sliding windows of size 3 are used as inputs to a `LinearRegression` model. The notebook saves `burst_predictor.pkl` (via `joblib`) and `model_metadata.txt`.



## **Simulator behavior (from `preemptive_scheduling.py`)**

- **Process generation**
  - `NUM_PROCESSES` default: `6`.
  - `arrival_time`: random integer in `[0, 20]` (ms) to model stochastic arrivals.
  - `base_burst` distribution (cloud-like):
    - 40% short tasks: uniform 1–15
    - 40% medium tasks: uniform 20–60
    - 20% long tasks: uniform 70–95
  - `history`: initialized with 3 samples close to `base_burst` (simulating burst stability).
  - `actual_burst`: `base_burst` ± noise; `remaining_time` initialized from `actual_burst`.

- **AI prediction**
  - `get_ai_prediction(process)` loads `burst_predictor.pkl` and predicts next burst from `process.history`. Predictions are clamped to at least 1 ms.

- **Non-Preemptive AI (SJF + Aging)**
  - At each scheduling decision, available processes (arrived, not completed) are scored:
    - `score = predicted_burst - (wait_duration * AGING_FACTOR)`
  - The process with the lowest `score` runs to completion (non-preemptive). `AGING_FACTOR` prevents starvation by reducing score with wait time.

- **Preemptive AI (SRTF)**
  - Tick-by-tick (1 ms resolution): ready queue is re-evaluated each ms.
  - Decision uses predicted bursts (and `remaining_time` for an already-started process). If a newcomer has a shorter predicted remaining time than the currently running process, a context switch occurs.

- **FCFS**
  - Standard first-come-first-served baseline (sorted by `arrival_time`).

- **Visuals & outputs**
  - `scheduling_comparison.png` — two stacked plots: Non-Preemptive AI (top) and Preemptive AI (bottom). Saved by `preemptive_scheduling.py`.
  - Console prints showing per-process arrival/prediction/actual and average waiting times for each algorithm.



## **Requirements**
- Python 3.12+
- Packages: `scikit-learn`, `pandas`, `joblib`, `matplotlib`

Install (PowerShell):

```
pip install scikit-learn pandas joblib matplotlib
```


## **Observed Results (example)**
From runs of the simulator we commonly observe:
- Standard FCFS: higher average waiting time (convoy effect present).
- Non-Preemptive AI (SJF + Aging): small improvement vs FCFS with better fairness.
- Preemptive AI (SRTF): substantial improvement in average waiting time and responsiveness; interrupts long tasks when short tasks arrive.

Example numbers (will vary by random seed and generated processes):

- FCFS: ~150 ms average wait
- Non-Preemptive AI: ~145 ms average wait
- Preemptive AI (SRTF): ~116 ms average wait


## **Design Notes & Rationale**
- `LinearRegression` is used because it provides constant-time inference (O(1)) and negligible overhead for scheduling decisions.
- Aging is implemented as a linear subtraction term to the predicted burst, preventing starvation without much added complexity.

