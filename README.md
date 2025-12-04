# Machine Learning–Based Predictive CPU Scheduling Simulator
### A Machine Learning Approach to Shortest Job First (SJF) — now upgraded with Aging

## 📖 Overview
This project demonstrates how a lightweight Machine Learning model can be used to make Shortest Job First (SJF) scheduling practical by predicting future CPU burst times. The repository now includes an upgraded scheduler that combines AI predictions with an "Aging" protection mechanism to avoid starvation and improve fairness.

## 🚀 Key Features (Updated)
- **Predictive Scheduling:** Uses a trained Linear Regression model to predict next CPU bursts from a sliding window of past bursts.
- **Aging Protection:** A configurable aging factor reduces starvation by boosting priority of processes that have waited longer.
- **Comparative Simulation:** Runs side-by-side comparisons between Standard FCFS and the AI-driven SJF with Aging.
- **Visual Output:** Produces a Gantt chart (`gantt_chart.png`) for visual analysis of the schedule.
- **Lightweight & Fast:** Prediction runs in O(1); suitable for fast scheduler decision-making.

---

## ⚙️ Project Components
Files in this folder and their roles:

- `generate_data.py` — Synthetic data generator. Produces `cpu_burst_dataset.csv` where each row contains the previous 3 bursts (features) and the next burst (target).
- `train_model.py` — Trains a Linear Regression model on the generated data and saves the trained predictor to `burst_predictor.pkl` (uses `joblib`).
- `main_scheduler.py` — Original simulator demonstrating the predictive SJF approach (keeps the simple sliding-window prediction + scheduling loop).
- `main_scheduler_upgraded.py` — Upgraded simulator that loads `burst_predictor.pkl`, simulates process "age" values, combines AI predictions with an Aging mechanism, compares against FCFS, and saves a Gantt chart (`gantt_chart.png`).
- `cpu_burst_dataset.csv` — Example output dataset from `generate_data.py` (or the dataset used for training).

---

## 🔬 Upgraded Scheduler Logic (`main_scheduler_upgraded.py`)

Summary of the updated behavior and configurable options:

- `NUM_PROCESSES` — Number of simulated processes in a run (default: `6`).
- `AGING_FACTOR` — Controls how strongly waiting time (age) reduces a process's priority score. A larger value gives stronger starvation protection.

Process initialization details:
- Each `Process` has a `pid`, `base_burst` (determined by parity: even PIDs are "Heavy" with a larger base burst, odd PIDs are "Light"), a `history` list initialized to the base burst, and an `actual_burst_now` which contains small randomness.
- Each process is also assigned a simulated `age` (a randomly generated wait time) to emulate processes that have been waiting in the ready queue prior to the simulation.

Scheduling & priority calculation:
- `get_ai_prediction(process)` loads the trained `burst_predictor.pkl` model (via `joblib`) and predicts the next burst from `process.history`.
- The scheduler computes a `priority_score` for each process:

    priority_score = predicted_burst - (age * AGING_FACTOR)

    Lower `priority_score` means higher scheduling priority. The subtraction gives waiting processes a boost (aging) so they don't starve even if their predicted burst is large.

Simulation outcomes:
- The script runs two simulations on identical process sets: Standard FCFS and AI-driven SJF with Aging. It prints average waiting times and percent improvement (if any).
- A Gantt chart of the AI+aging schedule is saved as `gantt_chart.png` for visual inspection.

Notes:
- The upgraded scheduler expects `burst_predictor.pkl` to exist in the working directory. If it's missing, `main_scheduler_upgraded.py` will print an error and exit.

---

## 🛠️ Requirements
- Python 3.12+
- Python packages: `scikit-learn`, `pandas`, `joblib`, `matplotlib`

Install dependencies (PowerShell example):

```
pip install scikit-learn pandas joblib matplotlib
```

---

## ▶️ How to run

1. Generate synthetic training data (if you need fresh data):

```
python generate_data.py
```

2. Train the model (creates `burst_predictor.pkl`):

```
python train_model.py
```

3. Run the upgraded scheduler (compares FCFS vs AI+AGING and writes `gantt_chart.png`):

```
python main_scheduler_upgraded.py
```

Expected outputs:
- `cpu_burst_dataset.csv` — dataset produced by `generate_data.py` (if run).
- `burst_predictor.pkl` — trained model produced by `train_model.py`.
- `gantt_chart.png` — saved visualization from `main_scheduler_upgraded.py`.
- Console output comparing average wait times between Standard FCFS and AI+AGING.

---

## 📌 Design Rationale

- Linear Regression is chosen to keep inference extremely fast (O(1)), which matters for scheduler decision latency.
- Aging is added to prevent starvation: pure SJF (even predictive) can starve long processes when short ones keep arriving. The linear aging term trades a small amount of average wait time for fairness.

## ✅ Next steps / Suggestions
- Tune `AGING_FACTOR` and `NUM_PROCESSES` in `main_scheduler_upgraded.py` to experiment with different fairness/throughput trade-offs.
- Hook process arrival times and dynamic updating of `history` during multi-round simulations if you want continuous, online scheduling evaluation.

---

If you'd like, I can:
- add a `requirements.txt` with the exact packages,
- add a small `run_demo.ps1` PowerShell script to run all steps in sequence,
- or extend the simulator to run multiple scheduling rounds and log per-round stats.

Enjoy experimenting with predictive scheduling!

