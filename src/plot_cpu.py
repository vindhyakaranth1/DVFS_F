import pandas as pd
import matplotlib.pyplot as plt
import os

# Locate data safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "cpu_log.csv")

# Load data
df = pd.read_csv(DATA_PATH)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ------------------ PLOTS ------------------

plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["cpu_util"])
plt.title("CPU Utilization Over Time")
plt.ylabel("CPU Utilization (%)")
plt.xlabel("Time")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["cpu_freq_mhz"])
plt.title("CPU Frequency Over Time")
plt.ylabel("Frequency (MHz)")
plt.xlabel("Time")
plt.grid(True)
plt.tight_layout()
plt.show()
