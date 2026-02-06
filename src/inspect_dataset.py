import pandas as pd
import os

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cpu_log_fixed.csv")


# -------- Load dataset --------
df = pd.read_csv(DATA_PATH)

print("\n===== DATASET OVERVIEW =====")
print("Total samples:", len(df))
print("Columns:", list(df.columns))

print("\n===== SAMPLE ROWS =====")
print(df.head())

print("\n===== SESSION DISTRIBUTION =====")
print(df["session"].value_counts())

print("\n===== CPU UTIL STATS =====")
print(df["cpu_util"].describe())

print("\n===== CPU FREQ STATS =====")
print(df["cpu_freq_mhz"].describe())
