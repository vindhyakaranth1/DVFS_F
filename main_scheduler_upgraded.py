import joblib
import random
import copy
import matplotlib.pyplot as plt
import warnings

# Silence warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
NUM_PROCESSES = 6
AGING_FACTOR = 2.0  # Higher number = Stronger protection against starvation

# Load Model
try:
    model = joblib.load('burst_predictor.pkl')
    print(" [System] AI Model loaded successfully.")
except FileNotFoundError:
    print(" [Error] Model not found! Run train_model.py first.")
    exit()

class Process:
    def __init__(self, pid):
        self.pid = pid
        self.arrival_time = 0
        
        # 1. SETUP PROCESS TYPE
        if pid % 2 == 0: 
            self.base_burst = 100 # Heavy
        else:            
            self.base_burst = 5   # Light
            
        self.history = [self.base_burst, self.base_burst, self.base_burst]
        
        # 2. GENERATE ACTUAL BURST
        self.actual_burst_now = max(1, self.base_burst + random.randint(-5, 5))
        
        # 3. SIMULATE "AGE" (For the Aging Mechanism)
        # We give each process a random "Wait Time" to simulate that 
        # some have been sitting in the queue longer than others.
        self.age = random.randint(0, 100) 
        
        self.predicted_burst = 0
        self.start_time = 0
        self.completion_time = 0
        self.waiting_time = 0

def get_ai_prediction(process):
    input_data = [process.history] 
    prediction = model.predict(input_data)
    return max(1, prediction[0])

def run_simulation(process_list, algorithm_name):
    current_time = 0
    total_waiting_time = 0
    print(f"\n--- Running: {algorithm_name} ---")
    
    for p in process_list:
        p.start_time = current_time
        
        # Total wait = Time spent in queue + Initial "Age" before simulation started
        p.waiting_time = (current_time - p.arrival_time) + p.age
        total_waiting_time += p.waiting_time
        
        current_time += p.actual_burst_now
        p.completion_time = current_time
        
        print(f" [Time {p.start_time:3}] P{p.pid} (Burst: {p.actual_burst_now}ms, Age: {p.age}ms) started.")

    avg_wait = total_waiting_time / len(process_list)
    return avg_wait

def plot_gantt_chart(process_list, title):
    fig, gnt = plt.subplots(figsize=(10, 5))
    gnt.set_ylim(0, 50)
    gnt.set_xlim(0, max(p.completion_time for p in process_list) + 10)
    gnt.set_xlabel('Time (ms)')
    gnt.set_ylabel('Process Stream')
    gnt.set_yticks([]) 
    gnt.grid(True, axis='x')

    for p in process_list:
        color = 'tab:red' if p.actual_burst_now > 50 else 'tab:green'
        gnt.broken_barh([(p.start_time, p.actual_burst_now)], (15, 20), facecolors=(color))
        center_x = p.start_time + (p.actual_burst_now / 2)
        gnt.text(center_x, 25, f"P{p.pid}", ha='center', color='white', fontweight='bold')

    plt.title(title)
    plt.savefig("gantt_chart.png")
    print(f"\n [Graph] Gantt Chart saved as 'gantt_chart.png'")

def main():
    print(" [System] Generating Processes with Random Ages...")
    original_processes = [Process(i) for i in range(1, NUM_PROCESSES + 1)]

    # Create Identical Copies for Fairness
    processes_fcfs = copy.deepcopy(original_processes)
    processes_ai   = copy.deepcopy(original_processes)

    # --- SIMULATION 1: STANDARD FCFS ---
    # Sort by PID (Arrival Order)
    processes_fcfs.sort(key=lambda x: x.pid) 
    avg_wait_fcfs = run_simulation(processes_fcfs, "Standard FCFS")

    # --- SIMULATION 2: AI + AGING ---
    print(f"\n [AI] Calculating Priorities (Prediction - Age * Factor)...")
    
    for p in processes_ai:
        p.predicted_burst = get_ai_prediction(p)
        
        # --- THE AGING LOGIC IS HERE ---
        # Priority Score = Predicted_Burst - (Age * AGING_FACTOR)
        # Lower Score = Higher Priority
        # If Age is high, Score becomes small (or negative), moving it to front!
        p.priority_score = p.predicted_burst - (p.age * AGING_FACTOR)

    # Sort by the new "Aging" Score
    processes_ai.sort(key=lambda x: x.priority_score)
    
    avg_wait_ai = run_simulation(processes_ai, "AI-SJF with Aging")

    # --- RESULTS ---
    print("\n" + "="*40)
    print("       FINAL COMPARATIVE RESULTS       ")
    print("="*40)
    print(f" Standard FCFS Avg Wait: {avg_wait_fcfs:.2f} ms")
    print(f" AI + Aging Avg Wait:    {avg_wait_ai:.2f} ms")
    
    if avg_wait_ai < avg_wait_fcfs:
        improvement = ((avg_wait_fcfs - avg_wait_ai) / avg_wait_fcfs) * 100
        print(f" Performance Improvement: {improvement:.1f}% 🚀")
    else:
        print(" Note: Aging might increase Wait Time slightly to ensure Fairness.")

    # Generate Graph
    plot_gantt_chart(processes_ai, "AI Scheduler (With Aging Protection)")

if __name__ == "__main__":
    main()