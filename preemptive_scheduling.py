import joblib
import random
import copy
import matplotlib.pyplot as plt
import warnings

# Silence warnings from Scikit-Learn
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
NUM_PROCESSES = 6
AGING_FACTOR = 1.5  # Prevents starvation in Non-Preemptive mode

# Load the AI Brain
try:
    model = joblib.load('burst_predictor.pkl')
    print(" [System] AI Model loaded successfully.")
except FileNotFoundError:
    print(" [Error] Model not found! Run train_model.py first.")
    exit()

class Process:
    def __init__(self, pid):
        self.pid = pid
        # To make Preemption interesting, processes must arrive at different times
        self.arrival_time = random.randint(0, 15) 
        
        # Determine Process Type (Heavy vs Light)
        if pid % 2 == 0: 
            self.base_burst = 50  # CPU Heavy
        else:            
            self.base_burst = 5   # I/O Light
            
        self.history = [self.base_burst, self.base_burst, self.base_burst]
        
        # The 'Truth' (Unknown to OS, used for simulation)
        self.actual_burst = max(1, self.base_burst + random.randint(-5, 5))
        self.remaining_time = self.actual_burst # For Preemptive logic
        
        # The 'Guess' (What AI gives us)
        self.predicted_burst = 0
        
        # Metrics
        self.start_time = -1
        self.completion_time = 0
        self.waiting_time = 0
        self.is_completed = False

def get_ai_prediction(process):
    prediction = model.predict([process.history])
    return max(1, prediction[0])

# ==========================================
# ALGORITHM 1: NON-PREEMPTIVE (AI-SJF + Aging)
# ==========================================
def run_non_preemptive(process_list):
    current_time = 0
    completed = 0
    n = len(process_list)
    print("\n--- Mode: Non-Preemptive AI (SJF + Aging) ---")

    # While processes remain...
    while completed < n:
        # 1. Who is here? (Arrival Time <= Current Time)
        available = [p for p in process_list if p.arrival_time <= current_time and not p.is_completed]
        
        if available:
            # 2. CALCULATE SCORES (Prediction - Aging)
            # We want the lowest score (Shortest Job)
            for p in available:
                wait_duration = current_time - p.arrival_time
                p.score = p.predicted_burst - (wait_duration * AGING_FACTOR)
            
            # 3. PICK WINNER (Lowest Score)
            p = min(available, key=lambda x: x.score)
            
            # 4. EXECUTE FULLY (Non-Preemptive means we don't stop)
            if p.start_time == -1: p.start_time = current_time
            
            print(f" [Time {current_time:3}] Starting P{p.pid} (Pred: {p.predicted_burst:.1f}ms)... runs for {p.actual_burst}ms")
            
            current_time += p.actual_burst
            p.completion_time = current_time
            p.waiting_time = p.start_time - p.arrival_time
            p.is_completed = True
            completed += 1
        else:
            # CPU Idle
            current_time += 1
            
    return sum(p.waiting_time for p in process_list) / n

# ==========================================
# ALGORITHM 2: PREEMPTIVE (AI-SRTF)
# ==========================================
def run_preemptive(process_list):
    current_time = 0
    completed = 0
    n = len(process_list)
    print("\n--- Mode: Preemptive AI (SRTF) ---")
    
    last_pid = -1
    
    # Run tick-by-tick
    while completed < n:
        # 1. Who is here?
        available = [p for p in process_list if p.arrival_time <= current_time and not p.is_completed]
        
        if available:
            # 2. PICK WINNER (Shortest PREDICTED Remaining Time)
            # Logic: Predicted_Total - (Actual_Total - Actual_Remaining)
            # Simplified: Just trust Predicted Burst for the decision
            shortest = min(available, key=lambda x: x.predicted_burst if x.start_time == -1 else x.remaining_time)
            
            # 3. CONTEXT SWITCH CHECK
            if shortest.pid != last_pid:
                print(f" [Time {current_time:3}] Switch -> P{shortest.pid} (Rem: {shortest.remaining_time})")
                if shortest.start_time == -1: shortest.start_time = current_time
                last_pid = shortest.pid
            
            # 4. EXECUTE FOR 1 TICK
            shortest.remaining_time -= 1
            current_time += 1
            
            # 5. CHECK COMPLETION
            if shortest.remaining_time == 0:
                shortest.is_completed = True
                shortest.completion_time = current_time
                # Wait = Turnaround - Burst
                shortest.waiting_time = (shortest.completion_time - shortest.arrival_time) - shortest.actual_burst
                completed += 1
        else:
            current_time += 1
            
    return sum(p.waiting_time for p in process_list) / n

# ==========================================
# HELPER: FCFS (For Comparison)
# ==========================================
def run_fcfs(process_list):
    # Sort purely by Arrival Time
    process_list.sort(key=lambda x: x.arrival_time)
    current_time = 0
    
    for p in process_list:
        if current_time < p.arrival_time:
            current_time = p.arrival_time
        
        p.start_time = current_time
        p.waiting_time = p.start_time - p.arrival_time
        current_time += p.actual_burst
        p.completion_time = current_time
        
    return sum(p.waiting_time for p in process_list) / len(process_list)

# ==========================================
# PLOTTING
# ==========================================
def plot_comparison(p_non, p_pre):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Non-Preemptive Chart
    ax1.set_title("Non-Preemptive AI (SJF)")
    ax1.set_xlim(0, max(p.completion_time for p in p_non) + 10)
    ax1.set_ylim(0, 40)
    ax1.set_yticks([])
    for p in p_non:
        color = 'tab:red' if p.actual_burst > 30 else 'tab:green'
        ax1.broken_barh([(p.start_time, p.actual_burst)], (10, 20), facecolors=color)
        ax1.text(p.start_time + 1, 15, f"P{p.pid}", color='white')

    # Preemptive Chart
    ax2.set_title("Preemptive AI (SRTF)")
    ax2.set_xlim(0, max(p.completion_time for p in p_pre) + 10)
    ax2.set_ylim(0, 40)
    ax2.set_yticks([])
    
    # For Preemptive, we can't just draw one bar per process. 
    # We ideally track every start/stop, but for the graph we'll just show the ranges.
    # (Simplified visualization for Preemptive)
    for p in p_pre:
        color = 'tab:red' if p.actual_burst > 30 else 'tab:green'
        # Draw from start to finish (shows fragmentation conceptually)
        ax2.broken_barh([(p.start_time, p.completion_time - p.start_time)], (10, 20), facecolors=color, alpha=0.5)
        ax2.text(p.start_time, 15, f"P{p.pid}", color='black')

    plt.tight_layout()
    plt.savefig("scheduling_comparison.png")
    print("\n [Graph] Saved comparison graph to 'scheduling_comparison.png'")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print(" [System] Generating Processes with Random Arrivals...")
    procs = [Process(i) for i in range(1, NUM_PROCESSES + 1)]
    
    # Predict for everyone first (The AI Step)
    for p in procs:
        p.predicted_burst = get_ai_prediction(p)
        print(f"   P{p.pid}: Arrives {p.arrival_time}, AI Pred: {p.predicted_burst:.1f}, Actual: {p.actual_burst}")

    # Create clones for fair testing
    list_fcfs = copy.deepcopy(procs)
    list_non  = copy.deepcopy(procs)
    list_pre  = copy.deepcopy(procs)

    # --- RUN SIMULATIONS ---
    avg_fcfs = run_fcfs(list_fcfs)
    avg_non  = run_non_preemptive(list_non)
    avg_pre  = run_preemptive(list_pre)

    # --- RESULTS TABLE ---
    print("\n" + "="*45)
    print(f"{'ALGORITHM':<20} | {'AVG WAIT TIME':<15}")
    print("-" * 45)
    print(f"{'Standard FCFS':<20} | {avg_fcfs:.2f} ms")
    print(f"{'Non-Preemptive AI':<20} | {avg_non:.2f} ms")
    print(f"{'Preemptive AI':<20} | {avg_pre:.2f} ms")
    print("="*45)
    
    # Generate Graph
    plot_comparison(list_non, list_pre)

if __name__ == "__main__":
    main()