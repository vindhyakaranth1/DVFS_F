import joblib
import pandas as pd
import random
import time

# --- CONFIGURATION ---
NUM_PROCESSES = 5
# Load the trained AI model
try:
    model = joblib.load('burst_predictor.pkl')
    print(" [System] AI Model loaded successfully.")
except FileNotFoundError:
    print(" [Error] Model not found! Run train_model.py first.")
    exit()

class Process:
    def __init__(self, pid):
        self.pid = pid
        self.arrival_time = 0  # For simplicity, all arrive at 0
        
        # 1. INITIALIZATION:
        # Real processes have some past behavior. We simulate this.
        # We give them a "pattern" so we can see if AI picks it up.
        if pid % 2 == 0: # Even PIDs are "CPU Heavy"
            self.base_burst = 100 
        else:            # Odd PIDs are "I/O Light"
            self.base_burst = 5
            
        # Initialize history with 3 values close to their base burst
        self.history = [self.base_burst, self.base_burst, self.base_burst]
        
        self.predicted_burst = 0
        self.actual_burst_now = 0
        self.waiting_time = 0

    def generate_current_burst(self):
        """Simulates the process actually running on CPU"""
        # Add some randomness (-5 to +5) to the base burst
        return max(1, self.base_burst + random.randint(-5, 5))

    def update_history(self, actual_time):
        """THE DYNAMIC LEARNING STEP"""
        # Sliding Window: Remove oldest, add newest
        self.history.pop(0)
        self.history.append(actual_time)

def get_ai_prediction(process):
    """Ask the AI: 'Based on history, how long will this take?'"""
    # Reshape data because model expects a 2D array
    input_data = [process.history] 
    prediction = model.predict(input_data)
    return max(1, prediction[0]) # Prediction can't be negative

def main():
    # 1. Create Processes
    print(f"\n [System] Creating {NUM_PROCESSES} processes...")
    ready_queue = [Process(i) for i in range(1, NUM_PROCESSES + 1)]
    
    current_time = 0
    total_waiting_time = 0
    
    print(f"\n{'PID':<5} | {'History (Inputs)':<20} | {'AI Prediction':<15} | {'Actual Burst'}")
    print("-" * 65)

    # --- SIMULATION LOOP ---
    # In a real OS, this happens continuously. 
    # Here, we simulate one round of scheduling.

    # STEP A: AI PREDICTS BURST TIMES
    for p in ready_queue:
        p.predicted_burst = get_ai_prediction(p)
        # We also generate the "Real" time now to show comparison
        p.actual_burst_now = p.generate_current_burst() 
        
        print(f"{p.pid:<5} | {str(p.history):<20} | {p.predicted_burst:.2f} ms{'':<8} | {p.actual_burst_now} ms")

    # STEP B: SCHEDULER SORTS QUEUE (SJF LOGIC)
    # This is the "Magic". We sort by PREDICTED time, not actual (since actual is unknown until run)
    print("\n [Scheduler] Sorting Queue based on AI Predictions (SJF)...")

# Define an Aging Factor (Sensitivity)
    AGING_FACTOR = 0.5 

# Update the sort logic
# The longer it waits, the smaller its 'rank' value becomes, so it moves to the front.
    ready_queue.sort(key=lambda x: x.predicted_burst - (x.waiting_time * AGING_FACTOR))

    # STEP C: EXECUTION
    print("\n--- EXECUTION ORDER (GANTT CHART) ---")
    
    for p in ready_queue:
        # Calculate Waiting Time
        p.waiting_time = current_time - p.arrival_time
        total_waiting_time += p.waiting_time
        
        # Simulate Execution
        print(f" [Time {current_time:3}] Running Process {p.pid}...", end="")
        
        # In a real GUI, you'd update a progress bar here
        # We just print the result
        print(f" Done. (Actual: {p.actual_burst_now}ms, Predicted: {p.predicted_burst:.2f}ms)")
        
        # Update System Clock
        current_time += p.actual_burst_now
        
        # STEP D: DYNAMIC UPDATE
        # The process finished. We update its history for next time.
        p.update_history(p.actual_burst_now)

    # --- METRICS ---
    avg_wait = total_waiting_time / NUM_PROCESSES
    print("-" * 40)
    print(f"Total Execution Time: {current_time} ms")
    print(f"Average Waiting Time: {avg_wait:.2f} ms")
    print("-" * 40)
    
    # Validation Logic
    print("\n [Analysis]")
    print(" Did the AI correctly prioritize Short Jobs?")
    first_process = ready_queue[0]
    last_process = ready_queue[-1]
    
    if first_process.actual_burst_now < last_process.actual_burst_now:
        print(" SUCCESS: The shortest job ran first!")
    else:
        print(" NOTE: AI made a slight error (acceptable in ML).")

if __name__ == "__main__":
    main()