"""
Smart-Watt DVFS Simulator
Adapted from vindhya/DVFS_F repository

Simulates Dynamic Voltage and Frequency Scaling (DVFS) with:
- Probability-aware decisions
- Hysteresis (frequency hold)
- Multi-level frequencies
- Physics-based energy modeling
"""

import numpy as np
import pandas as pd


class SmartWattDVFS:
    """
    Smart-Watt DVFS Governor.
    
    Features:
    - Predictive frequency scaling using ML probabilities
    - Hysteresis to prevent frequency oscillation
    - Multi-level frequencies (LOW, MID, HIGH)
    - Physics-based energy calculation
    """
    
    def __init__(self, low_freq=1520, mid_freq=2000, high_freq=2400,
                 hold_high=5, hold_low=3, window_cpu=5, alpha=0.5):
        """
        Initialize Smart-Watt DVFS governor.
        
        Args:
            low_freq: LOW frequency in MHz (default: 1520)
            mid_freq: MID frequency in MHz (default: 2000)
            high_freq: HIGH frequency in MHz (default: 2400)
            hold_high: Samples to hold HIGH frequency (default: 5)
            hold_low: Samples to hold LOW frequency (default: 3)
            window_cpu: CPU averaging window size (default: 5)
            alpha: Transition penalty coefficient (default: 0.5)
        """
        self.LOW_FREQ = low_freq
        self.MID_FREQ = mid_freq
        self.HIGH_FREQ = high_freq
        self.HOLD_HIGH = hold_high
        self.HOLD_LOW = hold_low
        self.WINDOW_CPU = window_cpu
        self.ALPHA = alpha
        self.LOGICAL_CORES = 8
        
        # State variables
        self.current_freq = None
        self.hold_counter = 0
        self.cpu_window = []
        
    def reset(self):
        """Reset governor state."""
        self.current_freq = None
        self.hold_counter = 0
        self.cpu_window = []
    
    def decide_frequency(self, cpu_util, prediction_prob):
        """
        Decide target frequency based on CPU and prediction probability.
        
        Args:
            cpu_util: Current CPU utilization (0-1 scale)
            prediction_prob: ML prediction probability for HIGH freq
        
        Returns:
            target_freq: Recommended frequency in MHz
        """
        # Update CPU window
        self.cpu_window.append(cpu_util)
        if len(self.cpu_window) > self.WINDOW_CPU:
            self.cpu_window.pop(0)
        
        recent_cpu_mean = sum(self.cpu_window) / len(self.cpu_window)
        
        # Decision logic with probability awareness
        if prediction_prob > 0.85 and recent_cpu_mean > 0.7:
            target_freq = self.HIGH_FREQ
        elif prediction_prob > 0.55:
            target_freq = self.MID_FREQ
        else:
            target_freq = self.LOW_FREQ
        
        # First iteration - initialize
        if self.current_freq is None:
            self.current_freq = target_freq
            self.hold_counter = self.HOLD_HIGH if target_freq == self.HIGH_FREQ else self.HOLD_LOW
            return self.current_freq
        
        # Hysteresis: hold current frequency if counter > 0
        if self.hold_counter > 0:
            self.hold_counter -= 1
            return self.current_freq
        
        # Allow transition after hold period
        if target_freq != self.current_freq:
            self.current_freq = target_freq
            self.hold_counter = self.HOLD_HIGH if target_freq == self.HIGH_FREQ else self.HOLD_LOW
        
        return self.current_freq
    
    def calculate_energy(self, frequency, prev_frequency, num_processes=None):
        """
        Calculate energy consumption using physics-based model.
        
        Energy = f² + α·|Δf|·f
        
        Args:
            frequency: Current frequency in MHz
            prev_frequency: Previous frequency in MHz
            num_processes: Number of active processes (optional)
        
        Returns:
            energy: Energy consumption (arbitrary units)
        """
        # Frequency transition penalty (Stack A)
        freq_delta = abs(frequency - prev_frequency)
        
        # Base energy + transition cost
        energy = frequency ** 2 + self.ALPHA * freq_delta * frequency
        
        # Core-idle awareness (Stack B)
        if num_processes is not None:
            active_ratio = min(1.0, num_processes / self.LOGICAL_CORES)
            energy *= active_ratio
        
        return energy


def simulate_smartwatt_dvfs(df, cpu_column, y_prob, num_processes_column=None,
                           governor=None, verbose=True):
    """
    Simulate Smart-Watt DVFS over entire dataset.
    
    Args:
        df: Dataframe with CPU data
        cpu_column: Name of CPU utilization column
        y_prob: Array of prediction probabilities
        num_processes_column: Name of process count column (optional)
        governor: SmartWattDVFS instance (creates default if None)
        verbose: Print progress (default: True)
    
    Returns:
        df_sim: Dataframe with simulation results
        total_energy: Total energy consumption
        stats: Dictionary with statistics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"SMART-WATT DVFS SIMULATION")
        print(f"{'='*60}")
    
    # Initialize governor
    if governor is None:
        governor = SmartWattDVFS()
    
    if verbose:
        print(f"\nGovernor configuration:")
        print(f"  Frequencies: LOW={governor.LOW_FREQ}, MID={governor.MID_FREQ}, HIGH={governor.HIGH_FREQ} MHz")
        print(f"  Hysteresis: HOLD_HIGH={governor.HOLD_HIGH}, HOLD_LOW={governor.HOLD_LOW}")
        print(f"  CPU window: {governor.WINDOW_CPU} samples")
        print(f"  Transition penalty: α={governor.ALPHA}")
    
    # Skip first window samples (used for features)
    window = 5
    df_sim = df.iloc[window:].copy().reset_index(drop=True)
    
    # Align probability array
    min_len = min(len(df_sim), len(y_prob))
    df_sim = df_sim.iloc[:min_len].copy()
    y_prob = y_prob[:min_len]
    
    if verbose:
        print(f"\nSimulating {len(df_sim):,} samples...")
    
    # Reset governor
    governor.reset()
    
    # Simulate
    smart_freqs = []
    energies = []
    
    for idx in range(len(df_sim)):
        cpu_util = df_sim.iloc[idx][cpu_column]
        prob = y_prob[idx]
        
        # Normalize CPU if needed
        if cpu_util > 1.0:
            cpu_util = cpu_util / 100.0
        
        # Decide frequency
        freq = governor.decide_frequency(cpu_util, prob)
        smart_freqs.append(freq)
        
        # Calculate energy
        prev_freq = smart_freqs[-2] if len(smart_freqs) > 1 else freq
        
        if num_processes_column and num_processes_column in df_sim.columns:
            num_procs = df_sim.iloc[idx][num_processes_column]
        else:
            num_procs = None
        
        energy = governor.calculate_energy(freq, prev_freq, num_procs)
        energies.append(energy)
    
    # Add results to dataframe
    df_sim['smart_freq'] = smart_freqs
    df_sim['prediction_prob'] = y_prob
    df_sim['energy'] = energies
    df_sim['freq_delta'] = df_sim['smart_freq'].diff().abs().fillna(0)
    
    # Calculate statistics
    total_energy = df_sim['energy'].sum()
    freq_transitions = (df_sim['freq_delta'] > 0).sum()
    freq_counts = df_sim['smart_freq'].value_counts()
    
    if verbose:
        print(f"\n✅ Simulation complete!")
        print(f"\nResults:")
        print(f"  Total energy: {total_energy:,.0f}")
        print(f"  Frequency transitions: {freq_transitions}")
        print(f"\n  Frequency usage:")
        for freq in sorted(freq_counts.index, reverse=True):
            count = freq_counts[freq]
            pct = count / len(df_sim) * 100
            print(f"    {freq} MHz: {count:,} samples ({pct:.1f}%)")
    
    stats = {
        'total_energy': total_energy,
        'transitions': freq_transitions,
        'freq_distribution': freq_counts,
        'avg_energy_per_sample': total_energy / len(df_sim)
    }
    
    return df_sim, total_energy, stats


def compare_baseline_vs_smartwatt(df, cpu_column, y_prob, threshold=0.30, 
                                  num_processes_column=None):
    """
    Compare baseline DVFS (threshold-based) vs Smart-Watt (ML-based).
    
    Args:
        df: Dataframe with CPU data
        cpu_column: Name of CPU utilization column
        y_prob: ML prediction probabilities
        threshold: CPU threshold for baseline (default: 30%)
        num_processes_column: Process count column (optional)
    
    Returns:
        comparison: Dictionary with comparison metrics
    """
    print(f"\n{'='*60}")
    print(f"BASELINE vs SMART-WATT COMPARISON")
    print(f"{'='*60}")
    
    # Baseline DVFS (simple threshold)
    window = 5
    df_baseline = df.iloc[window:].copy().reset_index(drop=True)
    min_len = min(len(df_baseline), len(y_prob))
    df_baseline = df_baseline.iloc[:min_len].copy()
    
    cpu_vals = df_baseline[cpu_column].values
    if cpu_vals.max() > 1.0:
        cpu_vals = cpu_vals / 100.0
    
    baseline_freqs = np.where(cpu_vals > threshold, 2400, 1520)
    df_baseline['baseline_freq'] = baseline_freqs
    df_baseline['freq_delta'] = df_baseline['baseline_freq'].diff().abs().fillna(0)
    
    # Calculate baseline energy
    governor = SmartWattDVFS()
    baseline_energies = []
    for idx in range(len(df_baseline)):
        freq = baseline_freqs[idx]
        prev_freq = baseline_freqs[idx-1] if idx > 0 else freq
        
        if num_processes_column and num_processes_column in df_baseline.columns:
            num_procs = df_baseline.iloc[idx][num_processes_column]
        else:
            num_procs = None
        
        energy = governor.calculate_energy(freq, prev_freq, num_procs)
        baseline_energies.append(energy)
    
    df_baseline['energy'] = baseline_energies
    baseline_energy = sum(baseline_energies)
    baseline_transitions = (df_baseline['freq_delta'] > 0).sum()
    
    print(f"\n📊 BASELINE (Threshold-based):")
    print(f"  Total energy: {baseline_energy:,.0f}")
    print(f"  Transitions: {baseline_transitions}")
    
    # Smart-Watt DVFS
    df_smart, smart_energy, smart_stats = simulate_smartwatt_dvfs(
        df, cpu_column, y_prob, num_processes_column, verbose=False
    )
    
    print(f"\n🧠 SMART-WATT (ML-based):")
    print(f"  Total energy: {smart_energy:,.0f}")
    print(f"  Transitions: {smart_stats['transitions']}")
    
    # Calculate savings
    energy_savings = (baseline_energy - smart_energy) / baseline_energy * 100
    transition_reduction = (baseline_transitions - smart_stats['transitions']) / baseline_transitions * 100
    
    print(f"\n💰 SAVINGS:")
    print(f"  Energy: {energy_savings:.2f}%")
    print(f"  Transitions: {transition_reduction:.2f}%")
    
    comparison = {
        'baseline_energy': baseline_energy,
        'smartwatt_energy': smart_energy,
        'energy_savings_pct': energy_savings,
        'baseline_transitions': baseline_transitions,
        'smartwatt_transitions': smart_stats['transitions'],
        'transition_reduction_pct': transition_reduction
    }
    
    return comparison


if __name__ == "__main__":
    # Test the module
    print("Testing Smart-Watt DVFS Simulator\n")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    df_test = pd.DataFrame({
        'cpu_util': np.random.rand(n_samples),
        'num_processes': np.random.randint(1, 16, n_samples)
    })
    
    # Fake probabilities
    y_prob_test = np.random.rand(n_samples - 5)
    
    # Test governor
    governor = SmartWattDVFS()
    print(f"Governor initialized:")
    print(f"  LOW: {governor.LOW_FREQ} MHz")
    print(f"  MID: {governor.MID_FREQ} MHz")
    print(f"  HIGH: {governor.HIGH_FREQ} MHz")
    
    # Simulate
    df_sim, energy, stats = simulate_smartwatt_dvfs(
        df_test, 'cpu_util', y_prob_test, 'num_processes'
    )
    
    print(f"\n✅ Test complete!")
    print(f"   Simulated samples: {len(df_sim)}")
    print(f"   Total energy: {energy:,.0f}")
