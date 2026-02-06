"""
Windows vs Ubuntu DVFS Comparison - Main Script

This script runs the complete Smart-Watt DVFS comparison analysis
between Windows and Ubuntu laptop data.

Usage:
    python run_comparison.py

Requirements:
    - cpu_log_prepared.csv (Windows data from vindhya folder)
    - ubuntu_laptop_data.csv (Ubuntu data)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from smartwatt_features import prepare_dataset
from smartwatt_train import train_smartwatt_classifier
from smartwatt_dvfs import simulate_smartwatt_dvfs, compare_baseline_vs_smartwatt

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def find_data_files():
    """Locate Windows and Ubuntu data files."""
    print("🔍 Searching for data files...\n")
    
    # Possible locations for Windows data
    windows_paths = [
        'data/cpu_log_prepared.csv',
        '../local_data/vindhya/DVFS_F/data/cpu_log_prepared.csv',
        '../vindhya/DVFS_F/data/cpu_log_prepared.csv'
    ]
    
    # Possible locations for Ubuntu data
    ubuntu_paths = [
        'data/ubuntu_laptop_data.csv',
        '../local_data/ubuntu_laptop_data.csv',
        '../ubuntu_laptop_data.csv'
    ]
    
    windows_path = None
    for path in windows_paths:
        if os.path.exists(path):
            windows_path = path
            print(f"✅ Found Windows data: {path}")
            break
    
    ubuntu_path = None
    for path in ubuntu_paths:
        if os.path.exists(path):
            ubuntu_path = path
            print(f"✅ Found Ubuntu data: {path}")
            break
    
    if not windows_path:
        print("❌ Windows data not found. Please place cpu_log_prepared.csv in data/ folder")
    if not ubuntu_path:
        print("❌ Ubuntu data not found. Please place ubuntu_laptop_data.csv in data/ folder")
    
    return windows_path, ubuntu_path


def analyze_dataset(name, df, cpu_col):
    """Print dataset statistics."""
    print(f"\n{'='*60}")
    print(f"{name} DATASET")
    print(f"{'='*60}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nCPU Statistics:")
    print(df[cpu_col].describe())


def plot_comparison(df_windows_sim, df_ubuntu_sim, output_dir='results'):
    """Create comparison visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 Creating visualizations...")
    
    # 1. Frequency decisions comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Frequency Decisions Comparison', fontsize=16, fontweight='bold')
    
    sample_range = min(500, len(df_windows_sim), len(df_ubuntu_sim))
    
    ax1.plot(df_windows_sim['smart_freq'][:sample_range], 
             linewidth=1.5, alpha=0.8, label='Windows Smart-Watt')
    ax1.set_title('Windows DVFS', fontweight='bold')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Frequency (MHz)')
    ax1.set_ylim([1400, 2500])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df_ubuntu_sim['smart_freq'][:sample_range], 
             linewidth=1.5, alpha=0.8, color='green', label='Ubuntu Smart-Watt')
    ax2.set_title('Ubuntu DVFS', fontweight='bold')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Frequency (MHz)')
    ax2.set_ylim([1400, 2500])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/frequency_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_dir}/frequency_comparison.png")
    
    # 2. Energy distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Energy Distribution Comparison', fontsize=16, fontweight='bold')
    
    ax1.hist(df_windows_sim['energy'], bins=50, alpha=0.7, edgecolor='black')
    ax1.set_title('Windows Energy Distribution', fontweight='bold')
    ax1.set_xlabel('Energy (arbitrary units)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(df_windows_sim['energy'].mean(), color='red', 
                linestyle='--', label=f'Mean: {df_windows_sim["energy"].mean():,.0f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(df_ubuntu_sim['energy'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('Ubuntu Energy Distribution', fontweight='bold')
    ax2.set_xlabel('Energy (arbitrary units)')
    ax2.set_ylabel('Frequency')
    ax2.axvline(df_ubuntu_sim['energy'].mean(), color='red', 
                linestyle='--', label=f'Mean: {df_ubuntu_sim["energy"].mean():,.0f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/energy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_dir}/energy_comparison.png")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("WINDOWS vs UBUNTU: SMART-WATT DVFS COMPARISON")
    print("="*70)
    
    # Find data files
    windows_path, ubuntu_path = find_data_files()
    
    if not windows_path or not ubuntu_path:
        print("\n❌ Cannot proceed without both datasets.")
        print("\nPlease ensure these files are available:")
        print("  1. cpu_log_prepared.csv (Windows data)")
        print("  2. ubuntu_laptop_data.csv (Ubuntu data)")
        return
    
    # Load and analyze Windows data
    print("\n" + "="*70)
    print("PART 1: WINDOWS DATA PROCESSING")
    print("="*70)
    
    X_win, y_win, df_win = prepare_dataset(
        windows_path, 
        cpu_column='cpu_util',
        window=5,
        horizon=5,
        threshold=0.30,
        normalize=False  # Already normalized
    )
    
    # Train Windows model
    model_win, y_prob_win, metrics_win = train_smartwatt_classifier(
        X_win, y_win,
        model_name="Windows Smart-Watt",
        save_path="models/smartwatt_windows.pkl"
    )
    
    # Simulate Windows DVFS
    df_win_sim, energy_win, stats_win = simulate_smartwatt_dvfs(
        df_win,
        cpu_column='cpu_util',
        y_prob=y_prob_win,
        num_processes_column='num_processes'
    )
    
    # Load and analyze Ubuntu data
    print("\n" + "="*70)
    print("PART 2: UBUNTU DATA PROCESSING")
    print("="*70)
    
    X_ubuntu, y_ubuntu, df_ubuntu = prepare_dataset(
        ubuntu_path,
        cpu_column='cpu_usage',
        window=5,
        horizon=5,
        threshold=0.30,
        normalize=True  # Normalize from 0-100 to 0-1
    )
    
    # Train Ubuntu model
    model_ubuntu, y_prob_ubuntu, metrics_ubuntu = train_smartwatt_classifier(
        X_ubuntu, y_ubuntu,
        model_name="Ubuntu Smart-Watt",
        save_path="models/smartwatt_ubuntu.pkl"
    )
    
    # Simulate Ubuntu DVFS
    df_ubuntu_sim, energy_ubuntu, stats_ubuntu = simulate_smartwatt_dvfs(
        df_ubuntu,
        cpu_column='cpu_usage',
        y_prob=y_prob_ubuntu,
        num_processes_column=None  # Ubuntu data doesn't have this
    )
    
    # Comparison
    print("\n" + "="*70)
    print("PART 3: COMPARISON ANALYSIS")
    print("="*70)
    
    comparison_data = {
        'OS': ['Windows', 'Ubuntu'],
        'Samples': [len(df_win_sim), len(df_ubuntu_sim)],
        'Avg_CPU_%': [
            df_win['cpu_util'].mean() * 100,
            df_ubuntu['cpu_usage'].mean()
        ],
        'Model_Accuracy_%': [
            metrics_win['test_accuracy'] * 100,
            metrics_ubuntu['test_accuracy'] * 100
        ],
        'Total_Energy': [energy_win, energy_ubuntu],
        'Energy_per_Sample': [
            energy_win / len(df_win_sim),
            energy_ubuntu / len(df_ubuntu_sim)
        ],
        'Freq_Transitions': [
            stats_win['transitions'],
            stats_ubuntu['transitions']
        ],
        'HIGH_freq_%': [
            (df_win_sim['smart_freq'] == 2400).sum() / len(df_win_sim) * 100,
            (df_ubuntu_sim['smart_freq'] == 2400).sum() / len(df_ubuntu_sim) * 100
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n📊 COMPARISON TABLE:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    os.makedirs('results', exist_ok=True)
    comparison_df.to_csv('results/os_comparison.csv', index=False)
    print(f"\n💾 Saved: results/os_comparison.csv")
    
    # Save simulation results
    df_win_sim.to_csv('results/windows_dvfs_results.csv', index=False)
    df_ubuntu_sim.to_csv('results/ubuntu_dvfs_results.csv', index=False)
    print(f"💾 Saved: results/windows_dvfs_results.csv")
    print(f"💾 Saved: results/ubuntu_dvfs_results.csv")
    
    # Create visualizations
    plot_comparison(df_win_sim, df_ubuntu_sim)
    
    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    print(f"\n1️⃣  CPU BEHAVIOR:")
    print(f"   • Windows avg CPU: {comparison_data['Avg_CPU_%'][0]:.2f}%")
    print(f"   • Ubuntu avg CPU: {comparison_data['Avg_CPU_%'][1]:.2f}%")
    
    print(f"\n2️⃣  MODEL ACCURACY:")
    print(f"   • Windows: {comparison_data['Model_Accuracy_%'][0]:.2f}%")
    print(f"   • Ubuntu: {comparison_data['Model_Accuracy_%'][1]:.2f}%")
    
    print(f"\n3️⃣  ENERGY EFFICIENCY:")
    win_energy = comparison_data['Energy_per_Sample'][0]
    ubuntu_energy = comparison_data['Energy_per_Sample'][1]
    more_efficient = 'Windows' if win_energy < ubuntu_energy else 'Ubuntu'
    savings = abs(win_energy - ubuntu_energy) / max(win_energy, ubuntu_energy) * 100
    print(f"   • Windows energy/sample: {win_energy:,.2f}")
    print(f"   • Ubuntu energy/sample: {ubuntu_energy:,.2f}")
    print(f"   • More efficient: {more_efficient} ({savings:.1f}% better)")
    
    print(f"\n4️⃣  FREQUENCY STABILITY:")
    print(f"   • Windows transitions: {comparison_data['Freq_Transitions'][0]}")
    print(f"   • Ubuntu transitions: {comparison_data['Freq_Transitions'][1]}")
    more_stable = 'Windows' if comparison_data['Freq_Transitions'][0] < comparison_data['Freq_Transitions'][1] else 'Ubuntu'
    print(f"   • More stable: {more_stable}")
    
    print(f"\n5️⃣  HIGH FREQUENCY USAGE:")
    print(f"   • Windows @ 2400 MHz: {comparison_data['HIGH_freq_%'][0]:.1f}% of time")
    print(f"   • Ubuntu @ 2400 MHz: {comparison_data['HIGH_freq_%'][1]:.1f}% of time")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  📁 models/")
    print("     • smartwatt_windows.pkl")
    print("     • smartwatt_ubuntu.pkl")
    print("  📁 results/")
    print("     • os_comparison.csv")
    print("     • windows_dvfs_results.csv")
    print("     • ubuntu_dvfs_results.csv")
    print("     • frequency_comparison.png")
    print("     • energy_comparison.png")
    

if __name__ == "__main__":
    main()
