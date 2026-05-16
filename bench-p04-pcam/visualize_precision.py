"""
Directional Precision Allocation Visualization for PCAM Precision Agent.

This script demonstrates that our agent (adapters.myteam:Engine) uses 
per-coordinate precision modulation to steer retrieval dynamics, 
rather than relying on uniform global precision.

Outputs:
    - precision_vector_visualization.png
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure the parent directory is in the path so we can import adapters
sys.path.append(os.getcwd())

from data import make_patterns, corrupt
from pcam_model import build_default_R
from adapters.myteam import Engine

def generate_visualization():
    # --- 1. Setup Environment (Matching Benchmark L1 Canonical Seed) ---
    seed = 42
    K, N = 16, 64
    noise_level = 0.8  # Heavy noise to see selective steering
    rng = np.random.default_rng(seed)

    print(f"Initializing Engine with seed {seed}, N={N}...")
    X = make_patterns(K=K, N=N, seed=seed)
    R = build_default_R(N=N, seed=seed)
    
    # Standard benchmark parameters
    params = {
        "R": R,
        "eta": 0.5,
        "beta": 8.0,
        "dt": 0.01,
        "T_max": 3000,
        "tol": 1e-6,
        "T_in": 100,
        "pi_min": 0.1,
        "pi_max": 10.0
    }

    agent = Engine(X, params)

    # --- 2. Generate Noisy Query ---
    pattern_idx = 0  # Focus on the first pattern
    original_pattern = X[pattern_idx]
    corrupted_query = corrupt(original_pattern, p=noise_level, rng=rng)

    # --- 3. Run Precision Prediction ---
    precision = agent.predict_precision(corrupted_query)

    # --- 4. Debug Info ---
    print("\n--- Precision Vector Statistics ---")
    print(f"Mean Precision: {np.mean(precision):.4f}")
    print(f"Max Precision:  {np.max(precision):.4f}")
    print(f"Min Precision:  {np.min(precision):.4f}")
    
    top_5_amp = np.argsort(precision)[-5:][::-1]
    top_5_sup = np.argsort(precision)[:5]
    
    print(f"Top 5 Amplified Coordinates: {top_5_amp}")
    print(f"Top 5 Suppressed Coordinates: {top_5_sup}")

    # --- 5. Visualization ---
    plt.style.use('bmh') # Clean presentation style
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    colors = ['#2E86C1' if p > 1.0 else '#E74C3C' for p in precision]
    
    # PANEL 1: Precision Weights
    bars = ax1.bar(range(N), precision, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax1.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label="Uniform Precision Baseline")
    
    # Annotate top 5
    for i in top_5_amp:
        ax1.annotate('boosted', xy=(i, precision[i]), xytext=(0, 5), 
                     textcoords="offset points", ha='center', fontsize=9, fontweight='bold', color='#1B4F72')
    for i in top_5_sup:
        ax1.annotate('suppressed', xy=(i, precision[i]), xytext=(0, -15), 
                     textcoords="offset points", ha='center', fontsize=9, fontweight='bold', color='#7B241C')

    ax1.set_ylabel("Precision Weight $(\pi_i)$", fontsize=14, fontweight='bold')
    ax1.set_title("Per-Coordinate Precision Steering", fontsize=18, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', frameon=True, fontsize=12)
    ax1.set_ylim(0, np.max(precision) * 1.2) 


    # PANEL 2: Query Magnitude (Correlation)
    # We show the absolute value to represent 'signal strength' at each coordinate
    query_mag = np.abs(corrupted_query)
    ax2.fill_between(range(N), query_mag, color='#7D3C98', alpha=0.3, label="Noisy Input Magnitude")
    ax2.plot(range(N), query_mag, color='#7D3C98', linewidth=1.5)
    
    ax2.set_ylabel("|Input Value|", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Coordinate Index (0 \u2192 63)", fontsize=14, fontweight='bold')
    ax2.set_title("Input Saliency vs. Allocation", fontsize=14, alpha=0.7)
    ax2.set_ylim(0, np.max(query_mag)*1.2)
    ax2.legend(loc='upper right', fontsize=12)

    # Styling
    plt.xticks(np.arange(0, 65, 5))
    plt.tight_layout()
    
    save_path = "precision_vector_visualization.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved successfully to: {save_path}")
    plt.show()

if __name__ == "__main__":
    generate_visualization()
