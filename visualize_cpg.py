#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read CPG data
df = pd.read_csv('data/cpg_data/gait_data/cpg_walk_2.0s.csv', header=None)
neuron_outputs = df.iloc[:, :8].values
time = np.arange(len(df)) / 500.0

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot all 8 neurons
axes[0].set_title('8-Neuron CPG Network Output (Walk Gait)', fontsize=14)
for i in range(8):
    axes[0].plot(time, neuron_outputs[:, i], label=f'Neuron {i+1}', linewidth=1.5)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Neuron Output')
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0].grid(True, alpha=0.3)

# Hip vs Knee neurons
axes[1].set_title('Hip vs Knee Neurons (1-4: Hip, 5-8: Knee)', fontsize=14)
hip_mean = neuron_outputs[:, :4].mean(axis=1)
knee_mean = neuron_outputs[:, 4:8].mean(axis=1)
axes[1].plot(time, hip_mean, label='Hip Neurons Avg', color='red', linewidth=2)
axes[1].plot(time, knee_mean, label='Knee Neurons Avg', color='blue', linewidth=2)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Neuron Output')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Phase relationships
axes[2].set_title('Phase Relationship (Neuron 1 vs 3 - Left vs Right)', fontsize=14)
axes[2].plot(time, neuron_outputs[:, 0], label='Neuron 1 (Left Hip)', color='green', linewidth=2)
axes[2].plot(time, neuron_outputs[:, 2], label='Neuron 3 (Right Hip)', color='orange', linewidth=2)
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Neuron Output')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cpg_signal_visualization.png', dpi=300, bbox_inches='tight')
print("Signal visualization saved as: cpg_signal_visualization.png")