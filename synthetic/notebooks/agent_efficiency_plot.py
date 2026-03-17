"""
Agent Efficiency Analysis: Task Completion vs Question Efficiency

This visualization shows which agents accomplish the most tasks while not
pinging the user for unnecessary information.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Load data from CSV
df = pd.read_csv('detection_frequency.csv')

# Define colors and markers for datasets and models
dataset_colors = {'MCP': '#2ecc71', 'TAC': '#3498db', 'SWE-Bench Pro': '#e74c3c'}
model_markers = {
    'Claude Opus 4.5': 'o',
    'Claude Sonnet 4.5': 'o',
    'Sonnet 4.5': 'o',  # alias
    'Gemini 3 Pro': '^',
    'Gemini 3 Flash': 'D',
    'GPT-5.2': 'P'
}

# Create the main figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# =============================================================================
# Plot 1: Pass@3 Score vs Pass3_Gain_per_Ask
# =============================================================================
ax1 = axes[0]

for idx, row in df.iterrows():
    marker = model_markers.get(row['Model'], 'o')
    color = dataset_colors.get(row['Data'], 'gray')
    ax1.scatter(row['Pass3_Gain_per_Ask'], row['Pass3_Score'], 
                marker=marker, c=color, s=200, alpha=0.8, edgecolors='black', linewidths=1)

ax1.set_xlabel('Gain/Ask (score improvement per question asked)', fontsize=12)
ax1.set_ylabel('Pass@3 Score (%)', fontsize=12)
ax1.set_title('Pass@3 Score (with ask_user) vs Question Efficiency\n(Upper-right = Best)', fontsize=14, fontweight='bold')
ax1.set_xlim(-0.02, 0.28)
ax1.set_ylim(25, 75)

# =============================================================================
# Plot 2: Ckpt Score vs Ckpt_Gain_per_Ask
# =============================================================================
ax2 = axes[1]

for idx, row in df.iterrows():
    marker = model_markers.get(row['Model'], 'o')
    color = dataset_colors.get(row['Data'], 'gray')
    ax2.scatter(row['Ckpt_Gain_per_Ask'], row['Ckpt_Score'], 
                marker=marker, c=color, s=200, alpha=0.8, edgecolors='black', linewidths=1)

ax2.set_xlabel('Gain/Ask (score improvement per question asked)', fontsize=12)
ax2.set_ylabel('Checkpoint Score (%)', fontsize=12)
ax2.set_title('Checkpoint Score (with ask_user) vs Question Efficiency\n(Upper-right = Best)', fontsize=14, fontweight='bold')
ax2.set_xlim(-0.12, 0.20)
ax2.set_ylim(25, 75)

# =============================================================================
# Create unified legend
# =============================================================================
# Model legend (markers)
model_handles = []
for model, marker in model_markers.items():
    if model != 'Sonnet 4.5':  # Skip alias
        handle = plt.Line2D([0], [0], marker=marker, color='gray', 
                           markersize=10, linestyle='None', markeredgecolor='black')
        model_handles.append((handle, model))

# Dataset legend (colors)
dataset_handles = []
for dataset, color in dataset_colors.items():
    handle = plt.Line2D([0], [0], marker='o', color=color, 
                       markersize=10, linestyle='None', markeredgecolor='black')
    dataset_handles.append((handle, dataset))

# Add legends to the figure
leg1 = fig.legend([h[0] for h in model_handles], [h[1] for h in model_handles],
                  loc='upper left', bbox_to_anchor=(0.02, 0.98), title='Model', 
                  framealpha=0.9, fontsize=9)
leg2 = fig.legend([h[0] for h in dataset_handles], [h[1] for h in dataset_handles],
                  loc='upper left', bbox_to_anchor=(0.02, 0.68), title='Dataset',
                  framealpha=0.9, fontsize=9)

plt.tight_layout()
plt.subplots_adjust(left=0.15)  # Make room for legends

plt.savefig('agent_efficiency_comparison.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('agent_efficiency_comparison.pdf', bbox_inches='tight')
print("Saved: agent_efficiency_comparison.png and .pdf")

# =============================================================================
# Print summary
# =============================================================================
print("\n" + "="*80)
print("AGENT EFFICIENCY SUMMARY")
print("="*80)
print(f"\n{'Model':<20} {'Dataset':<15} {'Pass3':<8} {'Pass3 G/A':<10} {'Ckpt':<8} {'Ckpt G/A':<10}")
print("-"*80)
for _, row in df.sort_values('Pass3_Gain_per_Ask', ascending=False).iterrows():
    print(f"{row['Model']:<20} {row['Data']:<15} {row['Pass3_Score']:<8.1f} {row['Pass3_Gain_per_Ask']:<10.2f} {row['Ckpt_Score']:<8.1f} {row['Ckpt_Gain_per_Ask']:<10.2f}")
