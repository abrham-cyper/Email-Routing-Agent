import json
import matplotlib.pyplot as plt
import numpy as np
import os

print("\n=== 5. Comparison & Visualization ===")

results = []
methods = []
accuracies = []
times = []

# 1. Load Results
print("Loading results...")
agents = [
    ("agent1", "Agent 1 (GPT-2 Zero)", "results/agent1.json"),
    ("agent2", "Agent 2 (GPT-2 LoRA)", "results/agent2.json"),
    ("agent3", "Agent 3 (DistilBERT)", "results/agent3.json"),
]

for agent_id, agent_name, path in agents:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                acc = data.get("accuracy", 0.0)
                time_taken = data.get("time", 0.0)
                
                results.append((agent_name, acc, time_taken))
                methods.append(agent_name)
                accuracies.append(acc)
                times.append(time_taken)
        except Exception as e:
            print(f"Error loading {path}: {e}")

if not results:
    print("No results found! Run the agents first.")
    exit()

# 2. Print Professional Table
print("\n" + "="*60)
print(f"{'METHOD':<25} | {'ACCURACY':<10} | {'TIME (s)':<10}")
print("-" * 60)
for m, a, t in results:
    acc_str = f"{a:.4f}" if a > 0 else "N/A (Skipped)"
    print(f"{m:<25} | {acc_str:<10} | {t:<10.2f}")
print("="*60 + "\n")

# 3. Generate Single Cool Grouped Chart
print("Generating Single Combined Chart...")
try:
    plt.style.use('fivethirtyeight')
except:
    pass

# Create groups
x = np.arange(len(methods))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 7))

# Plot Accuracy Bars (Green) on Left Axis
rects1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', color='#4CAF50', edgecolor='white')
ax1.set_xlabel('Agents', fontweight='bold', fontsize=12)
ax1.set_ylabel('Accuracy (0-1)', color='#4CAF50', fontweight='bold', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontweight='bold', fontsize=10)
ax1.tick_params(axis='y', labelcolor='#4CAF50')
ax1.set_ylim(0, 1.1)

# Plot Time Bars (Orange) on Right Axis
ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, times, width, label='Time (s)', color='#FF9800', edgecolor='white')
ax2.set_ylabel('Time (seconds)', color='#FF9800', fontweight='bold', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#FF9800')
# Give some headroom for time
max_time = max(times) if times else 10
ax2.set_ylim(0, max_time * 1.2)

# Add Labels ON TOP of bars
def autolabel(rects, ax, is_time=False):
    for rect in rects:
        height = rect.get_height()
        if is_time:
            label = f'{height:.1f}s'
        else:
            label = f'{height:.2f}' if height > 0 else 'N/A'
            
        ax.text(rect.get_x() + rect.get_width()/2., height,
                label,
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

autolabel(rects1, ax1, is_time=False)
autolabel(rects2, ax2, is_time=True)

plt.title('🏆 Accuracy & Time Comparison (One View)', fontweight='bold', fontsize=14, pad=20)
fig.tight_layout()

output_path = "results/comparison_chart.png"
plt.savefig(output_path, dpi=300)
print(f"✅ Chart saved as '{output_path}'")
print("\nDone! Open 'results/comparison_chart.png' to see the single combined graph.")
