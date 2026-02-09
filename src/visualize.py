import json
import matplotlib.pyplot as plt
import os
import glob

print("\n=== 5. Comparison ===")

results = []
methods = []
accuracies = []
times = []

# Load results
if os.path.exists("results/agent1.json"):
    with open("results/agent1.json", "r") as f:
        data = json.load(f)
        results.append(("Agent 1 (GPT-2 Zero-Shot)", data["accuracy"], data["time"]))

if os.path.exists("results/agent2.json"):
    with open("results/agent2.json", "r") as f:
        data = json.load(f)
        results.append(("Agent 2 (GPT-2 LoRA)", data["accuracy"], data["time"]))

if os.path.exists("results/agent3.json"):
    with open("results/agent3.json", "r") as f:
        data = json.load(f)
        results.append(("Agent 3 (DistilBERT)", data["accuracy"], data["time"]))

# Print Table
print(f"{'Method':<25} | {'Accuracy':<10} | {'Time (s)':<10}")
print("-" * 50)
for m, a, t in results:
    print(f"{m:<25} | {a:<10.4f} | {t:<10.2f}")
    methods.append(m)
    accuracies.append(a)
    times.append(t)

# Plot
if methods:
    try:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Accuracy', color=color)
        bars = ax1.bar(methods, accuracies, color=color, alpha=0.6)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 1.0)

        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')

        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('Time (s)', color=color)  
        ax2.plot(methods, times, color=color, marker='o', linestyle='dashed', linewidth=2, markersize=8)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  
        plt.title("NLP Project Result Comparison")
        plt.savefig("results/comparison_chart.png")
        print("\nChart saved as 'results/comparison_chart.png'")
    except Exception as e:
        print(f"Could not save plot: {e}")
else:
    print("No results found to plot.")
