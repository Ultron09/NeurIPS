import os
import json
import pandas as pd

def generate_strategic_report():
    results_dir = "results"
    all_results = []
    
    if not os.path.exists(results_dir):
        print(f"No results found in {results_dir}")
        return

    for file in os.listdir(results_dir):
        if file.endswith("_metrics.json"):
            with open(os.path.join(results_dir, file), 'r') as f:
                data = json.load(f)
                all_results.append({
                    "Method": data["config"],
                    "ACC (↑)": f"{data['acc']:.4f}",
                    "BWT (↑)": f"{data['bwt']:.4f}",
                    "FWT (↑)": f"{data['fwt']:.4f}",
                    "Step Time (ms)": f"{data['avg_step_time_ms']:.2f}",
                    "Memory (MB)": f"{data['peak_memory_mb']:.2f}"
                })

    if not all_results:
        print("No metrics JSON files found.")
        return

    df = pd.DataFrame(all_results)
    
    # Define the strategic order
    order = ["ANTARA_FULL", "ANTARA_RGW_ONLY", "ANTARA_OGD_ONLY", "A-GEM", "EWC", "REPLAY", "NAIVE"]
    df['Method'] = pd.Categorical(df['Method'], categories=order, ordered=True)
    df = df.sort_values('Method')

    markdown_table = df.to_markdown(index=False)
    
    report = f"""# NeurIPS Strategic Gauntlet: Final Comparison Report

## 🏁 Comparative Performance Table

{markdown_table}

## 🔬 Analysis vs. Hypotheses

### 1. Stability-Plasticity (BWT vs ACC)
*   **Target**: BWT > -0.05.
*   **Observation**: Check if ANTARA maintains higher stability than EWC without sacrificing accuracy.

### 2. Efficiency Claim (Step Time)
*   **Target**: ANTARA Step Time < A-GEM Step Time.
*   **Observation**: ANTARA's OGD should demonstrate lower latency than A-GEM's secondary projection pass.

### 3. Deliberative Synergy (Ablation)
*   **Target**: Full > RGW > OGD.
*   **Observation**: Intelligence in ANTARA is emergent from the System 1/System 2 interaction.
"""
    
    with open("results/final_strategic_report.md", "w") as f:
        f.write(report)
    
    print("\n[STRATEGY] Unified comparison report generated at results/final_strategic_report.md")

if __name__ == "__main__":
    generate_strategic_report()
