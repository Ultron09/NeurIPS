import json
import sys
import os

def run_sentinel_audit(results_path, target_acc=60.0):
    print("\n--- 🤖 ANTARA PERFORMANCE SENTINEL ---")
    
    if not os.path.exists(results_path):
        print(f"❌ Error: Results file not found at {results_path}")
        sys.exit(1)
        
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Extract metrics (assuming MetricsEngine format)
    # Most Antara results store mean accuracy as 'mean_accuracy' or compute it from 'task_accuracies'
    task_accs = data.get('task_accuracies', [])
    if not task_accs:
        # Fallback for different JSON formats
        task_accs = data.get('results', {}).get('accuracies', [])
        
    if not task_accs:
        print("❌ Error: Could not extract accuracy metrics from results.")
        sys.exit(1)
        
    mean_acc = sum(task_accs) / len(task_accs)
    # Check if it's 0-1 or 0-100
    if mean_acc <= 1.0: mean_acc *= 100.0
    
    # Forgetting check (Retention Stability)
    # Looking for 'backward_transfer' or similar
    forgetting = data.get('forgetting', 0.0)
    
    print(f"Target Accuracy: {target_acc}%")
    print(f"Current Mean Accuracy: {mean_acc:.2f}%")
    print(f"Stability (Forgetting): {forgetting:.2f}")
    
    is_better = mean_acc >= target_acc
    
    if is_better:
        print("\n🏆 VERDICT: IMPROVED / STABLE")
        print(f"✅ The model met the {target_acc}% threshold.")
    else:
        print("\n⚠️ VERDICT: REGRESSION / SUB-TARGET")
        print(f"❌ The model failed the {target_acc}% gauntlet.")
        
    # Create a summary markdown for GitHub Actions
    with open('sentinel_report.md', 'w') as f:
        f.write("# 🤖 Antara Sentinel Report\n\n")
        f.write(f"| Metric | Value | Status |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        f.write(f"| **Mean Accuracy** | {mean_acc:.2f}% | {'✅' if is_better else '❌'} |\n")
        f.write(f"| **Stability (BWT)** | {forgetting:.2f} | {'🛡️' if forgetting > -5 else '📉'} |\n\n")
        f.write(f"**Verdict:** {'🏆 Target Met' if is_better else '⚠️ Needs Optimization'}\n")

    if not is_better:
        # We don't necessarily want to fail the CI build, 
        # but we want the report to be visible.
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--target", type=float, default=60.0)
    args = parser.parse_args()
    
    run_sentinel_audit(args.results, args.target)
