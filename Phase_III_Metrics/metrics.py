import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsEngine:
    """
    Standardizes Continual Learning metrics (ACC, BWT, FWT).
    Uses the Ri,j accuracy matrix.
    """
    def __init__(self, num_tasks=10, config_name="Default"):
        self.num_tasks = num_tasks
        self.config_name = config_name
        self.r_matrix = np.zeros((num_tasks, num_tasks))
        # Baseline per task (1/10 for CIFAR-100 task-heads)
        self.baselines = np.full(num_tasks, 0.10)
        
        # Compute Telemetry
        self.avg_step_time_ms = 0.0
        self.peak_memory_mb = 0.0
        self.total_compute_time_sec = 0.0

    def update(self, t, j, acc):
        self.r_matrix[t, j] = acc

    def calculate_acc(self):
        """Average Accuracy over all tasks after last training phase."""
        return np.mean(self.r_matrix[-1])

    def calculate_bwt(self):
        """Backward Transfer: influence of future tasks on past ones."""
        t = self.num_tasks
        bwt = 0
        for j in range(t - 1):
            bwt += self.r_matrix[t-1, j] - self.r_matrix[j, j]
        return bwt / (t - 1)

    def calculate_fwt(self):
        """Forward Transfer: influence of past tasks on future ones."""
        t = self.num_tasks
        fwt = 0
        for j in range(1, t):
            fwt += self.r_matrix[j-1, j] - self.baselines[j]
        return fwt / (t - 1)

    def save_results(self, path):
        results = {
            "config": self.config_name,
            "acc": self.calculate_acc(),
            "bwt": self.calculate_bwt(),
            "fwt": self.calculate_fwt(),
            "avg_step_time_ms": self.avg_step_time_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "total_compute_time_sec": self.total_compute_time_sec,
            "matrix": self.r_matrix.tolist()
        }
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)

    def generate_summary_report(self):
        acc = self.calculate_acc()
        bwt = self.calculate_bwt()
        fwt = self.calculate_fwt()
        
        # Strategic Alarms
        bwt_status = "✅ PASS" if bwt >= -0.05 else "❌ FAIL (Forgetting too high)"
        fwt_status = "✅ PASS" if fwt >= 0 else "⚠️ NEUTRAL (No Transfer seen)"
        
        print(f"\n========================================\n"
              f"      NEURIPS GAUNTLET REPORT: {self.config_name}\n"
              f"========================================\n"
              f"  Average Accuracy (ACC):  {acc:.4f}\n"
              f"  Backward Transfer (BWT): {bwt:.4f}  [{bwt_status}]\n"
              f"  Forward Transfer (FWT):  {fwt:.4f}  [{fwt_status}]\n"
              f"----------------------------------------\n"
              f"  Avg Step Time:           {self.avg_step_time_ms:.2f} ms\n"
              f"  Peak Memory:             {self.peak_memory_mb:.2f} MB\n"
              f"  Total Training Time:     {self.total_compute_time_sec:.2f} sec\n"
              f"========================================\n")

    def plot_heatmap(self, path):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.r_matrix, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title(f"Accuracy Matrix: {self.config_name}")
        plt.xlabel("Target Task")
        plt.ylabel("Training Milestone")
        plt.savefig(path)
        plt.close()
