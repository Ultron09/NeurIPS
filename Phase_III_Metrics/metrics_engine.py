import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

class MetricsEngine:
    """
    Standardizes Continual Learning metrics using the Ri,j accuracy matrix.
    Includes baseline calibration for Forward Transfer (FWT).
    """
    def __init__(self, num_tasks: int = 10, classes_per_task: int = 10):
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        
        # R[i, j] is the accuracy on task j after training on task i
        self.r_matrix = np.zeros((num_tasks, num_tasks))
        
        # Standard Random Baseline (1 / Total Classes seen for a static head, 
        # but for FWT in sequential task-heads, it's 1 / classes_in_task)
        self.random_accuracy = np.full(num_tasks, 1.0 / self.classes_per_task)

    def update_result(self, current_task: int, eval_task: int, accuracy: float):
        self.r_matrix[current_task, eval_task] = accuracy

    def calculate_acc(self) -> float:
        """Average Accuracy (ACC) after all tasks are completed."""
        return np.mean(self.r_matrix[self.num_tasks-1])

    def calculate_bwt(self) -> float:
        """Backward Transfer (BWT): Influence of new tasks on older ones."""
        t = self.num_tasks
        bwt = 0
        for i in range(t - 1):
            bwt += self.r_matrix[t-1, i] - self.r_matrix[i, i]
        return bwt / (t - 1)

    def calculate_fwt(self) -> float:
        """Forward Transfer (FWT): Influence of past tasks on future tasks."""
        t = self.num_tasks
        fwt = 0
        for i in range(1, t):
            # R[i-1, i] is the accuracy on task i before training on it
            fwt += self.r_matrix[i-1, i] - self.random_accuracy[i]
        return fwt / (t - 1)

    def generate_report(self):
        acc = self.calculate_acc()
        bwt = self.calculate_bwt()
        fwt = self.calculate_fwt()
        
        print("\n" + "=" * 40)
        print("     NEURIPS CONTINUAL LEARNING REPORT")
        print("=" * 40)
        print(f"Average Accuracy (ACC):  {acc:.4f}")
        print(f"Backward Transfer (BWT): {bwt:.4f}")
        print(f"Forward Transfer (FWT):  {fwt:.4f}")
        print("-" * 40)
        print("Note: FWT baseline calibrated to 10% for Class-Incremental tasks.")
        print("=" * 40 + "\n")

    def plot_heatmap(self, filename="accuracy_heatmap.png"):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.r_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                    xticklabels=[f"Task {i+1}" for i in range(self.num_tasks)],
                    yticklabels=[f"After T{i+1}" for i in range(self.num_tasks)])
        plt.title("R Matrix: Accuracy Evolution Matrix")
        plt.xlabel("Evaluation Task Domain")
        plt.ylabel("Training Phase (Sequential)")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
