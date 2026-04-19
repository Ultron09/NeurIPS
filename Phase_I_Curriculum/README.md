# Phase I: The 'Split CIFAR-100' Curriculum

This phase implements a standardized, "grueling" curriculum designed to test a model's resilience against severe distribution shifts. 

## The Task
CIFAR-100 contains 100 object classes. Under the **Split CIFAR-100** protocol, the classes are partitioned into **10 distinct, sequential tasks**, with 10 classes per task.

1. **Sequential Learning**: The model trains on Task 1, then Task 2, and so on.
2. **Zero Visibility**: Once a new task begins, the training distribution of previous tasks becomes entirely invisible (no new samples provided).
3. **Catastrophic Forgetting Test**: This curriculum is the gold standard for measuring a model's ability to retain historical knowledge without direct rehearsal.

## Script Usage

Run the curriculum generator to verify the task partitions:
```bash
python curriculum.py
```

## Implementation Details
- **Dataset**: `torchvision.datasets.CIFAR100`
- **Normalization**: Per-channel mean/std for CIFAR-100.
- **Task Sizes**: 5,000 training images and 1,000 test images per task (10 classes).
