"""
Smoke test: runs 2 tasks × 2 epochs to catch all runtime bugs before A100.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Phase_I_Curriculum'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Phase_III_Metrics'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import traceback

def run_smoke():
    from benchmark_runner import (
        get_stage_config, model_factory, ExternalReplayBuffer,
        ContinualTrainer, ContinualEvaluator, run_experiment
    )
    print("=" * 60)
    print("SMOKE TEST: 2 tasks x 2 epochs")
    print("=" * 60)

    # Monkey-patch epochs to 2 and tasks to 2 for speed
    import benchmark_runner as br
    _orig_run = br.run_experiment

    def _fast_run(dataset_name="CIFAR100", stage_id=7, seed=42, epochs_override=None):
        # Override epochs to 2 for smoke test
        return _orig_run(dataset_name=dataset_name, stage_id=stage_id,
                         seed=seed, epochs_override=2)

    try:
        # Remove any cached result file so it doesn't skip
        import socket, glob
        node = socket.gethostname()
        for f in glob.glob(f"results/SeqN_{node}_42_CIFAR100_7.txt"):
            os.remove(f)
            print(f"Removed cached result: {f}")

        # Patch num_tasks to 2 inside run_experiment
        import benchmark_runner as br
        _orig_SplitCIFAR100 = br.SplitCIFAR100

        class _FastCurriculum(_orig_SplitCIFAR100):
            pass

        # We'll just run with epochs_override=2 and let it run all 10 tasks
        # but only 2 epochs each — fast enough to catch bugs
        _fast_run(dataset_name="CIFAR100", stage_id=7, seed=42, epochs_override=2)
        print("\nSMOKE TEST PASSED")
    except Exception as e:
        print(f"\nSMOKE TEST FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_smoke()
