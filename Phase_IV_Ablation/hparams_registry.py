import json
import os
import datetime
from pathlib import Path
from typing import Any, Dict

class HyperparameterRegistry:
    """
    [NEURIPS 2025] Scientific Hyperparameter Registry.
    Ensures reproducibility by logging every experiment's exact configuration.
    """
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "experiment_manifest.json"

    def log_experiment(self, method_name: str, config: Any):
        """
        Logs a single experiment config to the manifest.
        Handles both AdaptiveFrameworkConfig and standard dicts.
        """
        entry = {
            "method": method_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "config": self._to_dict(config)
        }

        manifest = []
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                    # Deduplicate: Remove previous entry for same method to keep only latest
                    manifest = [m for m in manifest if m["method"] != method_name]
            except Exception:
                manifest = []

        manifest.append(entry)

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)
        
        print(f"  [LOG] Configuration registered in manifest: {self.manifest_path}")

    def _to_dict(self, config: Any) -> Dict[str, Any]:
        """Deep conversion of config objects to serializable dicts."""
        if hasattr(config, "__dict__"):
            return {k: self._to_dict(v) for k, v in config.__dict__.items() if not k.startswith("_")}
        elif isinstance(config, dict):
            return {k: self._to_dict(v) for k, v in config.items()}
        elif isinstance(config, (list, tuple)):
            return [self._to_dict(v) for v in config]
        elif isinstance(config, (str, int, float, bool, type(None))):
            return config
        else:
            return str(config)

# Global singleton for easy import
registry = HyperparameterRegistry()
