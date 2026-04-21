import sys
import os

# Emulate the path setup in benchmark_runner.py
sys.path.append(os.path.join(os.getcwd(), 'Phase_IV_Ablation'))

# Ensure local version of airborne_antara is used over installed package
framework_path = os.path.join(os.path.dirname(os.getcwd()), 'Mirror_mind')
if os.path.exists(framework_path):
    sys.path.insert(0, framework_path)

try:
    import airborne_antara
    from airborne_antara import AdaptiveFrameworkConfig
except ImportError as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print(f"PATH: {airborne_antara.__file__}")
config = AdaptiveFrameworkConfig()
print(f"USE_OGD_PRESENT: {hasattr(config, 'use_ogd')}")
if hasattr(config, 'use_ogd'):
    print(f"USE_OGD_VALUE: {config.use_ogd}")
