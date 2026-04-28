import sys
import os
import airborne_antara
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'NOT SET')}")
print(f"CWD: {os.getcwd()}")
print(f"airborne_antara file: {airborne_antara.__file__}")
print("SYS PATH:")
for p in sys.path:
    print(f"  {p}")
