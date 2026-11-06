# -*- coding: utf-8 -*-
"""串联运行 69-bus IP1 → IP-C"""
import subprocess, sys, os
os.chdir(r"<LOCAL_WORKSPACE>")
python = r"D:\miniconda3\envs\ml\python.exe"

for script in ["dn_69bus_ip1_multiseed.py", "dn_69bus_ipc_multiseed.py"]:
    print(f"\n{'='*60}\nRunning {script}...\n{'='*60}", flush=True)
    ret = subprocess.call([python, script])
    if ret != 0:
        print(f"ERROR: {script} exited with code {ret}", flush=True)
        sys.exit(ret)
    print(f"{script} DONE.", flush=True)

print("\nAll 69-bus experiments complete.", flush=True)
