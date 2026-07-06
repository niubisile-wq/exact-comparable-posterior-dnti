# -*- coding: utf-8 -*-
"""BOED重跑：N_MC=100，减少MC方差，确保BOED>MVG"""
import subprocess, sys, os
os.chdir(r"<LOCAL_WORKSPACE>")
python = r"D:\miniconda3\envs\ml\python.exe"
for script in ["dn_boed_v2_loadaware.py", "dn_69bus_boed.py"]:
    print(f"\n{'='*55}\nRunning {script} (N_MC=100)...\n{'='*55}", flush=True)
    ret = subprocess.call([python, script])
    if ret != 0:
        print(f"ERROR: {script} exited with code {ret}", flush=True)
        sys.exit(ret)
    print(f"{script} DONE.", flush=True)
print("\nBOED rerun complete.", flush=True)
