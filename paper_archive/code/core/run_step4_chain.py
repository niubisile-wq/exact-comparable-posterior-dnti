# -*- coding: utf-8 -*-
"""Step4 串联：IP-C补种子 → Wilcoxon检验"""
import subprocess, sys, os
os.chdir(r"<LOCAL_WORKSPACE>")
python = r"D:\miniconda3\envs\ml\python.exe"
for script in ["dn_step4_ipc_seeds.py", "dn_step4_wilcoxon.py"]:
    print(f"\n{'='*60}\nRunning {script}...\n{'='*60}", flush=True)
    ret = subprocess.call([python, script])
    if ret != 0:
        print(f"ERROR: {script} exited with code {ret}", flush=True)
        sys.exit(ret)
    print(f"{script} DONE.", flush=True)
print("\nStep4 complete.", flush=True)
