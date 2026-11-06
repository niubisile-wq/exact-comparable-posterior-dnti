# -*- coding: utf-8 -*-
"""Step5串联：DNN基线 → Fisher-OED → 消融表"""
import subprocess, sys, os
os.chdir(r"<LOCAL_WORKSPACE>")
python = r"D:\miniconda3\envs\ml\python.exe"
for script in ["dn_step5_dnn_baseline.py",
               "dn_step5_fisher_oed.py",
               "dn_step5_ablation_table.py"]:
    print(f"\n{'='*60}\nRunning {script}...\n{'='*60}", flush=True)
    ret = subprocess.call([python, script])
    if ret != 0:
        print(f"ERROR: {script} exited with code {ret}", flush=True)
        sys.exit(ret)
    print(f"{script} DONE.", flush=True)
print("\nStep5 complete.", flush=True)
