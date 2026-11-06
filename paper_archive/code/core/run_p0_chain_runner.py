# -*- coding: utf-8 -*-
"""P0走稳链式运行器：顺序执行所有6个脚本，遇错停止"""
import subprocess, sys, time, os

PY = r"D:\miniconda3\envs\ml\python.exe"
DIR = r"<LOCAL_WORKSPACE>"

STEPS = [
    ("ip1_33bus_n8",   "dn_ip1_33bus_n8.py"),
    ("ip1_69bus_n8",   "dn_ip1_69bus_n8.py"),
    ("ipc_33bus_n8",   "dn_ipc_33bus_n8.py"),
    ("ipc_69bus_n8",   "dn_ipc_69bus_n8.py"),
    ("ipc_119bus_5s",  "dn_ipc_119bus_5seeds.py"),
    ("final_stats",    "dn_p0_final_stats_n8.py"),
]

for name, script in STEPS:
    log_path = os.path.join(DIR, f"{name}_run_log.txt")
    script_path = os.path.join(DIR, script)
    # 如果result文件已存在则跳过（断点续跑）
    result_map = {
        "ip1_33bus_n8":  "ip1_33bus_n8_result.txt",
        "ip1_69bus_n8":  "ip1_69bus_n8_result.txt",
        "ipc_33bus_n8":  "ipc_33bus_n8_result.txt",
        "ipc_69bus_n8":  "ipc_69bus_n8_result.txt",
        "ipc_119bus_5s": "ipc_119bus_5seed_result.txt",
        "final_stats":   None,
    }
    result_file = result_map.get(name)
    if result_file and os.path.exists(os.path.join(DIR, result_file)):
        print(f"[SKIP] {name}: result file exists", flush=True)
        continue
    print(f"\n{'='*60}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] START: {name}", flush=True)
    with open(log_path, 'w', encoding='utf-8') as lf:
        proc = subprocess.run([PY, script_path], stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[FAIL] {name} exited with code {proc.returncode}", flush=True)
        print(f"  Log: {log_path}", flush=True)
        sys.exit(1)
    print(f"[{time.strftime('%H:%M:%S')}] DONE: {name}", flush=True)

print(f"\nP0 SOLID CHAIN COMPLETE  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
