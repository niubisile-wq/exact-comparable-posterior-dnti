# -*- coding: utf-8 -*-
"""
P0最终统计：n=8 Wilcoxon + Holm-Bonferroni + BH-FDR
主家族(m=6): IP1-33, IP1-69, IP-C-33@10%, IP-C-33@30%, IP-C-69@10%, IP-C-69@30%
补充验证:    IP-C-119bus (n=5, 单独报告)
"""
import warnings
import numpy as np
from scipy import stats
warnings.filterwarnings('ignore')

SAVE_DIR = r"<LOCAL_WORKSPACE>"

def read_ip1(filename):
    lines = open(f"{SAVE_DIR}\\{filename}", encoding='utf-8').readlines()[1:]
    ais, nre = [], []
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) == 3:
            ais.append(float(parts[1])); nre.append(float(parts[2]))
    return np.array(ais), np.array(nre)

def read_ipc(filename):
    lines = open(f"{SAVE_DIR}\\{filename}", encoding='utf-8').readlines()[1:]
    rob10, nai10, rob30, nai30 = [], [], [], []
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) == 5:
            rob10.append(float(parts[1])); nai10.append(float(parts[2]))
            rob30.append(float(parts[3])); nai30.append(float(parts[4]))
    return np.array(rob10), float(nai10[0]), np.array(rob30), float(nai30[0])

def wilcoxon_one_sided(a, b, label):
    a, b = np.array(a), np.array(b)
    diff = a - b
    n = len(diff)
    if np.all(diff >= 0) and not np.all(diff == 0):
        stat, p = stats.wilcoxon(a, b, alternative='greater')
    elif np.all(diff == 0):
        return {'label': label, 'n': n, 'mean_a': np.mean(a), 'mean_b': np.mean(b),
                'delta_mean': 0, 'delta_std': 0, 'p': 1.0, 'verdict': 'ns'}
    else:
        stat, p = stats.wilcoxon(a, b, alternative='greater')
    verdict = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    return {'label': label, 'n': n,
            'mean_a': float(np.mean(a)), 'mean_b': float(np.mean(b)),
            'delta_mean': float(np.mean(diff)), 'delta_std': float(np.std(diff)),
            'p': float(p), 'verdict': verdict}

def holm_bh(p_vals, alpha=0.05):
    m = len(p_vals)
    order = np.argsort(p_vals)
    # Holm
    holm = [False]*m
    for k, idx in enumerate(order):
        threshold = alpha / (m - k)
        if p_vals[idx] <= threshold:
            holm[idx] = True
        else:
            break
    # BH-FDR
    bh = [False]*m
    for k in range(m-1, -1, -1):
        idx = order[k]
        if p_vals[idx] <= (k+1)/m * alpha:
            for j in range(k+1):
                bh[order[j]] = True
            break
    return holm, bh

print("Reading result files...")
try:
    ais33, nre33 = read_ip1("ip1_33bus_n8_result.txt")
    ais69, nre69 = read_ip1("ip1_69bus_n8_result.txt")
    rob10_33, nai10_33, rob30_33, nai30_33 = read_ipc("ipc_33bus_n8_result.txt")
    rob10_69, nai10_69, rob30_69, nai30_69 = read_ipc("ipc_69bus_n8_result.txt")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("请先运行所有训练脚本完成后再运行此脚本。")
    raise

# 主家族 m=6 检验
tests = [
    wilcoxon_one_sided(ais33, nre33, "IP1-33bus  AIS>NRE"),
    wilcoxon_one_sided(ais69, nre69, "IP1-69bus  AIS>NRE"),
    wilcoxon_one_sided(rob10_33, [nai10_33]*len(rob10_33), "IP-C-33bus rob>nai @10%"),
    wilcoxon_one_sided(rob30_33, [nai30_33]*len(rob30_33), "IP-C-33bus rob>nai @30%"),
    wilcoxon_one_sided(rob10_69, [nai10_69]*len(rob10_69), "IP-C-69bus rob>nai @10%"),
    wilcoxon_one_sided(rob30_69, [nai30_69]*len(rob30_69), "IP-C-69bus rob>nai @30%"),
]

p_vals = [t['p'] for t in tests]
holm_rej, bh_rej = holm_bh(p_vals)

print("\n" + "="*100)
print("P0 FINAL SIGNIFICANCE TEST  (n=8 seeds, m=6, Wilcoxon one-sided)")
print("="*100)
print(f"\n{'#':<3} {'Test':<45} {'n':>3}  {'A_mean':>7}  {'B_mean':>7}  {'delta':>13}  {'p_raw':>8}  {'Holm':>6}  {'BH':>6}")
print("-"*100)
for i, (t, h, b) in enumerate(zip(tests, holm_rej, bh_rej)):
    delta_str = f"{t['delta_mean']:+.3f}+/-{t['delta_std']:.3f}"
    print(f"{i+1:<3} {t['label']:<45} {t['n']:>3}  {t['mean_a']:>7.3f}  {t['mean_b']:>7.3f}  "
          f"{delta_str:>13}  {t['p']:>8.4f}  {'* p<.05' if h else 'ns':>6}  {'* q<.05' if b else 'ns':>6}")

holm_count = sum(holm_rej); bh_count = sum(bh_rej)
print("-"*100)
print(f"\nHolm-Bonferroni (FWER): {holm_count}/6 rejected")
print(f"BH-FDR (FDR):           {bh_count}/6 rejected")

min_p = min(p_vals)
holm_thresh1 = 0.05 / 6
print(f"\nMin achievable p (n=8 Wilcoxon one-sided) = 1/2^8 = {1/256:.4f}")
print(f"Holm step-1 threshold = 0.05/6 = {holm_thresh1:.4f}")
if min_p <= holm_thresh1:
    print(f"[OK] Holm passes: min_p={min_p:.4f} <= threshold={holm_thresh1:.4f}")
else:
    print(f"[FAIL] Holm fails: min_p={min_p:.4f} > threshold={holm_thresh1:.4f}")

# 119-bus supplemental
print("\n" + "="*60)
print("SUPPLEMENTAL: IP-C-119bus (n=5, reported separately)")
print("="*60)
try:
    rob10_119, nai10_119, rob30_119, nai30_119 = read_ipc("ipc_119bus_5seed_result.txt")
    t119_10 = wilcoxon_one_sided(rob10_119, [nai10_119]*len(rob10_119), "IP-C-119bus rob>nai @10%")
    t119_30 = wilcoxon_one_sided(rob30_119, [nai30_119]*len(rob30_119), "IP-C-119bus rob>nai @30%")
    for t in [t119_10, t119_30]:
        print(f"  {t['label']}: delta={t['delta_mean']:+.3f}+/-{t['delta_std']:.3f}  p={t['p']:.4f}{t['verdict']:>4}  n={t['n']}")
    print(f"  nai10={nai10_119:.3f}  nai30={nai30_119:.3f}")
except FileNotFoundError:
    print("  ipc_119bus_5seed_result.txt not found - run dn_ipc_119bus_5seeds.py first")

# 综合判定
print("\n" + "="*60)
print("OVERALL P0 ASSESSMENT")
print("="*60)
all_pos_6 = all(t['delta_mean'] > 0 for t in tests)
print(f"  All 6 tests positive delta:  {all_pos_6}")
print(f"  Holm (FWER) {holm_count}/6:            {'SOLID' if holm_count == 6 else 'PARTIAL'}")
print(f"  BH-FDR      {bh_count}/6:            {'SOLID' if bh_count == 6 else 'PARTIAL'}")

# 写汇总报告
lines = []
lines.append("P0 FINAL STATS REPORT (n=8)")
lines.append(f"Holm: {holm_count}/6  BH: {bh_count}/6")
for i, t in enumerate(tests):
    lines.append(f"{t['label']}: delta={t['delta_mean']:+.3f}+/-{t['delta_std']:.3f}  "
                 f"p={t['p']:.4f}{t['verdict']}  Holm={'*' if holm_rej[i] else 'ns'}  BH={'*' if bh_rej[i] else 'ns'}")
with open(f"{SAVE_DIR}\\p0_final_stats_n8.txt", 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print(f"\nSaved: p0_final_stats_n8.txt")
