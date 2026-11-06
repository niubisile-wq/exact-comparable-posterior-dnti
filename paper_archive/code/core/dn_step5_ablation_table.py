# -*- coding: utf-8 -*-
"""
Step5-C: 消融表整理 + 论文用汇总表
把所有已有实验结果整理成论文格式
需在DNN和Fisher-OED跑完后执行
"""
import os
SAVE_DIR = r"<LOCAL_WORKSPACE>"

print("="*70)
print("STEP5 ABLATION & SUMMARY TABLES")
print("="*70)

# ── 1. IP1消融：不同方法的推断速度和精度 ─────────────────────────────────
print("\n【Table 1: IP1 推断方法对比（33-bus & 69-bus）】")
print(f"  {'方法':<25}  {'33-bus top-1':>13}  {'69-bus top-1':>13}  {'速度':>12}")
print("  " + "-"*65)
rows_t1 = [
    ("AIS (Xu 2021)",        "72.1%±0.0%",  "40.2%±0.0%", "1/1887ms×"),
    ("DNN-fixed-lf (基线)",   "见step5_dnn", "见step5_dnn", "~1ms"),
    ("NRE-variable-lf (IP1)","68.2%±0.8%",  "35.9%±1.0%", "~1ms (1176×/3302×)"),
]
for r in rows_t1:
    print(f"  {r[0]:<25}  {r[1]:>13}  {r[2]:>13}  {r[3]:>12}")
print("  注：DNN-fixed-lf数字待step5_dnn_result.txt生成后填入")

# ── 2. IP-A消融：传感器选址策略对比 ────────────────────────────────────
print("\n【Table 2: IP-A 传感器选址对比（K=4和K=7）】")
print(f"  {'策略':<20}  {'33bus K=4':>10}  {'33bus K=7':>10}  {'69bus K=4':>10}  {'69bus K=7':>10}")
print("  " + "-"*65)
rows_t2 = [
    ("Random",      "29.4%", "41.8%", "19.0%", "24.0%"),
    ("GreedyLoop",  "29.0%", "36.4%", "15.5%", "24.0%"),
    ("Fisher-OED",  "见fisher", "见fisher", "见fisher", "见fisher"),
    ("BOED (IP-A)", "57.4%", "75.4%", "39.5%", "67.0%"),
]
for r in rows_t2:
    print(f"  {r[0]:<20}  {r[1]:>10}  {r[2]:>10}  {r[3]:>10}  {r[4]:>10}")
print("  注：Fisher-OED数字待step5_fisher_result.txt生成后填入")

# ── 3. IP-C消融：量测缺失下的鲁棒性 ────────────────────────────────────
print("\n【Table 3: IP-C 通信缺失鲁棒性（33-bus，5种子）】")
print(f"  {'方法':<20}  {'miss=0%':>8}  {'miss=10%':>9}  {'miss=30%':>9}  {'p@30%':>8}")
print("  " + "-"*60)
rows_t3 = [
    ("朴素NRE",    "67.0%", "40.1%",  "10.1%",  "-"),
    ("鲁棒NRE(IP-C)","68.7%±0.4%","65.3%±0.6%","61.3%±0.5%","0.031*"),
    ("Delta",      "+1.7pp","**+25.2pp**","**+51.2pp**","Wilcoxon"),
]
for r in rows_t3:
    print(f"  {r[0]:<20}  {r[1]:>8}  {r[2]:>9}  {r[3]:>9}  {r[4]:>8}")

# ── 4. IP4消融：可识别性理论分析 ────────────────────────────────────────
print("\n【Table 4: IP4 贝叶斯可识别性（H(K)，K=8时）】")
print(f"  {'网络':<15}  {'H(K=8)':>8}  {'top-1(K=8)':>12}  {'K_lower':>8}")
print("  " + "-"*48)
rows_t4 = [
    ("33-bus (32拓扑)",  "1.184", "52.7%", "2"),
    ("69-bus (61拓扑)",  "2.105", "27.0%", "2"),
    ("119-bus (100拓扑)","2.610", "30.2%", "4"),
]
for r in rows_t4:
    print(f"  {r[0]:<15}  {r[1]:>8}  {r[2]:>12}  {r[3]:>8}")
print("  H(K)>1.0 表示多峰后验，点估计可靠性不足 → IP1全后验推断的必要性")

# ── 5. 统计检验汇总 ─────────────────────────────────────────────────────
print("\n【Table 5: Wilcoxon显著性检验（Step4结果）】")
print(f"  {'比较':<40}  {'Delta均值':>10}  {'p值':>8}  {'sig':>4}  {'n':>3}")
print("  " + "-"*68)
rows_t5 = [
    ("IP1-33bus: AIS vs NRE gap",          "+3.9pp±0.8pp", "0.0312", "*",  "5"),
    ("IP1-69bus: AIS vs NRE gap",          "+4.3pp±1.0pp", "0.0312", "*",  "5"),
    ("IP-C-33bus: robust>naive @miss=10%", "+25.2pp±0.6pp","0.0312", "*",  "5"),
    ("IP-C-33bus: robust>naive @miss=30%", "+51.2pp±0.5pp","0.0312", "*",  "5"),
    ("IP-C-69bus: robust>naive @miss=10%", "+12.3pp±0.7pp","0.1250", "ns", "3"),
    ("IP-C-69bus: robust>naive @miss=30%", "+23.6pp±0.6pp","0.1250", "ns", "3"),
]
for r in rows_t5:
    print(f"  {r[0]:<40}  {r[1]:>10}  {r[2]:>8}  {r[3]:>4}  {r[4]:>3}")
print("  IP-C 69-bus n=3,Wilcoxon最小p=0.125,在论文中主动声明检验力限制")

# ── 检查是否已有DNN和Fisher结果 ─────────────────────────────────────────
print(f"\n{'='*70}")
missing = []
for fname in ['step5_dnn_result.txt', 'step5_fisher_result.txt']:
    path = f"{SAVE_DIR}\\{fname}"
    if os.path.exists(path):
        print(f"[OK] {fname} 已存在，读取数字...")
        with open(path, encoding='utf-8') as f:
            print("  " + f.read().replace('\n', '\n  '))
    else:
        missing.append(fname)
        print(f"[缺] {fname} 未找到，请先运行对应脚本")

if not missing:
    print("\n所有Step5数据就位，可以写论文了。")
else:
    print(f"\n还需要运行：{missing}")
print("="*70)

# 保存消融表
with open(f"{SAVE_DIR}\\step5_ablation_tables.txt", 'w', encoding='utf-8') as f:
    f.write("Step5 Ablation Tables - placeholder, fill DNN and Fisher numbers after experiments\n\n")
    f.write("Table3 IP-C 33-bus (complete):\n")
    for r in rows_t3:
        f.write(f"  {r[0]}: miss0={r[1]} miss10={r[2]} miss30={r[3]}\n")
    f.write("\nTable5 Wilcoxon (complete):\n")
    for r in rows_t5:
        f.write(f"  {r[0]}: delta={r[1]} p={r[2]} {r[3]} n={r[4]}\n")
print(f"Saved: {SAVE_DIR}\\step5_ablation_tables.txt")
