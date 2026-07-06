import re
import networkx as nx
from itertools import combinations

path = r"<LOCAL_WORKSPACE>\ElectricalSystemsDataForReconfiguration\SystemData_119.txt"
with open(path, encoding="utf-8") as f:
    content = f.read()
branch_lines = []
for line in content.split("\n"):
    s = line.strip()
    if re.match(r"^\d+\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+", s):
        nums = re.findall(r"[-+]?\d*\.?\d+", s)
        if len(nums) >= 5:
            branch_lines.append((int(nums[0]),int(nums[1]),int(nums[2]),float(nums[3]),float(nums[4])))
sorted_br = sorted(branch_lines, key=lambda x: x[2])
normal_br = [b for b in sorted_br if b[2] <= 118]
tie_br    = [b for b in sorted_br if b[2] > 118]
all_nodes = sorted(set(n for b in normal_br+tie_br for n in (b[0],b[1])))
node2idx = {n: i for i, n in enumerate(all_nodes)}
ne = [(node2idx[b[0]], node2idx[b[1]]) for b in normal_br]
te = [(node2idx[b[0]], node2idx[b[1]]) for b in tie_br]
n_bus = len(all_nodes)

# 完全复现compute_k_lower的内部逻辑，打印它看到的loops
G = nx.Graph(); G.add_edges_from(ne)
loops = [set(nx.shortest_path(G,t[0],t[1])) for t in te]
candidates = list(range(1, n_bus))

print(f"Loops computed inside compute_k_lower ({len(loops)} total):")
for i, loop in enumerate(loops):
    print(f"  Loop {i+1}: size={len(loop)}, nodes={sorted(loop)}")

# 测试[12,30,66,101]
s = {12, 30, 66, 101}
print(f"\nTesting {s}:")
for i, loop in enumerate(loops):
    cov = any(node in loop for node in s)
    found = [n for n in s if n in loop]
    print(f"  Loop {i+1}: {'COVERED'+str(found) if cov else 'NOT COVERED'}")
result = all(any(node in loop for node in s) for loop in loops)
print(f"  all() result: {result}")
