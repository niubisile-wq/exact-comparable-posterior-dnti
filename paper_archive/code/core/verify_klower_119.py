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
idx2node = {i: n for n, i in node2idx.items()}

ne = [(node2idx[b[0]], node2idx[b[1]]) for b in normal_br]
te = [(node2idx[b[0]], node2idx[b[1]]) for b in tie_br]

# 计算所有15个基本回路的节点集合
G = nx.Graph(); G.add_edges_from(ne)
loops = []
for tie in te:
    try:
        path_nodes = nx.shortest_path(G, tie[0], tie[1])
        loops.append(set(path_nodes))
    except:
        loops.append(set())

print(f"Tie switches: {len(te)}")
print(f"Fundamental loops found: {len([l for l in loops if l])}")
for i, (tie, loop) in enumerate(zip(te, loops)):
    orig = (idx2node[tie[0]], idx2node[tie[1]])
    print(f"  Loop {i+1}: tie{orig}, path length={len(loop)} nodes")

# 验证K_lower=4节点[12,30,66,101]是否覆盖所有15个回路
klower_nodes = [12, 30, 66, 101]  # pandapower indices
print(f"\nVerifying K_lower nodes {klower_nodes} (original: {[idx2node[n] for n in klower_nodes]}):")
covered = []
for i, loop in enumerate(loops):
    cov = any(n in loop for n in klower_nodes)
    covered.append(cov)
    if not cov:
        print(f"  Loop {i+1} NOT covered! tie={te[i]}, loop nodes={sorted(loop)[:5]}...")

if all(covered):
    print(f"  ALL {len(loops)} loops covered. K_lower=4 VERIFIED.")
else:
    print(f"  FAILED: {sum(covered)}/{len(loops)} loops covered")

# 验证不存在3个节点能覆盖所有回路（K_lower确实=4不是3）
print("\nVerifying no 3 nodes can cover all loops...")
candidates = list(range(1, len(all_nodes)))
found3 = False
for subset in combinations(candidates, 3):
    s = set(subset)
    if all(any(n in loop for n in s) for loop in loops if loop):
        print(f"  Found 3-node cover: {subset} -> K_lower should be 3, not 4!")
        found3 = True; break
if not found3:
    print(f"  Confirmed: no 3-node cover exists. K_lower=4 is exact minimum.")
