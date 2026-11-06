import re
import networkx as nx

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

G = nx.Graph(); G.add_edges_from(ne)
loops = []
for tie in te:
    try:
        pn = nx.shortest_path(G, tie[0], tie[1])
        loops.append(set(pn))
    except:
        loops.append(set())

klower_nodes = [12, 30, 66, 101]
print(f"K_lower nodes (pp idx): {klower_nodes}")
print(f"K_lower nodes (orig):   {[idx2node[n] for n in klower_nodes]}")
print()

for i, (tie, loop) in enumerate(zip(te, loops)):
    orig = (idx2node[tie[0]], idx2node[tie[1]])
    cov = any(n in loop for n in klower_nodes)
    found = [n for n in klower_nodes if n in loop]
    status = f"COVERED by {found}" if cov else "NOT COVERED"
    print(f"Loop {i+1:2d}: tie(orig {orig[0]:3d}-{orig[1]:3d}) "
          f"pp({tie[0]:3d}-{tie[1]:3d})  len={len(loop):2d}  {status}")
    if not cov:
        print(f"         Full path (sorted): {sorted(loop)}")

all_covered = all(any(n in loop for n in klower_nodes) for loop in loops)
print(f"\nOverall: {'ALL COVERED' if all_covered else 'FAILS'}")
