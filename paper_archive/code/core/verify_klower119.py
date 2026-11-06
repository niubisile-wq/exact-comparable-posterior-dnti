import sys
sys.path.insert(0, r"<LOCAL_WORKSPACE>")
import re, pandapower as pp, networkx as nx

data_path = r"<LOCAL_WORKSPACE>\ElectricalSystemsDataForReconfiguration\SystemData_119.txt"
with open(data_path, encoding="utf-8") as f:
    content = f.read()

vnom = float(re.search(r"Vnominal\s*=\s*([\d.]+)", content).group(1))
bus_data = {}
in_bus = False
for line in content.split("\n"):
    s = line.strip()
    if re.match(r"Bus\s+PD", s): in_bus = True; continue
    if in_bus:
        nums = re.findall(r"[-+]?\d*\.?\d+", s)
        if len(nums) >= 3: bus_data[int(nums[0])] = (float(nums[1]), float(nums[2]))
        elif s == "": in_bus = False

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

# 用pandapower索引建图
G = nx.Graph()
for b in normal_br:
    G.add_edge(node2idx[b[0]], node2idx[b[1]])

# 找每条tie switch对应的回路
loops = []
for b in tie_br:
    u, v = node2idx[b[0]], node2idx[b[1]]
    try:
        path = nx.shortest_path(G, u, v)
        loops.append((u, v, path))
    except nx.NetworkXNoPath:
        pass

print(f"Tie switches: {len(tie_br)}")
print(f"Fundamental loops found: {len(loops)}")
print()

# 候选集合 K_lower=4: nodes [12, 30, 66, 101]
candidates = [12, 30, 66, 101]

print("Each loop and whether covered by candidates:")
covered = 0
uncovered_loops = []
for i, (u, v, path) in enumerate(loops):
    all_nodes_in_loop = set(path) | {u, v}
    hit = [c for c in candidates if c in all_nodes_in_loop]
    is_covered = len(hit) > 0
    if is_covered:
        covered += 1
    else:
        uncovered_loops.append(i+1)
    status = "OK" if is_covered else "MISS"
    print(f"  Loop {i+1:2d}: tie({u},{v})  path={path}  hit={hit}  [{status}]")

print()
print(f"Coverage: {covered}/{len(loops)} loops covered by {candidates}")
if not uncovered_loops:
    print("K_lower VERIFIED: all loops covered.")
else:
    print(f"FAIL: loops {uncovered_loops} not covered!")