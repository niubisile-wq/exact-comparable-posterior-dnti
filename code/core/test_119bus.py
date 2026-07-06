import re, copy, time, warnings
import numpy as np
import pandapower as pp
import networkx as nx
warnings.filterwarnings("ignore")

# 解析SystemData_119.txt
path = r"<LOCAL_WORKSPACE>\ElectricalSystemsDataForReconfiguration\SystemData_119.txt"

with open(path, encoding="utf-8") as f:
    content = f.read()

# 找Vnominal
vnom = float(re.search(r"Vnominal\s*=\s*([\d.]+)", content).group(1))

# 解析Bus数据
bus_lines = []
in_bus = False
for line in content.split("\n"):
    line = line.strip()
    if "Bus" in line and "PD" in line: in_bus = True; continue
    if in_bus:
        nums = re.findall(r"[-+]?\d*\.?\d+", line)
        if len(nums) >= 3:
            bus_lines.append([int(nums[0]), float(nums[1]), float(nums[2])])
        elif line == "" and bus_lines: in_bus = False

# 解析Branch数据（FB, TB, #, R, X）
branch_lines = []
in_branch = False
for line in content.split("\n"):
    line = line.strip()
    if re.match(r"^\d+\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+", line):
        nums = re.findall(r"[-+]?\d*\.?\d+", line)
        if len(nums) >= 5:
            fb, tb, br_num = int(nums[0]), int(nums[1]), int(nums[2])
            r, x = float(nums[3]), float(nums[4])
            branch_lines.append((fb, tb, br_num, r, x))

# 分离正常支路和联络开关
n_buses = max(max(b[0] for b in branch_lines), max(b[1] for b in branch_lines)) + 1
n_normal = len(bus_lines)  # 大约等于bus数-1
# 找分界线：最后出现的支路编号跳跃点
br_nums = [b[2] for b in branch_lines]
max_normal = max(bus_lines, key=lambda x: x[0])[0]  # 总线数-1作为正常支路数估计

# 实际上：先排序，然后找gap
sorted_br = sorted(branch_lines, key=lambda x: x[2])
# 正常支路: br_num <= n_buses-1 (约等于bus数)
normal_br = [(b[0],b[1],b[3],b[4]) for b in sorted_br if b[2] <= 118]
tie_br    = [(b[0],b[1],b[3],b[4]) for b in sorted_br if b[2] >  118]

print(f"Vnominal: {vnom} kV")
print(f"Buses: {len(bus_lines)}")
print(f"Normal branches: {len(normal_br)}")
print(f"Tie switches: {len(tie_br)}")
print(f"Tie switch connections: {[(b[0],b[1]) for b in tie_br]}")

# 枚举有效拓扑（快速版，不跑潮流）
ne = [(b[0],b[1]) for b in normal_br]
te = [(b[0],b[1]) for b in tie_br]
N = max(max(e) for e in ne+te) + 1

G = nx.Graph(); G.add_edges_from(ne)
print(f"\nBase graph connected: {nx.is_connected(G)}")
print(f"N buses in graph: {G.number_of_nodes()}")

t0 = time.time()
topos = [list(range(len(ne)))]
seen = {frozenset(range(len(ne)))}

for ti2, tie in enumerate(te):
    if tie[0] not in G or tie[1] not in G: continue
    try: path = nx.shortest_path(G, tie[0], tie[1])
    except: continue
    for i in range(len(path)-1):
        oe = frozenset([path[i],path[i+1]])
        ni = [j for j,e in enumerate(ne) if frozenset(e)!=oe]
        key = frozenset(ni)
        if key in seen: continue
        edges = [ne[j] for j in ni]+[tie]
        Gt = nx.Graph(); Gt.add_nodes_from(range(N)); Gt.add_edges_from(edges)
        if nx.is_connected(Gt) and nx.is_tree(Gt):
            seen.add(key); topos.append(ni+[len(ne)+ti2])

print(f"Valid topologies: {len(topos)}  ({time.time()-t0:.1f}s)")
