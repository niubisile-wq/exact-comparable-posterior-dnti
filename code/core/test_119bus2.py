import re, time, warnings
import networkx as nx
warnings.filterwarnings("ignore")

path = r"<LOCAL_WORKSPACE>\ElectricalSystemsDataForReconfiguration\SystemData_119.txt"
with open(path, encoding="utf-8") as f:
    content = f.read()

# 解析所有支路
branch_lines = []
for line in content.split("\n"):
    line = line.strip()
    if re.match(r"^\d+\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+", line):
        nums = re.findall(r"[-+]?\d*\.?\d+", line)
        if len(nums) >= 5:
            branch_lines.append((int(nums[0]), int(nums[1]), int(nums[2]), float(nums[3]), float(nums[4])))

sorted_br = sorted(branch_lines, key=lambda x: x[2])
normal_br = [(b[0],b[1],b[3],b[4]) for b in sorted_br if b[2] <= 118]
tie_br    = [(b[0],b[1],b[3],b[4]) for b in sorted_br if b[2] >  118]

ne = [(b[0],b[1]) for b in normal_br]
te = [(b[0],b[1]) for b in tie_br]

# 只使用实际出现的节点（不用range(N)）
all_nodes = set()
for e in ne+te: all_nodes.update(e)
print(f"Normal: {len(ne)}, Tie: {len(te)}, Actual nodes: {len(all_nodes)}")

G = nx.Graph(); G.add_edges_from(ne)
print(f"Base graph connected: {nx.is_connected(G)}, nodes: {G.number_of_nodes()}")

t0 = time.time()
topos = [list(range(len(ne)))]
seen = {frozenset(range(len(ne)))}

for ti2, tie in enumerate(te):
    try: path_nodes = nx.shortest_path(G, tie[0], tie[1])
    except: print(f"  No path for tie {tie}"); continue
    for i in range(len(path_nodes)-1):
        oe = frozenset([path_nodes[i], path_nodes[i+1]])
        ni = [j for j,e in enumerate(ne) if frozenset(e) != oe]
        key = frozenset(ni)
        if key in seen: continue
        edges = [ne[j] for j in ni] + [tie]
        Gt = nx.Graph()
        Gt.add_edges_from(edges)  # 只加实际边，不加孤立节点！
        if nx.is_connected(Gt) and nx.is_tree(Gt):
            seen.add(key); topos.append(ni + [len(ne)+ti2])

elapsed = time.time()-t0
print(f"Valid topologies: {len(topos)}  ({elapsed:.1f}s)")

# 也查一下136-bus
path2 = r"<LOCAL_WORKSPACE>\ElectricalSystemsDataForReconfiguration\SystemData_136.txt"
with open(path2, encoding="utf-8") as f:
    c2 = f.read()
bl2 = []
for line in c2.split("\n"):
    line=line.strip()
    if re.match(r"^\d+\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+",line):
        nums=re.findall(r"[-+]?\d*\.?\d+",line)
        if len(nums)>=5: bl2.append((int(nums[0]),int(nums[1]),int(nums[2]),float(nums[3]),float(nums[4])))
sb2=sorted(bl2,key=lambda x:x[2])
ne2=[(b[0],b[1]) for b in sb2 if b[2]<=len([b for b in sb2 if b[2]<=200])-len([b for b in sb2 if b[2]>200 and b[2]<=220])]
# 简单一点：直接数
n_total=len(sb2)
print(f"\n136-bus total branches: {n_total}")
print(f"  First 5: {[(b[0],b[1],b[2]) for b in sb2[:5]]}")
print(f"  Last 10: {[(b[0],b[1],b[2]) for b in sb2[-10:]]}")
