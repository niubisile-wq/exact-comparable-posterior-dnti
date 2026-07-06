import sys, re, itertools
sys.path.insert(0, r"<LOCAL_WORKSPACE>")
import pandapower as pp, networkx as nx

data_path = r"<LOCAL_WORKSPACE>\ElectricalSystemsDataForReconfiguration\SystemData_119.txt"
with open(data_path, encoding="utf-8") as f:
    content = f.read()
vnom = float(re.search(r"Vnominal\s*=\s*([\d.]+)", content).group(1))
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
all_nodes_list = sorted(set(n for b in normal_br+tie_br for n in (b[0],b[1])))
node2idx = {n: i for i, n in enumerate(all_nodes_list)}
G = nx.Graph()
for b in normal_br:
    G.add_edge(node2idx[b[0]], node2idx[b[1]])

# 每条基本回路包含的节点集合
loop_nodesets = []
for b in tie_br:
    u, v = node2idx[b[0]], node2idx[b[1]]
    try:
        path = nx.shortest_path(G, u, v)
        loop_nodesets.append(frozenset(path) | {u, v})
    except nx.NetworkXNoPath:
        pass

print(f"Loops to cover: {len(loop_nodesets)}")
n_nodes = len(all_nodes_list)
candidates = list(range(1, n_nodes))  # 排除slack 0

def covers_all(nodes_set):
    for ls in loop_nodesets:
        if not ls.intersection(nodes_set):
            return False
    return True

# 先检查K=1
for n in candidates:
    if covers_all({n}): print(f"K_lower=1: {n}"); break
else:
    print("K_lower>1")

# K=2
found2 = None
for combo in itertools.combinations(candidates, 2):
    if covers_all(set(combo)):
        found2 = combo; break
if found2: print(f"K_lower=2: {found2}")
else: print("K_lower>2")

# K=3
found3 = None
for combo in itertools.combinations(candidates, 3):
    if covers_all(set(combo)):
        found3 = combo; break
if found3: print(f"K_lower=3: {found3}")
else: print("K_lower>3")

# K=4
found4 = None
cnt = 0
for combo in itertools.combinations(candidates, 4):
    cnt += 1
    if cnt % 500000 == 0: print(f"  K=4 checked {cnt}...", flush=True)
    if covers_all(set(combo)):
        found4 = combo; break
if found4: print(f"K_lower=4: {found4}")
else: print(f"K_lower>4 (checked {cnt} combos)")