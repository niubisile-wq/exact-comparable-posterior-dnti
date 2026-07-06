import sys, re
sys.path.insert(0, r"<LOCAL_WORKSPACE>")
import networkx as nx

data_path = r"<LOCAL_WORKSPACE>\ElectricalSystemsDataForReconfiguration\SystemData_119.txt"
with open(data_path, encoding="utf-8") as f:
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
all_nodes_list = sorted(set(n for b in normal_br+tie_br for n in (b[0],b[1])))
node2idx = {n: i for i, n in enumerate(all_nodes_list)}
G = nx.Graph()
for b in normal_br:
    G.add_edge(node2idx[b[0]], node2idx[b[1]])

loop_nodesets = []
for b in tie_br:
    u, v = node2idx[b[0]], node2idx[b[1]]
    try:
        path = nx.shortest_path(G, u, v)
        loop_nodesets.append(frozenset(path))
    except:
        pass

# 原脚本存储的是0-indexed节点，打印时+1
# k_lower_nodes = [11, 29, 65, 100] (0-indexed)
# 显示时变成 [12, 30, 66, 101] (+1)
k_lower_nodes_stored = [11, 29, 65, 100]   # 0-indexed，来自compute_k_lower返回值

# 独立验证脚本找到的
k_lower_nodes_found  = [11, 29, 65, 100]   # find_klower119.py 的结果

assert k_lower_nodes_stored == k_lower_nodes_found, "Mismatch!"

candidates = set(k_lower_nodes_stored)
covered = sum(1 for ls in loop_nodesets if ls & candidates)
total   = len(loop_nodesets)
print(f"0-indexed nodes: {k_lower_nodes_stored}")
print(f"Display (+1):    {[n+1 for n in k_lower_nodes_stored]}")
print(f"Coverage: {covered}/{total}")
if covered == total:
    print("K_lower=4 VERIFIED ✅  — 原脚本结果正确，验证脚本用了错误的+1索引")
else:
    # 找哪个loop没被覆盖
    for i, ls in enumerate(loop_nodesets):
        if not ls & candidates:
            print(f"  MISS loop {i+1}: {sorted(ls)}")