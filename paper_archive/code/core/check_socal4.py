import json, glob, pandas as pd, networkx as nx
from itertools import combinations

# 读CB状态文件
cb_dir = r"<LOCAL_WORKSPACE>\digital-twin-dataset\sample_dataset\topology\parameter_timeseries\circuit3"
csvs = glob.glob(cb_dir + "\\*.csv")
print(f"CB status files found: {len(csvs)}")
for f in csvs[:3]:
    df = pd.read_csv(f)
    print(f"\n{f.split(chr(92))[-1]}:")
    print(df.to_string())

# 读网络JSON做图分析
path = r"<LOCAL_WORKSPACE>\digital-twin-dataset\sample_dataset\topology\network_files\circuit3\2023-08-01T00h00m00.000000s.json"
with open(path, encoding="utf-8") as f:
    d = json.load(f)

# 统计NC（闭合）的正常支路
print("\n\n=== Network Graph Analysis ===")
# 所有Line连接
G = nx.Graph()
for line in d["Line"]:
    f_bus = line["fbus"]
    for tb in line["tbus"]:
        G.add_edge(f_bus, tb["name"], etype="line")

# NC的Switch（正常闭合）
for sw in d["Switch"]:
    if sw.get("status","NC") == "NC":
        G.add_edge(sw["fbus"], sw["tbus"][0]["name"], etype="switch_nc")

print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
print(f"Connected: {nx.is_connected(G)}")
print(f"Components: {nx.number_connected_components(G)}")

# tie switch = SwitchMultiPosition里有NO的位置
tie_edges = []
for swmp in d["SwitchMultiPosition"]:
    for tb in swmp["tbus"]:
        if tb.get("status","") == "NO" or tb.get("nominal_status","") == "NO":
            tie_edges.append((swmp["fbus"], tb["name"]))
            break
print(f"\nTie switches (NO positions): {len(tie_edges)}")
for e in tie_edges:
    print(f"  {e[0]} -- {e[1]}")

# 计算基本回路和K_lower
print("\n=== Fundamental Loops ===")
loops = []
for f_bus, t_bus in tie_edges:
    try:
        path = nx.shortest_path(G, f_bus, t_bus)
        loops.append(set(path))
        print(f"  Loop via ({f_bus},{t_bus}): length={len(path)} nodes")
    except: print(f"  No path: {f_bus} -- {t_bus}")

print(f"\nTotal fundamental loops: {len(loops)}")

# 穷举K_lower
nodes = list(G.nodes())
k_lower = None
for k in range(1, len(loops)+1):
    found = False
    for subset in combinations(nodes, k):
        s = set(subset)
        if all(any(n in loop for n in s) for loop in loops):
            print(f"K_lower = {k} (e.g. {list(subset)[:3]}...)")
            k_lower = k; found = True; break
    if found: break
