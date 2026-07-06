import sys
sys.path.insert(0, r"<LOCAL_WORKSPACE>")
import pandapower as pp, pandapower.networks as pn, networkx as nx, itertools

net = pn.case33bw()
G = nx.Graph()
for _, row in net.line.iterrows():
    if row.in_service:
        G.add_edge(int(row.from_bus), int(row.to_bus))
tie_edges = [(int(row.from_bus), int(row.to_bus))
             for _, row in net.line.iterrows() if not row.in_service]

print(f"Normal edges: {G.number_of_edges()}")
print(f"Tie edges: {len(tie_edges)}")

loops = []
for u, v in tie_edges:
    try:
        path = nx.shortest_path(G, u, v)
        loops.append(frozenset(path))
    except:
        pass
print(f"Loops: {len(loops)}")

# 验证 K_lower=2 nodes [2,8] (0-indexed，显示时+1=[3,9])
cands = {2, 8}
cov = sum(1 for ls in loops if ls & cands)
print(f"nodes [2,8] (0-indexed) covers {cov}/{len(loops)} loops")

# 也验证一下69-bus K_lower=2 nodes [3,12] (0-indexed，显示[4,13])
net69 = pn.case69()
G69 = nx.Graph()
for _, row in net69.line.iterrows():
    if row.in_service:
        G69.add_edge(int(row.from_bus), int(row.to_bus))
tie69 = [(int(row.from_bus), int(row.to_bus))
         for _, row in net69.line.iterrows() if not row.in_service]
loops69 = []
for u, v in tie69:
    try:
        path = nx.shortest_path(G69, u, v)
        loops69.append(frozenset(path))
    except:
        pass
cands69 = {3, 12}
cov69 = sum(1 for ls in loops69 if ls & cands69)
print(f"69-bus nodes [3,12] (0-indexed) covers {cov69}/{len(loops69)} loops")