import sys
sys.path.insert(0, r"<LOCAL_WORKSPACE>")
import importlib.util, networkx as nx

spec = importlib.util.spec_from_file_location("hk", r"<LOCAL_WORKSPACE>\dn_ip4_hk_curve.py")
mod = importlib.util.load_from_spec = None  # don't run __main__

# 直接exec只提取函数
with open(r"<LOCAL_WORKSPACE>\dn_ip4_hk_curve.py", encoding="utf-8") as f:
    src = f.read()

# 截取到主程序前
src_funcs = src[:src.find("print(\"IP4 H(K)")]
exec(src_funcs, glb := {})

net69, ne69, te69, n69 = glb["build_ieee69"]()
print(f"69-bus: {n69} buses, {len(ne69)} normal edges, {len(te69)} tie edges")

G69 = nx.Graph(); G69.add_edges_from(ne69)
loops69 = []
for u, v in te69:
    try:
        path = nx.shortest_path(G69, u, v)
        loops69.append(frozenset(path))
    except:
        pass
print(f"Loops: {len(loops69)}")

# K_lower=2 nodes [3,12] (0-indexed, display [4,13])
cands69 = {3, 12}
cov69 = sum(1 for ls in loops69 if ls & cands69)
print(f"nodes [3,12] (0-indexed, display [4,13]) covers {cov69}/{len(loops69)} loops")
if cov69 == len(loops69):
    print("69-bus K_lower=2 VERIFIED")
else:
    miss = [i+1 for i, ls in enumerate(loops69) if not ls & cands69]
    print(f"FAIL: loops {miss} not covered")