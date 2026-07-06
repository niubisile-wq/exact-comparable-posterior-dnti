import re, copy, warnings
import numpy as np
import pandapower as pp
import networkx as nx
warnings.filterwarnings("ignore")

path = r"<LOCAL_WORKSPACE>\ElectricalSystemsDataForReconfiguration\SystemData_119.txt"
with open(path, encoding="utf-8") as f:
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
ne = [(node2idx[b[0]], node2idx[b[1]]) for b in normal_br]
te = [(node2idx[b[0]], node2idx[b[1]]) for b in tie_br]
substation_pp = node2idx[0]

def build_net():
    net = pp.create_empty_network()
    for n in all_nodes: pp.create_bus(net, vn_kv=vnom)
    for b in normal_br:
        pp.create_line_from_parameters(net, node2idx[b[0]], node2idx[b[1]], 1, max(b[3],0.0001), max(b[4],0.0001), 0, 9999, in_service=True)
    for b in tie_br:
        pp.create_line_from_parameters(net, node2idx[b[0]], node2idx[b[1]], 1, b[3], b[4], 0, 9999, in_service=False)
    for bus_net, (p, q) in bus_data.items():
        if (p > 0 or q > 0) and bus_net in node2idx:
            pp.create_load(net, node2idx[bus_net], p/1000, q/1000)
    pp.create_ext_grid(net, substation_pp, vm_pu=1.0)
    return net

net = build_net()
G = nx.Graph(); G.add_edges_from(ne)
topos=[list(range(len(ne)))]; seen={frozenset(range(len(ne)))}
for ti2, tie in enumerate(te):
    try: pn=nx.shortest_path(G,tie[0],tie[1])
    except: continue
    for i in range(len(pn)-1):
        oe=frozenset([pn[i],pn[i+1]])
        ni=[j for j,e in enumerate(ne) if frozenset(e)!=oe]
        key=frozenset(ni)
        if key in seen: continue
        Gt=nx.Graph(); Gt.add_edges_from([ne[j] for j in ni]+[tie])
        if nx.is_connected(Gt) and nx.is_tree(Gt):
            seen.add(key); topos.append(ni+[len(ne)+ti2])

def run_pf(net_base, topo_raw, tol=1e-4, maxiter=200):
    n = copy.deepcopy(net_base)
    n_ne=len(ne); act_ne={x for x in topo_raw if x<n_ne}; act_te={x-n_ne for x in topo_raw if x>=n_ne}
    for li in range(n_ne): n.line.at[n.line.index[li],"in_service"]=(li in act_ne)
    for li in range(len(te)): n.line.at[n.line.index[n_ne+li],"in_service"]=(li in act_te)
    try:
        pp.runpp(n,algorithm="bfsw",numba=False,max_iteration=maxiter,tolerance_mva=tol)
        if n.converged: return n.res_bus.vm_pu.values.copy()
    except: pass
    return None

results=[run_pf(net,t) for t in topos]
n_ok=sum(1 for r in results if r is not None)
V_ok=[r for r in results if r is not None]
print(f"Converged: {n_ok}/{len(topos)} ({n_ok/len(topos)*100:.1f}%)")
if V_ok:
    V_all=np.stack(V_ok)
    vmin_vals=V_all.min(axis=1)
    # 过滤掉V_min<0.7 pu的不现实拓扑
    realistic = sum(1 for v in vmin_vals if v >= 0.70)
    print(f"With V_min>=0.70 pu: {realistic}/{n_ok}")
    print(f"V_min stats: avg={vmin_vals.mean():.4f} min={vmin_vals.min():.4f} max={vmin_vals.max():.4f}")
    print(f"\nFINAL: IEEE 119-bus")
    print(f"  Total valid topos: {len(topos)}")
    print(f"  Convergence:       {n_ok}/{len(topos)} ({n_ok/len(topos)*100:.1f}%)")
    print(f"  Realistic (V>=0.7):{realistic}")
    verdict = "PASS" if n_ok/len(topos) >= 0.93 else "MARGINAL"
    print(f"  Verdict: {verdict}")
