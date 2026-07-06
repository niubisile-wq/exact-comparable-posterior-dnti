import re, copy, time, warnings
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
        if len(nums) >= 3:
            bus_data[int(nums[0])] = (float(nums[1]), float(nums[2]))
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
    for n in all_nodes:
        pp.create_bus(net, vn_kv=vnom)
    for b in normal_br:
        r = max(b[3], 0.0001)  # 修复零阻抗
        x = max(b[4], 0.0001)
        pp.create_line_from_parameters(net, node2idx[b[0]], node2idx[b[1]], 1, r, x, 0, 9999, in_service=True)
    for b in tie_br:
        pp.create_line_from_parameters(net, node2idx[b[0]], node2idx[b[1]], 1, b[3], b[4], 0, 9999, in_service=False)
    for bus_net, (p, q) in bus_data.items():
        if (p > 0 or q > 0) and bus_net in node2idx:
            pp.create_load(net, node2idx[bus_net], p/1000, q/1000)
    pp.create_ext_grid(net, substation_pp, vm_pu=1.0)
    return net

print("Building 119-bus model...")
net = build_net()
print(f"  Buses:{len(net.bus)}, Lines:{len(net.line)}, Loads:{len(net.load)}")
print(f"  Total P={net.load.p_mw.sum()*1000:.0f}kW")

pp.runpp(net, algorithm="bfsw", numba=False, max_iteration=100, tolerance_mva=1e-6)
if net.converged:
    vm = net.res_bus.vm_pu.values
    loss = net.res_line.pl_mw.sum()*1000
    print(f"  Base PF: V_min={vm.min():.4f} V_max={vm.max():.4f} loss={loss:.1f}kW  CONVERGED")
else:
    print("  Base PF FAILED"); exit(1)

# 枚举拓扑
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
print(f"  Valid topologies: {len(topos)}")

# 全部潮流验证
def run_pf(net_base, topo_raw):
    net=copy.deepcopy(net_base)
    n_ne=len(ne)
    act_ne={x for x in topo_raw if x<n_ne}
    act_te={x-n_ne for x in topo_raw if x>=n_ne}
    for li in range(n_ne): net.line.at[net.line.index[li],"in_service"]=(li in act_ne)
    for li in range(len(te)): net.line.at[net.line.index[n_ne+li],"in_service"]=(li in act_te)
    try:
        pp.runpp(net,algorithm="bfsw",numba=False,max_iteration=100,tolerance_mva=1e-6)
        if net.converged: return net.res_bus.vm_pu.values.copy()
    except: pass
    return None

print(f"Testing all {len(topos)} topologies...")
t0=time.time()
results=[run_pf(net,t) for t in topos]
n_ok=sum(1 for r in results if r is not None)
print(f"  Converged: {n_ok}/{len(topos)} ({time.time()-t0:.1f}s)")
if n_ok > 0:
    V_all=np.stack([r for r in results if r is not None])
    print(f"  V_min: avg={V_all.min(axis=1).mean():.4f} worst={V_all.min():.4f}")
    print(f"  VERDICT: {'PASS' if n_ok/len(topos)>=0.95 else 'MARGINAL ('+str(round(n_ok/len(topos)*100))+'%)'}")
