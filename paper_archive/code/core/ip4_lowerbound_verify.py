"""
IP4下界验证：比较固定负荷 vs 变负荷下的H(K)曲线
证明固定负荷H(K)是变负荷H(K)的下界
"""
import copy, warnings, time
import numpy as np
import pandapower as pp
import pandapower.networks as pn
warnings.filterwarnings("ignore")

SIGMA = 0.009
N_TEST = 500
LF_MIN, LF_MAX = 0.8, 1.2
np.random.seed(42)

net = pn.case33bw()
ne = [(int(r.from_bus), int(r.to_bus)) for _, r in net.line.iterrows() if r.in_service]
te = [(int(r.from_bus), int(r.to_bus)) for _, r in net.line.iterrows() if not r.in_service]
N_BUS = len(net.bus)

import networkx as nx
def enum_topos(ne, te, n=33):
    G = nx.Graph(); G.add_edges_from(ne)
    topos = [list(range(32))]; seen = {frozenset(range(32))}
    for ti2, tie in enumerate(te):
        path = nx.shortest_path(G, tie[0], tie[1])
        for i in range(len(path)-1):
            oe = frozenset([path[i], path[i+1]])
            ni = [j for j, e in enumerate(ne) if frozenset(e) != oe]
            key = frozenset(ni)
            if key in seen: continue
            edges = [ne[j] for j in ni] + [tie]
            Gt = nx.Graph(); Gt.add_nodes_from(range(n)); Gt.add_edges_from(edges)
            if nx.is_connected(Gt) and nx.is_tree(Gt):
                seen.add(key); topos.append(ni + [32+ti2])
    return topos

def run_pf_scaled(net_base, t_raw, lf=1.0):
    net = copy.deepcopy(net_base)
    net.load["p_mw"] = net.load["p_mw"] * lf
    net.load["q_mvar"] = net.load["q_mvar"] * lf
    ns = {x for x in t_raw if x < 32}; ts = {x-32 for x in t_raw if x >= 32}
    for li in range(37):
        active = (li in ns) if li < 32 else ((li-32) in ts)
        net.line.at[net.line.index[li], "in_service"] = active
    try:
        pp.runpp(net, algorithm="bfsw", numba=False, max_iteration=50, tolerance_mva=1e-8)
        if net.converged: return net.res_bus.vm_pu.values.copy()
    except: pass
    return None

topos_raw = enum_topos(ne, te)
N_TOPOS = len(topos_raw)

# 预计算固定负荷 V_fixed
print("Computing V_fixed (lf=1.0)...")
V_fixed = np.stack([run_pf_scaled(net, t, 1.0) for t in topos_raw])

# 预计算变负荷库
N_LF = 41
lf_grid = np.linspace(LF_MIN, LF_MAX, N_LF)
print(f"Computing V_library ({N_TOPOS}x{N_LF} runs)...")
V_lib = np.zeros((N_TOPOS, N_LF, N_BUS), dtype=np.float32)
for i, topo in enumerate(topos_raw):
    for j, lf in enumerate(lf_grid):
        V = run_pf_scaled(net, topo, lf)
        V_lib[i,j,:] = V if V is not None else (V_lib[i,max(j-1,0),:] if j>0 else 1.0)

def entropy(p): return -np.sum(p * np.log(p + 1e-10))

def compute_H_fixed(K, n_test=N_TEST):
    rng = np.random.RandomState(42)
    Hs = []
    for _ in range(n_test):
        ti = rng.randint(0, N_TOPOS)
        obs = np.sort(rng.choice(range(1, N_BUS), K, replace=False))
        obs_v = V_fixed[ti][obs] + rng.normal(0, SIGMA, K)
        diff = (V_fixed[:,obs] - obs_v) / SIGMA
        ll = -0.5*np.sum(diff**2, axis=1); ll -= ll.max()
        p = np.exp(ll); p /= p.sum()
        Hs.append(entropy(p))
    return np.mean(Hs)

def compute_H_variable(K, n_test=N_TEST):
    rng = np.random.RandomState(42)
    Hs = []
    for _ in range(n_test):
        ti = rng.randint(0, N_TOPOS)
        lf_idx = rng.randint(0, N_LF)
        obs = np.sort(rng.choice(range(1, N_BUS), K, replace=False))
        obs_v = V_lib[ti, lf_idx, obs] + rng.normal(0, SIGMA, K)
        # 已知lf条件下的精确后验
        diff = (V_lib[:, lf_idx, :][:, obs] - obs_v) / SIGMA
        ll = -0.5*np.sum(diff**2, axis=1); ll -= ll.max()
        p = np.exp(ll); p /= p.sum()
        Hs.append(entropy(p))
    return np.mean(Hs)

print("\nComputing H(K) for fixed vs variable load...")
print(f"{'K':>4}  {'H_fixed':>9}  {'H_variable':>11}  {'diff':>7}  {'lb_holds':>9}")
print("-"*50)
all_lb = True
for K in [5, 8, 12, 15, 20, 25]:
    hf = compute_H_fixed(K)
    hv = compute_H_variable(K)
    lb = hv >= hf
    if not lb: all_lb = False
    print(f"  {K:>2}  {hf:>9.4f}  {hv:>11.4f}  {hv-hf:>+7.4f}  {'YES' if lb else 'NO':>9}")

print()
if all_lb:
    print("CONCLUSION: H_variable(K) >= H_fixed(K) for all K tested.")
    print("IP4 fixed-load curve is a LOWER BOUND for variable-load scenario. [VERIFIED]")
else:
    print("WARNING: Lower bound property does NOT hold for some K values.")
    print("IP4 framing as lower bound is NOT valid.")