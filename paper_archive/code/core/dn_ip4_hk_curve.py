# -*- coding: utf-8 -*-
"""
阶段1 Step 1：IP4 贝叶斯可识别性曲线实验
  - H(K)：K个智能电表下，后验熵均值
  - N_sig(K)：后验概率>5%的拓扑数均值
  - K_lower：理论下界（最少节点数使每个基本回路都被覆盖）
两个网络：IEEE 33-bus（32种拓扑）+ IEEE 69-bus（61种拓扑）
参数：N_TEST=200 个随机测试场景，K从1扫到N_bus-1
"""
import copy, time, warnings
import numpy as np
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')

OUT_PATH = r"<LOCAL_WORKSPACE>\ip4_hk_result_v5.txt"

# ════════════════════════════════════════════════════════════════════════
# 网络构建函数
# ════════════════════════════════════════════════════════════════════════

def build_ieee33():
    net = pp.create_empty_network()
    for i in range(33): pp.create_bus(net, vn_kv=12.66)
    br = [(0,1,0.0922,0.0470),(1,2,0.4930,0.2511),(2,3,0.3660,0.1864),
          (3,4,0.3811,0.1941),(4,5,0.8190,0.7070),(5,6,0.1872,0.6188),
          (6,7,0.7114,0.2351),(7,8,1.0300,0.7400),(8,9,1.0440,0.7400),
          (9,10,0.1966,0.0650),(10,11,0.3744,0.1238),(11,12,1.4680,1.1550),
          (12,13,0.5416,0.7129),(13,14,0.5910,0.5260),(14,15,0.7463,0.5450),
          (15,16,1.2890,1.7210),(16,17,0.7320,0.5740),(1,18,0.1640,0.1565),
          (18,19,1.5042,1.3554),(19,20,0.4095,0.4784),(20,21,0.7089,0.9373),
          (2,22,0.4512,0.3083),(22,23,0.8980,0.7091),(23,24,0.8960,0.7011),
          (5,25,0.2030,0.1034),(25,26,0.2842,0.1447),(26,27,1.0590,0.9337),
          (27,28,0.8042,0.7006),(28,29,0.5075,0.2585),(29,30,0.9744,0.9630),
          (30,31,0.3105,0.3619),(31,32,0.3410,0.5302)]
    ti = [(7,20,0.089,0.089),(8,14,0.059,0.059),
          (11,21,0.089,0.089),(17,32,0.038,0.085),(24,28,0.056,0.065)]
    for f,t,r,x in br:
        pp.create_line_from_parameters(net,f,t,1,r,x,0,1,in_service=True)
    for f,t,r,x in ti:
        pp.create_line_from_parameters(net,f,t,1,r,x,0,1,in_service=False)
    ld = [(1,100,60),(2,90,40),(3,120,80),(4,60,30),(5,60,20),(6,200,100),
          (7,200,100),(8,60,20),(9,60,20),(10,45,30),(11,60,35),(12,60,35),
          (13,120,80),(14,60,10),(15,60,20),(16,60,20),(17,90,40),(18,90,40),
          (19,90,40),(20,90,40),(21,90,40),(22,90,50),(23,420,200),(24,420,200),
          (25,60,25),(26,60,25),(27,60,20),(28,120,70),(29,200,600),
          (30,150,70),(31,210,100),(32,60,40)]
    for b,p,q in ld: pp.create_load(net,b,p/1000,q/1000)
    pp.create_ext_grid(net,0,vm_pu=1.0)
    ne = [(int(f),int(t)) for f,t,r,x in br]
    te = [(int(f),int(t)) for f,t,r,x in ti]
    return net, ne, te, 33

def build_ieee69():
    net = pp.create_empty_network()
    for i in range(69): pp.create_bus(net, vn_kv=12.66)
    branch_data = [
        (1,2,0.0005,0.0012,0,0),(2,3,0.0005,0.0012,0,0),
        (3,4,0.0015,0.0036,0,0),(4,5,0.0251,0.0294,0,0),
        (5,6,0.3660,0.1864,2.6,2.2),(6,7,0.3811,0.1941,40.4,30.0),
        (7,8,0.0922,0.0470,75.0,54.0),(8,9,0.0493,0.0251,30.0,22.0),
        (9,10,0.8190,0.2707,28.0,19.0),(10,11,0.1872,0.0619,145.0,104.0),
        (11,12,0.7114,0.2351,145.0,104.0),(12,13,1.0300,0.3400,8.0,5.5),
        (13,14,1.0440,0.3450,8.0,5.5),(14,15,1.0580,0.3496,0.0,0.0),
        (15,16,0.1966,0.0650,45.5,30.0),(16,17,0.3744,0.1238,60.0,35.0),
        (17,18,0.0047,0.0016,60.0,35.0),(18,19,0.3276,0.1083,0.0,0.0),
        (19,20,0.2106,0.0690,1.0,0.6),(20,21,0.3416,0.1129,114.0,81.0),
        (21,22,0.0140,0.0046,5.3,3.5),(22,23,0.1591,0.0526,0.0,0.0),
        (23,24,0.3463,0.1145,28.0,20.0),(24,25,0.7488,0.2475,0.0,0.0),
        (25,26,0.3089,0.1021,14.0,10.0),(26,27,0.1732,0.0572,14.0,10.0),
        (3,28,0.0044,0.0108,26.0,18.6),(28,29,0.0640,0.1565,26.0,18.6),
        (29,30,0.3978,0.1315,0.0,0.0),(30,31,0.0702,0.0232,0.0,0.0),
        (31,32,0.3510,0.1160,0.0,0.0),(32,33,0.8390,0.2816,14.0,10.0),
        (33,34,1.7080,0.5646,19.5,14.0),(34,35,1.4740,0.4873,6.0,4.0),
        (35,36,0.0044,0.0108,26.0,18.6),(36,37,0.0640,0.1565,26.0,18.6),
        (37,38,0.1053,0.1230,0.0,0.0),(38,39,0.0304,0.0355,24.0,17.0),
        (39,40,0.0018,0.0021,24.0,17.0),(40,41,0.7283,0.8509,1.2,1.0),
        (41,42,0.3100,0.3623,0.0,0.0),(42,43,0.0410,0.0478,6.0,4.3),
        (43,44,0.0092,0.0116,0.0,0.0),(44,45,0.1089,0.1373,39.2,26.3),
        (45,46,0.0009,0.0012,39.2,26.3),(4,47,0.0034,0.0084,0.0,0.0),
        (47,48,0.0851,0.2083,79.0,56.4),(48,49,0.2898,0.7091,384.7,274.5),
        (49,50,0.0822,0.2011,384.7,274.5),(8,51,0.0928,0.0473,40.5,28.3),
        (51,52,0.3319,0.1114,3.6,2.7),(9,53,0.1740,0.0886,4.35,3.5),
        (53,54,0.2030,0.1034,26.4,19.0),(54,55,0.2842,0.1447,24.0,17.2),
        (55,56,0.2813,0.1433,0.0,0.0),(56,57,1.5900,0.5337,0.0,0.0),
        (57,58,0.7837,0.2630,0.0,0.0),(58,59,0.3042,0.1006,100.0,72.0),
        (59,60,0.3861,0.1172,0.0,0.0),(60,61,0.5075,0.2585,1244.0,888.0),
        (61,62,0.0974,0.0496,32.0,23.0),(62,63,0.1450,0.0738,0.0,0.0),
        (63,64,0.7105,0.3619,227.0,162.0),(64,65,1.0410,0.5302,59.0,42.0),
        (11,66,0.2012,0.0611,18.0,13.0),(66,67,0.0047,0.0014,18.0,13.0),
        (12,68,0.7394,0.2444,28.0,20.0),(68,69,0.0047,0.0014,28.0,20.0),
    ]
    tie_data = [
        (11,43,0.5,0.5),(13,21,0.5,0.5),(15,46,0.5,0.5),
        (50,59,0.5,0.5),(27,65,0.5,0.5),
    ]
    ne, te = [], []
    for f1,t1,r,x,p,q in branch_data:
        f0,t0=f1-1,t1-1
        pp.create_line_from_parameters(net,f0,t0,1,r,x,0,9999,in_service=True)
        if p>0 or q>0: pp.create_load(net,t0,p/1000,q/1000)
        ne.append((f0,t0))
    for f1,t1,r,x in tie_data:
        f0,t0=f1-1,t1-1
        pp.create_line_from_parameters(net,f0,t0,1,r,x,0,9999,in_service=False)
        te.append((f0,t0))
    pp.create_ext_grid(net,0,vm_pu=1.0)
    return net, ne, te, 69

# ════════════════════════════════════════════════════════════════════════
# 通用函数
# ════════════════════════════════════════════════════════════════════════

def enum_topos(ne, te, n):
    G = nx.Graph(); G.add_edges_from(ne)
    topos=[list(range(len(ne)))]; seen={frozenset(range(len(ne)))}
    for ti2,tie in enumerate(te):
        try: path=nx.shortest_path(G,tie[0],tie[1])
        except: continue
        for i in range(len(path)-1):
            oe=frozenset([path[i],path[i+1]])
            ni=[j for j,e in enumerate(ne) if frozenset(e)!=oe]
            key=frozenset(ni)
            if key in seen: continue
            edges=[ne[j] for j in ni]+[tie]
            Gt=nx.Graph(); Gt.add_nodes_from(range(n)); Gt.add_edges_from(edges)
            if nx.is_connected(Gt) and nx.is_tree(Gt):
                seen.add(key); topos.append(ni+[len(ne)+ti2])
    return topos

def run_pf(net_base, topo_raw, ne, te):
    net=copy.deepcopy(net_base)
    n_ne=len(ne)
    act_ne={x for x in topo_raw if x<n_ne}
    act_te={x-n_ne for x in topo_raw if x>=n_ne}
    for li in range(n_ne):
        net.line.at[net.line.index[li],'in_service']=(li in act_ne)
    for li in range(len(te)):
        net.line.at[net.line.index[n_ne+li],'in_service']=(li in act_te)
    try:
        pp.runpp(net,algorithm='bfsw',numba=False,max_iteration=100,tolerance_mva=1e-6)
        if net.converged: return net.res_bus.vm_pu.values.copy()
    except: pass
    return None

def ais_posterior(obs_nodes, obs_vals, V_all, sigma=0.009):
    if len(obs_nodes)==0:
        return np.ones(len(V_all))/len(V_all)
    diff=(V_all[:,obs_nodes]-obs_vals)/sigma
    ll=-0.5*np.sum(diff**2,axis=1); ll-=ll.max()
    w=np.exp(ll); return w/w.sum()

def entropy(p):
    p=np.clip(p,1e-10,1.0); return -np.sum(p*np.log(p))

def n_significant(p, thresh=0.05):
    return int(np.sum(p>thresh))

def compute_k_lower(ne, te, n):
    """穷举最小集覆盖：最小节点集使每个基本回路至少有1个节点（精确解）"""
    from itertools import combinations
    G=nx.Graph(); G.add_edges_from(ne)
    loops=[set(nx.shortest_path(G,t[0],t[1])) for t in te]
    candidates=list(range(1,n))  # 排除slack bus 0
    n_loops=len(loops)
    for k in range(1, n_loops+1):
        for subset in combinations(candidates, k):
            s=set(subset)
            if all(any(node in loop for node in s) for loop in loops):
                return k, list(subset)
    return n_loops, list(range(1,n_loops+1))

# ════════════════════════════════════════════════════════════════════════
# H(K) 曲线计算
# ════════════════════════════════════════════════════════════════════════

def compute_hk_curve(name, net, ne, te, n_bus, N_TEST=1000, sigma=0.009):
    print(f"\n{'='*60}")
    print(f"Network: {name}")
    print(f"{'='*60}")

    # 枚举拓扑
    print("Enumerating topologies...")
    topos=enum_topos(ne,te,n_bus)
    N_TOPOS=len(topos)
    print(f"  Valid topologies: {N_TOPOS}")

    # 计算所有拓扑电压
    print("Computing voltage profiles...")
    t0=time.time()
    voltages=[run_pf(net,t,ne,te) for t in topos]
    V_all=np.stack([v for v in voltages if v is not None])
    n_ok=sum(1 for v in voltages if v is not None)
    print(f"  Converged: {n_ok}/{N_TOPOS}  ({time.time()-t0:.1f}s)")
    N_TOPOS=len(V_all)  # 只用收敛的

    # 理论下界 K_lower
    k_lower, k_lower_nodes = compute_k_lower(ne, te, n_bus)
    print(f"  K_lower={k_lower} (nodes: {[n+1 for n in k_lower_nodes]})")

    # K扫描范围（采样到覆盖N_bus-1个节点）
    max_K=min(n_bus-1, 30)  # 最多扫30个，避免计算量过大
    K_vals=list(range(1, max_K+1))

    rng=np.random.RandomState(2026)
    candidates=list(range(1,n_bus))  # 排除slack bus 0

    H_mean=[]
    H_std=[]
    N_sig_mean=[]
    top1_acc=[]

    print(f"  Computing H(K) for K=1..{max_K}, N_TEST={N_TEST}...")
    print(f"  Method: sequential node addition per trial (DPI guarantees monotone)")
    print(f"  sigma={sigma} (same for obs and posterior, no mismatch)")
    t0=time.time()

    # 累积式：每个trial随机排列节点，依次累加——同一序列DPI保证单调
    all_H    = np.zeros((N_TEST, max_K))
    all_acc  = np.zeros((N_TEST, max_K))
    all_nsig = np.zeros((N_TEST, max_K))

    for trial in range(N_TEST):
        ti   = rng.randint(0, N_TOPOS)
        perm = rng.permutation(candidates)
        obs_nodes_seq = []
        obs_vals_seq  = []
        for k_idx in range(max_K):
            node = int(perm[k_idx])
            obs_nodes_seq.append(node)
            obs_vals_seq.append(float(V_all[ti][node] + rng.normal(0, sigma)))
            p = ais_posterior(obs_nodes_seq, np.array(obs_vals_seq), V_all, sigma)
            all_H[trial, k_idx]    = entropy(p)
            all_acc[trial, k_idx]  = int(np.argmax(p) == ti)
            all_nsig[trial, k_idx] = n_significant(p)
        if trial % 200 == 0:
            print(f"    trial {trial}/{N_TEST}  ({time.time()-t0:.0f}s)")

    H_mean    = list(all_H.mean(axis=0))
    H_std     = list(all_H.std(axis=0))
    top1_acc  = list(all_acc.mean(axis=0))
    N_sig_mean= list(all_nsig.mean(axis=0))

    for K in [1, 3, 5, 8, 12, 15, 20, 25, 30]:
        if K <= max_K:
            idx = K-1
            print(f"    K={K:2d}: H={H_mean[idx]:.3f}+-{H_std[idx]:.3f}  "
                  f"N_sig={N_sig_mean[idx]:.1f}  top1={top1_acc[idx]:.2f}"
                  f"  ({time.time()-t0:.0f}s)")

    # 单调性检验（强制输出警告，不修改数据）
    violations = [(K_vals[i], H_mean[i], H_mean[i+1])
                  for i in range(len(H_mean)-1) if H_mean[i+1] > H_mean[i] + 0.02]
    if violations:
        print(f"\n  [WARNING] Non-monotone H(K) at: "
              + ", ".join(f"K={k}({h1:.3f})->K={k+1}({h2:.3f})" for k,h1,h2 in violations))
        print(f"  N_TEST={N_TEST} may still be too small at these K values")
    else:
        print(f"\n  [OK] H(K) is monotone decreasing — curve is clean")

    return {
        'name': name,
        'N_TOPOS': N_TOPOS,
        'n_bus': n_bus,
        'k_lower': k_lower,
        'k_lower_nodes': k_lower_nodes,
        'K_vals': K_vals,
        'H_mean': H_mean,
        'H_std': H_std,
        'N_sig_mean': N_sig_mean,
        'top1_acc': top1_acc,
        'monotone': len(violations) == 0,
    }

def build_ieee119():
    """IEEE 119-bus，来自IEEE DataPort DOI:10.21227/m49t-q808"""
    import re
    data_path = r"<LOCAL_WORKSPACE>\ElectricalSystemsDataForReconfiguration\SystemData_119.txt"
    with open(data_path, encoding="utf-8") as f:
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
    net = pp.create_empty_network()
    for n in all_nodes: pp.create_bus(net, vn_kv=vnom)
    for b in normal_br:
        pp.create_line_from_parameters(net, node2idx[b[0]], node2idx[b[1]], 1,
                                       max(b[3],0.0001), max(b[4],0.0001), 0, 9999, in_service=True)
    for b in tie_br:
        pp.create_line_from_parameters(net, node2idx[b[0]], node2idx[b[1]], 1,
                                       b[3], b[4], 0, 9999, in_service=False)
    for bus_net, (p, q) in bus_data.items():
        if (p > 0 or q > 0) and bus_net in node2idx:
            pp.create_load(net, node2idx[bus_net], p/1000, q/1000)
    pp.create_ext_grid(net, node2idx[0], vm_pu=1.0)
    ne = [(node2idx[b[0]], node2idx[b[1]]) for b in normal_br]
    te = [(node2idx[b[0]], node2idx[b[1]]) for b in tie_br]
    return net, ne, te, len(all_nodes)

# ════════════════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════════════════
print("IP4 H(K) Identifiability Curve Experiment")
print("Three networks: IEEE 33-bus + IEEE 69-bus + IEEE 119-bus")
print("="*60)

results = []

# IEEE 33-bus
print("\n[1/3] Building IEEE 33-bus...")
net33, ne33, te33, n33 = build_ieee33()
r33 = compute_hk_curve("IEEE 33-bus (32 topos)", net33, ne33, te33, n33, N_TEST=1000)
results.append(r33)

# IEEE 69-bus
print("\n[2/3] Building IEEE 69-bus...")
net69, ne69, te69, n69 = build_ieee69()
r69 = compute_hk_curve("IEEE 69-bus (60 topos)", net69, ne69, te69, n69, N_TEST=1000)
results.append(r69)

# IEEE 119-bus
print("\n[3/3] Building IEEE 119-bus...")
net119, ne119, te119, n119 = build_ieee119()
r119 = compute_hk_curve("IEEE 119-bus (100 topos)", net119, ne119, te119, n119, N_TEST=1000)
results.append(r119)

# ════════════════════════════════════════════════════════════════════════
# 输出结果
# ════════════════════════════════════════════════════════════════════════
lines = []
lines.append("="*70)
lines.append("IP4 H(K) IDENTIFIABILITY CURVE RESULTS")
lines.append(f"Three networks: IEEE 33-bus + IEEE 69-bus + IEEE 119-bus")
lines.append(f"N_TEST=1000 trials (sequential cumulative), sigma=0.009 pu")
lines.append("="*70)

for r in results:
    lines.append(f"\n{r['name']}")
    lines.append(f"  Buses={r['n_bus']}, Topologies={r['N_TOPOS']}, "
                 f"K_lower={r['k_lower']} (nodes {[n+1 for n in r['k_lower_nodes']]})")
    lines.append(f"  {'K':>3}  {'H(K)':>7}  {'std':>5}  {'N_sig':>6}  {'top1_acc':>9}")
    lines.append(f"  {'-'*38}")
    # 只打印关键K值
    key_Ks = sorted(set([1,2,3,5,8,10,12,15,20,25,30]) & set(r['K_vals']))
    for K in key_Ks:
        idx = r['K_vals'].index(K)
        lines.append(f"  {K:>3}  {r['H_mean'][idx]:>7.3f}  "
                     f"{r['H_std'][idx]:>5.3f}  "
                     f"{r['N_sig_mean'][idx]:>6.1f}  "
                     f"{r['top1_acc'][idx]:>9.3f}")

# 两网络对比关键数字
lines.append("\n\nComparison at key K values:")
lines.append(f"{'K':>4}  {'33-bus H':>10}  {'69-bus H':>10}  {'119-bus H':>10}  "
             f"{'33-acc':>8}  {'69-acc':>8}  {'119-acc':>8}")
lines.append("-"*65)
for K in [3, 5, 8, 12, 15, 20]:
    r3 = results[0]; r6 = results[1]; r11 = results[2]
    if K in r3['K_vals'] and K in r6['K_vals'] and K in r11['K_vals']:
        i3=r3['K_vals'].index(K); i6=r6['K_vals'].index(K); i11=r11['K_vals'].index(K)
        lines.append(f"{K:>4}  {r3['H_mean'][i3]:>10.3f}  {r6['H_mean'][i6]:>10.3f}  "
                     f"{r11['H_mean'][i11]:>10.3f}  "
                     f"{r3['top1_acc'][i3]:>8.3f}  {r6['top1_acc'][i6]:>8.3f}  "
                     f"{r11['top1_acc'][i11]:>8.3f}")

lines.append("\nK_lower comparison (exhaustive search, necessary-not-sufficient):")
for r in results:
    h_at_klower = r['H_mean'][r['k_lower']-1]
    mm = "multi-modal" if h_at_klower > 1.0 else "borderline"
    lines.append(f"  {r['name']}: K_lower={r['k_lower']} "
                 f"-> H(K_lower)={h_at_klower:.3f} ({mm})")

lines.append("\nMonotonicity check (sequential DPI guarantee):")
for r in results:
    status = "PASS (monotone)" if r.get('monotone') else "FAIL (non-monotone)"
    lines.append(f"  {r['name']}: {status}")

lines.append("\nClaim boundary analysis:")
for r in results:
    Ks = r['K_vals']
    Hs = r['H_mean']
    accs = r['top1_acc']
    k_below1 = next((Ks[i] for i in range(len(Hs)) if Hs[i] < 1.0), None)
    k_80acc  = next((Ks[i] for i in range(len(accs)) if accs[i] >= 0.80), None)
    lines.append(f"  {r['name']}:")
    lines.append(f"    H drops below 1.0 at K={k_below1} "
                 f"({'never in scan' if k_below1 is None else f'H={Hs[Ks.index(k_below1)]:.3f}'})")
    lines.append(f"    top-1 reaches 80% at K={k_80acc} "
                 f"({'never in scan' if k_80acc is None else ''})")
    lines.append(f"    => Defensible claim: 'multi-modal (H>1.0, N_sig>3) for K<={k_below1-1 if k_below1 else max(Ks)}'")
    lines.append(f"    => Point-estimate unreliable claim: top1<80% for K<={k_80acc-1 if k_80acc else max(Ks)}")
lines.append("\nMethodology note:")
lines.append("  obs_v = V_all[true_topo][obs_nodes] + Normal(0, sigma)")
lines.append("  posterior = AIS with same sigma  [NO mismatch]")
lines.append("  N_TEST=1000 per K  [high enough for monotone curve]")
lines.append("="*70)

output = "\n".join(lines)
print("\n" + output)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write(output)
print(f"\nSaved: {OUT_PATH}")
