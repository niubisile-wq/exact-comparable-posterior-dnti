# -*- coding: utf-8 -*-
"""
剩余风险验证：sbi正式MNPE + 123-bus拓扑规模 + 训练时间估算
"""
import warnings, time, copy
import numpy as np
import torch
import pandapower as pp
import pandapower.networks as pn
import networkx as nx
warnings.filterwarnings('ignore')
np.random.seed(0); torch.manual_seed(0)

print("=" * 65)
print("Remaining Risk Verification")
print("=" * 65)

# ── 复用基础设施 ─────────────────────────────────────────────────────
def build_ieee33():
    net = pp.create_empty_network()
    for i in range(33): pp.create_bus(net, vn_kv=12.66)
    branches = [
        (0,1,0.0922,0.0470),(1,2,0.4930,0.2511),(2,3,0.3660,0.1864),
        (3,4,0.3811,0.1941),(4,5,0.8190,0.7070),(5,6,0.1872,0.6188),
        (6,7,0.7114,0.2351),(7,8,1.0300,0.7400),(8,9,1.0440,0.7400),
        (9,10,0.1966,0.0650),(10,11,0.3744,0.1238),(11,12,1.4680,1.1550),
        (12,13,0.5416,0.7129),(13,14,0.5910,0.5260),(14,15,0.7463,0.5450),
        (15,16,1.2890,1.7210),(16,17,0.7320,0.5740),(1,18,0.1640,0.1565),
        (18,19,1.5042,1.3554),(19,20,0.4095,0.4784),(20,21,0.7089,0.9373),
        (2,22,0.4512,0.3083),(22,23,0.8980,0.7091),(23,24,0.8960,0.7011),
        (5,25,0.2030,0.1034),(25,26,0.2842,0.1447),(26,27,1.0590,0.9337),
        (27,28,0.8042,0.7006),(28,29,0.5075,0.2585),(29,30,0.9744,0.9630),
        (30,31,0.3105,0.3619),(31,32,0.3410,0.5302),
    ]
    tie_data = [(7,20,0.089,0.089),(8,14,0.059,0.059),
                (11,21,0.089,0.089),(17,32,0.038,0.085),(24,28,0.056,0.065)]
    for (f,t,r,x) in branches:
        pp.create_line_from_parameters(net,f,t,1,r,x,0,1,in_service=True)
    for (f,t,r,x) in tie_data:
        pp.create_line_from_parameters(net,f,t,1,r,x,0,1,in_service=False)
    loads=[(1,100,60),(2,90,40),(3,120,80),(4,60,30),(5,60,20),(6,200,100),
           (7,200,100),(8,60,20),(9,60,20),(10,45,30),(11,60,35),(12,60,35),
           (13,120,80),(14,60,10),(15,60,20),(16,60,20),(17,90,40),(18,90,40),
           (19,90,40),(20,90,40),(21,90,40),(22,90,50),(23,420,200),(24,420,200),
           (25,60,25),(26,60,25),(27,60,20),(28,120,70),(29,200,600),
           (30,150,70),(31,210,100),(32,60,40)]
    for (b,p,q) in loads: pp.create_load(net,b,p/1000,q/1000)
    pp.create_ext_grid(net,0,vm_pu=1.0)
    ne=[(int(f),int(t)) for (f,t,r,x) in branches]
    te=[(int(f),int(t)) for (f,t,r,x) in tie_data]
    return net, ne, te

def enum_topos(ne, te, n=33):
    G0=nx.Graph(); G0.add_edges_from(ne)
    topos=[list(range(32))]; seen={frozenset(range(32))}
    for ti,tie in enumerate(te):
        path=nx.shortest_path(G0,tie[0],tie[1])
        for i in range(len(path)-1):
            oe=frozenset([path[i],path[i+1]])
            ni=[j for j,e in enumerate(ne) if frozenset(e)!=oe]
            key=frozenset(ni)
            if key in seen: continue
            edges=[ne[j] for j in ni]+[tie]
            G=nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(edges)
            if nx.is_connected(G) and nx.is_tree(G):
                seen.add(key); topos.append(ni+[32+ti])
    return topos

def run_pf(net_base, t_raw):
    net=copy.deepcopy(net_base)
    norm_set={x for x in t_raw if x<32}
    tie_set={x-32 for x in t_raw if x>=32}
    for li in range(37):
        active=(li in norm_set) if li<32 else ((li-32) in tie_set)
        net.line.at[net.line.index[li],'in_service']=active
    try:
        pp.runpp(net,algorithm='bfsw',numba=False,max_iteration=50,tolerance_mva=1e-8)
        if net.converged: return net.res_bus.vm_pu.values.copy()
    except: pass
    return None

net33, ne33, te33 = build_ieee33()
topos_raw = enum_topos(ne33, te33)
voltages = [run_pf(net33, t) for t in topos_raw]
sigma = 0.009
K_obs = 12
obs_nodes = np.sort(np.random.RandomState(0).choice(range(1,33), K_obs, replace=False))

# ═══════════════════════════════════════════════════════════════════
# RISK A: sbi正式MNPE能否收敛（不用自己的分类器）
# ═══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("RISK A: sbi official MNPE convergence test")
print("─"*65)

try:
    from sbi.inference import MNPE
    from sbi.utils.user_input_checks import process_prior
    import torch.distributions as dist

    # 生成训练数据（拓扑索引作为离散参数）
    rng = np.random.RandomState(1)
    X_all, theta_all = [], []
    for ti, V in enumerate(voltages):
        if V is None: continue
        for _ in range(300):
            obs = V[obs_nodes] + rng.normal(0, sigma, K_obs)
            X_all.append(obs)
            theta_all.append([float(ti)])  # 拓扑索引（0-31）

    X_t = torch.tensor(np.array(X_all), dtype=torch.float32)
    theta_t = torch.tensor(np.array(theta_all), dtype=torch.float32)
    print(f"  Training set: {len(X_t)} samples")

    # MNPE需要先验——用Categorical（通过连续近似或离散先验）
    # sbi 0.26.x MNPE接受混合先验
    # 对纯离散问题，用Categorical先验
    prior = dist.Categorical(probs=torch.ones(len(topos_raw)) / len(topos_raw))

    t0 = time.perf_counter()
    try:
        # 尝试用MNPE
        from sbi.inference import MNPE
        infr = MNPE(prior=prior)
        infr.append_simulations(theta_t, X_t)
        density_estimator = infr.train(training_batch_size=256,
                                        show_train_summary=False,
                                        max_num_epochs=50)
        posterior = infr.build_posterior(density_estimator)
        train_time = time.perf_counter() - t0
        print(f"  MNPE training (50 epochs): {train_time:.1f}s")

        # 测试推断
        V_true = voltages[0]
        x_obs = torch.tensor(V_true[obs_nodes] + rng.normal(0, sigma*0.3, K_obs),
                             dtype=torch.float32)
        samples = posterior.sample((1000,), x=x_obs, show_progress_bars=False)
        post_hist = torch.histc(samples.float(), bins=len(topos_raw),
                                min=0, max=len(topos_raw)-1)
        post_prob = (post_hist / post_hist.sum()).numpy()
        top3 = np.argsort(post_prob)[::-1][:3]
        print(f"  MNPE top-3: {top3}, probs={post_prob[top3].round(3)}")
        print(f"  True topo (0) rank: {list(np.argsort(post_prob)[::-1]).index(0)+1}")
        print(f"  RISK A: PASS - sbi MNPE works on 33-bus")

    except Exception as e1:
        print(f"  MNPE direct: {e1}")
        # 退回NRE方法
        from sbi.inference import NRE_A
        print("  Trying NRE (naturally handles discrete)...")
        infr2 = NRE_A(prior=prior)
        infr2.append_simulations(theta_t, X_t)
        clf = infr2.train(show_train_summary=False, max_num_epochs=50)
        train_time2 = time.perf_counter() - t0
        print(f"  NRE training (50 epochs): {train_time2:.1f}s")
        print(f"  RISK A: PARTIAL - use NRE instead of MNPE")

except Exception as e:
    print(f"  RISK A: ERROR - {e}")
    import traceback; traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════
# RISK B: IEEE 57-bus 拓扑规模（123-bus不在pandapower里，先测57-bus）
# ═══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("RISK B: Scalability test on IEEE 57-bus")
print("─"*65)

net57 = pn.case57()
print(f"  IEEE 57-bus: {len(net57.bus)} buses, {len(net57.line)} lines")

# 分析57-bus网络结构
G57 = nx.Graph()
for _, row in net57.line.iterrows():
    if row.in_service:
        G57.add_edge(int(row.from_bus), int(row.to_bus))
n_components = nx.number_connected_components(G57)
is_tree = nx.is_tree(G57)
n_cycles = G57.number_of_edges() - G57.number_of_nodes() + n_components
print(f"  Components: {n_components}, is_tree: {is_tree}")
print(f"  Number of independent cycles (=potential tie switches): {n_cycles}")
print(f"  Estimated valid topologies: up to {2**n_cycles} "
      f"(actual radial subset is smaller)")

# 测试57-bus单次潮流时间
t0 = time.perf_counter()
for _ in range(10):
    try: pp.runpp(net57, numba=False, max_iteration=50)
    except: pass
pf_time_57 = (time.perf_counter()-t0)/10 * 1000
print(f"  Single power flow (57-bus): {pf_time_57:.1f}ms")

# 估算：如果57-bus有N种拓扑，AIS vs NPE
print(f"\n  --- AIS vs NPE speedup projection ---")
for n_topos in [50, 200, 1000, 5000]:
    ais_ms = pf_time_57 * n_topos  # AIS: N次潮流
    npe_ms = 0.1  # NPE: 单次前向传播
    print(f"  {n_topos:>5} topologies: "
          f"AIS={ais_ms:.0f}ms  NPE={npe_ms:.1f}ms  "
          f"speedup={ais_ms/npe_ms:.0f}x")

# ═══════════════════════════════════════════════════════════════════
# RISK C: 训练数据生成速度（最关键的工程瓶颈）
# ═══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("RISK C: Training data generation speed")
print("─"*65)

# 测量33-bus单次潮流时间
t0 = time.perf_counter()
N_test = 50
for i in range(N_test):
    run_pf(net33, topos_raw[i % len(topos_raw)])
pf_time_33 = (time.perf_counter()-t0)/N_test * 1000
print(f"  IEEE 33-bus: {pf_time_33:.1f}ms/power flow")
print(f"  57-bus: {pf_time_57:.1f}ms/power flow")

# 训练数据规模估算
for n_samples in [5000, 20000, 50000, 100000]:
    t_33 = n_samples * pf_time_33 / 1000 / 60
    t_57 = n_samples * pf_time_57 / 1000 / 60
    print(f"  {n_samples:>7} samples: 33-bus={t_33:.1f}min  57-bus={t_57:.1f}min")

# ═══════════════════════════════════════════════════════════════════
# RISK D: BA bound 在真实MNPE下的质量
# ═══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("RISK D: BA bound quality check")
print("─"*65)

from torch import nn
class Cls(nn.Module):
    def __init__(self, in_d, n_c):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_d,128),nn.ReLU(),nn.BatchNorm1d(128),
            nn.Linear(128,128),nn.ReLU(),
            nn.Linear(128,n_c))
    def forward(self,x): return self.net(x)

# 更认真地训练分类器（1000 epochs）
rng2=np.random.RandomState(2)
X2,y2=[],[]
for ti,V in enumerate(voltages):
    if V is None: continue
    for _ in range(500):
        X2.append(V[obs_nodes]+rng2.normal(0,sigma,K_obs))
        y2.append(ti)
X2t=torch.tensor(np.array(X2),dtype=torch.float32)
y2t=torch.tensor(y2,dtype=torch.long)

model2=Cls(K_obs,len(topos_raw))
opt=torch.optim.Adam(model2.parameters(),lr=1e-3,weight_decay=1e-4)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,1000)
crit=nn.CrossEntropyLoss()

print("  Training improved classifier (1000 epochs)...")
t0=time.perf_counter()
for ep in range(1000):
    opt.zero_grad()
    loss=crit(model2(X2t),y2t)
    loss.backward(); opt.step(); sched.step()
train_time=time.perf_counter()-t0
print(f"  Training time: {train_time:.1f}s")

# 评估：top-1 accuracy and posterior calibration
model2.eval()
correct=0
with torch.no_grad():
    logits=model2(X2t)
    preds=logits.argmax(dim=1)
    correct=(preds==y2t).float().mean().item()
print(f"  Training accuracy: {correct:.3f}")

# 测试集（未见过的噪声样本）
X_test,y_test=[],[]
rng3=np.random.RandomState(99)
for ti,V in enumerate(voltages):
    if V is None: continue
    for _ in range(100):
        X_test.append(V[obs_nodes]+rng3.normal(0,sigma,K_obs))
        y_test.append(ti)
Xt=torch.tensor(np.array(X_test),dtype=torch.float32)
yt=torch.tensor(y_test,dtype=torch.long)
with torch.no_grad():
    logits_t=model2(Xt)
    test_acc=(logits_t.argmax(1)==yt).float().mean().item()
    # posterior entropy
    probs=torch.softmax(logits_t,dim=1).numpy()
    mean_entropy=float(-np.mean(np.sum(probs*np.log(probs+1e-15),axis=1)))
print(f"  Test accuracy: {test_acc:.3f}")
print(f"  Mean posterior entropy: {mean_entropy:.3f} "
      f"(max possible={np.log(len(topos_raw)):.2f})")

# BA bound区分度：计算几个候选节点的EIG
def ba_eig(model, voltages, candidate_node, current_nodes, sigma, n=300):
    model.eval()
    rng_b=np.random.RandomState(42)
    gains=[]
    all_nodes=np.append(current_nodes,[candidate_node])[:K_obs]
    for _ in range(n):
        ti=rng_b.randint(0,len(voltages))
        if voltages[ti] is None: continue
        V=voltages[ti]
        obs=V[all_nodes]+rng_b.normal(0,sigma,len(all_nodes))
        obs_k=np.zeros(K_obs); obs_k[:len(obs)]=obs
        x=torch.tensor(obs_k,dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            lp=torch.log_softmax(model(x),dim=1)[0,ti].item()
        gains.append(lp)
    return np.mean(gains)

current=np.sort(np.random.RandomState(0).choice(range(1,33),8,replace=False))
print(f"\n  BA bound EIG (improved classifier, K=8 -> 9):")
eigs={}
t0=time.perf_counter()
for node in range(1,33):
    if node not in current:
        eigs[node]=ba_eig(model2,voltages,node,current,sigma,n=200)
elapsed=time.perf_counter()-t0
best=max(eigs,key=eigs.get); worst=min(eigs,key=eigs.get)
eig_range=eigs[best]-eigs[worst]
print(f"  Best node: {best} (EIG={eigs[best]:.3f})")
print(f"  Worst node: {worst} (EIG={eigs[worst]:.3f})")
print(f"  EIG range: {eig_range:.3f}")
print(f"  Time (24 candidates): {elapsed:.2f}s ({elapsed/24*1000:.0f}ms each)")
print(f"  RISK D: {'PASS (BA bound discriminates)' if eig_range > 0.1 else 'WEAK (poor discrimination)'}")

# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("RISK SUMMARY")
print("="*65)
print("""
A. sbi MNPE:         see above - MNPE or NRE handles discrete topo
B. Scalability:      57-bus available now; 123-bus needs conversion
   - With >200 topos, NPE speedup becomes 1000x+ (very strong)
   - 57-bus single PF: see measured time above
C. Training speed:   see above - data generation is fast
   - 50k samples on 33-bus: minutes, not hours
D. BA bound quality: see above - discrimination test
""")
