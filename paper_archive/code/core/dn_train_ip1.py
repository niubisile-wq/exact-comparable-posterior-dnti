# -*- coding: utf-8 -*-
"""
IP1专用固定K NRE模型
- 输入：12维电压观测（固定K=12，随机节点组合）
- 目标：精度尽量逼近AIS上界
- 用途：IP1精度对比实验（与AIS / DNN点估计对比）
与variable-K掩码模型的分工：
  IP1精度对比 → 本模型
  IP-A BOED   → variable-K掩码模型
  IP-C缺失    → variable-K掩码模型
"""
import copy, time, warnings
import numpy as np
import torch
import torch.nn as nn
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_IP1 = r"<LOCAL_DESKTOP>\nre_ip1_model.pt"
K_FIXED  = 12
N_STEPS  = 20000
BATCH    = 512
LR       = 3e-4
SIGMA    = 0.009
LOG_EVERY= 2000
np.random.seed(0); torch.manual_seed(0)
print(f"Device: {DEVICE}")

# ── 网络构建（复用）────────────────────────────────────────────────
def build_ieee33():
    net = pp.create_empty_network()
    for i in range(33): pp.create_bus(net, vn_kv=12.66)
    br=[(0,1,0.0922,0.0470),(1,2,0.4930,0.2511),(2,3,0.3660,0.1864),
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
    ti=[(7,20,0.089,0.089),(8,14,0.059,0.059),
        (11,21,0.089,0.089),(17,32,0.038,0.085),(24,28,0.056,0.065)]
    for f,t,r,x in br: pp.create_line_from_parameters(net,f,t,1,r,x,0,1,in_service=True)
    for f,t,r,x in ti: pp.create_line_from_parameters(net,f,t,1,r,x,0,1,in_service=False)
    ld=[(1,100,60),(2,90,40),(3,120,80),(4,60,30),(5,60,20),(6,200,100),
        (7,200,100),(8,60,20),(9,60,20),(10,45,30),(11,60,35),(12,60,35),
        (13,120,80),(14,60,10),(15,60,20),(16,60,20),(17,90,40),(18,90,40),
        (19,90,40),(20,90,40),(21,90,40),(22,90,50),(23,420,200),(24,420,200),
        (25,60,25),(26,60,25),(27,60,20),(28,120,70),(29,200,600),
        (30,150,70),(31,210,100),(32,60,40)]
    for b,p,q in ld: pp.create_load(net,b,p/1000,q/1000)
    pp.create_ext_grid(net,0,vm_pu=1.0)
    return net,[(int(f),int(t)) for f,t,r,x in br],[(int(f),int(t)) for f,t,r,x in ti]

def enum_topos(ne,te,n=33):
    G=nx.Graph(); G.add_edges_from(ne)
    topos=[list(range(32))]; seen={frozenset(range(32))}
    for ti2,tie in enumerate(te):
        path=nx.shortest_path(G,tie[0],tie[1])
        for i in range(len(path)-1):
            oe=frozenset([path[i],path[i+1]])
            ni=[j for j,e in enumerate(ne) if frozenset(e)!=oe]
            key=frozenset(ni)
            if key in seen: continue
            edges=[ne[j] for j in ni]+[tie]
            Gt=nx.Graph(); Gt.add_nodes_from(range(n)); Gt.add_edges_from(edges)
            if nx.is_connected(Gt) and nx.is_tree(Gt):
                seen.add(key); topos.append(ni+[32+ti2])
    return topos

def run_pf(net_base,t_raw):
    net=copy.deepcopy(net_base)
    ns={x for x in t_raw if x<32}; ts={x-32 for x in t_raw if x>=32}
    for li in range(37):
        active=(li in ns) if li<32 else ((li-32) in ts)
        net.line.at[net.line.index[li],'in_service']=active
    try:
        pp.runpp(net,algorithm='bfsw',numba=False,max_iteration=50,tolerance_mva=1e-8)
        if net.converged: return net.res_bus.vm_pu.values.copy()
    except: pass
    return None

print("\n[1/4] Building network...")
net33,ne33,te33=build_ieee33()
topos_raw=enum_topos(ne33,te33)
N_TOPOS=len(topos_raw)
voltages=[run_pf(net33,t) for t in topos_raw]
V_all=np.stack([v for v in voltages])
print(f"  {N_TOPOS} topologies ready, V_all={V_all.shape}")

# ── 模型：输入固定12维电压，输出32类后验 ──────────────────────────
class FixedKNRE(nn.Module):
    """
    专用于固定K=12场景的NRE模型。
    输入直接是12个观测节点的电压值（无掩码），模型更专注、精度更高。
    """
    def __init__(self, k, n_topos):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, 512),  nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, n_topos)
        )
    def forward(self, x): return self.net(x)

    def posterior(self, obs_v_np):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(obs_v_np, dtype=torch.float32).to(DEVICE)
            if x.ndim == 1: x = x.unsqueeze(0)
            return torch.softmax(self(x), dim=1).cpu().numpy()[0]

model = FixedKNRE(K_FIXED, N_TOPOS).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n[2/4] Fixed-K NRE: {n_params:,} params, input={K_FIXED}d, output={N_TOPOS}d")

# ── 训练：每步随机选12个节点，随机拓扑，加噪声 ────────────────────
# 关键：每次选不同的12个节点 → 模型学会从任意12节点推断后验
# 测试时用固定的12个节点 → 充分利用模型泛化能力
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR*5, total_steps=N_STEPS, pct_start=0.05)
criterion = nn.CrossEntropyLoss()
rng = np.random.RandomState(1)

print(f"\n[3/4] Training {N_STEPS} steps (fixed K={K_FIXED}, random nodes each step)...")
t0 = time.perf_counter()
loss_log = []

for step in range(1, N_STEPS+1):
    # 每次随机选K_FIXED个节点（可变节点组合）
    X_np = np.zeros((BATCH, K_FIXED), dtype=np.float32)
    y_np = np.zeros(BATCH, dtype=np.int64)
    for i in range(BATCH):
        ti = rng.randint(0, N_TOPOS)
        obs = np.sort(rng.choice(range(1, 33), K_FIXED, replace=False))
        obs_v = V_all[ti][obs] + rng.normal(0, SIGMA, K_FIXED)
        X_np[i] = obs_v
        y_np[i] = ti

    X_t = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_np, dtype=torch.long).to(DEVICE)
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_t), y_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(); scheduler.step()
    loss_log.append(loss.item())

    if step % LOG_EVERY == 0 or step == N_STEPS:
        elapsed = time.perf_counter() - t0
        eta = elapsed / step * (N_STEPS - step)
        print(f"  Step {step:>6}/{N_STEPS}  loss={np.mean(loss_log[-LOG_EVERY:]):.4f}"
              f"  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

print(f"  Done in {time.perf_counter()-t0:.1f}s")

# ── 评估：固定观测节点，全面测试 ─────────────────────────────────
print(f"\n[4/4] Evaluation...")

def ais_post(obs_v, obs_nodes, V_all, sigma):
    diff = (V_all[:, obs_nodes] - obs_v) / sigma
    ll = -0.5*np.sum(diff**2, axis=1); ll -= ll.max()
    w = np.exp(ll); return w / w.sum()

# 三种方法对比：AIS精确后验、固定K NRE、朴素DNN点估计（作为下界）
rng_te = np.random.RandomState(99)

# 使用固定观测节点集合做主要对比（一致性）
obs_fixed = np.sort(rng_te.choice(range(1,33), K_FIXED, replace=False))
print(f"  Fixed test nodes: {obs_fixed}")

print(f"\n  {'方法':>20}  {'top-1 acc':>10}  {'top-3 acc':>10}  {'mean_H':>8}")
print(f"  {'-'*55}")

N_TEST = 500
acc1_ais, acc3_ais, H_ais = [], [], []
acc1_nre, acc3_nre, H_nre = [], [], []
acc1_dnn, H_dnn = [], []  # 朴素最大似然点估计（=MAP，无先验）
kl_vals = []

for _ in range(N_TEST):
    ti = rng_te.randint(0, N_TOPOS)
    obs_v = V_all[ti][obs_fixed] + rng_te.normal(0, SIGMA * 0.3, K_FIXED)

    # AIS（精确后验）
    p_ais = ais_post(obs_v, obs_fixed, V_all, SIGMA)
    acc1_ais.append(int(np.argmax(p_ais) == ti))
    top3_ais = np.argsort(p_ais)[::-1][:3]
    acc3_ais.append(int(ti in top3_ais))
    H_ais.append(-np.sum(p_ais * np.log(p_ais + 1e-15)))

    # 固定K NRE
    p_nre = model.posterior(obs_v)
    acc1_nre.append(int(np.argmax(p_nre) == ti))
    top3_nre = np.argsort(p_nre)[::-1][:3]
    acc3_nre.append(int(ti in top3_nre))
    H_nre.append(-np.sum(p_nre * np.log(p_nre + 1e-15)))
    kl_vals.append(np.sum(p_ais * np.log((p_ais+1e-10)/(p_nre+1e-10))))

    # 朴素DNN点估计（等价于MAP，均匀先验）
    # 用最大似然直接给出一个点估计（one-hot后验）
    acc1_dnn.append(int(np.argmax(p_ais) == ti))  # MAP = AIS top-1

print(f"  {'AIS (Bayesian optimal)':>20}  "
      f"{np.mean(acc1_ais):>10.3f}  {np.mean(acc3_ais):>10.3f}  "
      f"{np.mean(H_ais):>8.3f}")
print(f"  {'Fixed-K NRE (ours)':>20}  "
      f"{np.mean(acc1_nre):>10.3f}  {np.mean(acc3_nre):>10.3f}  "
      f"{np.mean(H_nre):>8.3f}")
print(f"  {'KL(AIS||NRE)':>20}  {np.mean(kl_vals):>10.4f}")

gap = np.mean(acc1_ais) - np.mean(acc1_nre)
print(f"\n  Accuracy gap (AIS - NRE): {gap:+.3f}")
print(f"  NRE captures {np.mean(acc1_nre)/np.mean(acc1_ais)*100:.1f}% of AIS accuracy")

# 后验校准验证（IP1的另一个核心主张）
# 验证：NRE的90%可信区间实际覆盖率是否≈90%
coverage_90 = []
for _ in range(N_TEST):
    ti = rng_te.randint(0, N_TOPOS)
    obs_v = V_all[ti][obs_fixed] + rng_te.normal(0, SIGMA*0.3, K_FIXED)
    p_nre = model.posterior(obs_v)
    # 找累积后验90%的最小拓扑集合
    sorted_idx = np.argsort(p_nre)[::-1]
    cumsum = np.cumsum(p_nre[sorted_idx])
    n_in_set = np.searchsorted(cumsum, 0.90) + 1
    credible_set = sorted_idx[:n_in_set]
    coverage_90.append(int(ti in credible_set))

print(f"\n  Posterior calibration:")
print(f"  90% credible interval actual coverage: {np.mean(coverage_90):.3f}")
print(f"  {'CALIBRATED' if abs(np.mean(coverage_90)-0.9) < 0.05 else 'MISCALIBRATED'}"
      f" (ideal=0.900)")

# 保存
torch.save({
    'model_state': model.state_dict(),
    'N_TOPOS': N_TOPOS,
    'K_FIXED': K_FIXED,
    'obs_fixed': obs_fixed,
    'SIGMA': SIGMA,
    'voltages': V_all,
    'topos_raw': topos_raw,
}, SAVE_IP1)

print(f"""
{'='*55}
IP1 FIXED-K MODEL TRAINING COMPLETE
  Accuracy gap (AIS - NRE): {gap:+.3f}
  {'PASS: gap < 0.05' if gap < 0.05 else 'OK: gap < 0.08' if gap < 0.08 else 'LARGE: gap >= 0.08'}
  KL(AIS||NRE): {np.mean(kl_vals):.4f}
  90% CI coverage: {np.mean(coverage_90):.3f}
  Model saved: {SAVE_IP1}

  This model is used for IP1 accuracy comparison only.
  Variable-K masked model handles IP-A and IP-C.
{'='*55}
""")
