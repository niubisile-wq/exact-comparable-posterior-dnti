# -*- coding: utf-8 -*-
"""
IP1专用模型 v2：固定K=12，66维掩码输入（正确设计）
关键：模型必须知道是哪些节点被观测，靠mask编码节点身份
"""
import copy, time, warnings
import numpy as np
import torch
import torch.nn as nn
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_IP1 = r"<LOCAL_DESKTOP>\nre_ip1_v2.pt"
K_FIXED  = 12
N_STEPS  = 25000
BATCH    = 512
LR       = 3e-4
SIGMA    = 0.009
LOG_EVERY= 2500
np.random.seed(0); torch.manual_seed(0)
print(f"Device: {DEVICE}")

# ── 网络构建 ──────────────────────────────────────────────────────
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
print(f"  {N_TOPOS} topologies, V_all={V_all.shape}")

# ── 模型：66维输入（33电压+33掩码），固定K=12 ────────────────────
# 与variable-K模型相同架构，但训练时K始终=12
# 好处：模型专注于K=12场景，精度更高
class FixedKMaskedNRE(nn.Module):
    def __init__(self, n_topos):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(66, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, n_topos)
        )
    def forward(self, x): return self.net(x)

    def posterior(self, obs_v, obs_nodes):
        """给定观测节点和电压，返回后验概率"""
        self.eval()
        with torch.no_grad():
            x = np.zeros(66, dtype=np.float32)
            x[obs_nodes] = obs_v
            x[33 + obs_nodes] = 1.0
            xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            return torch.softmax(self(xt), dim=1).cpu().numpy()[0]

model = FixedKMaskedNRE(N_TOPOS).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n[2/4] Model: {n_params:,} params, input=66d (masked), K_fixed={K_FIXED}")

# ── 训练：K固定=12，随机节点，66维掩码输入 ──────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR*5, total_steps=N_STEPS, pct_start=0.05)
criterion = nn.CrossEntropyLoss()
rng = np.random.RandomState(1)

print(f"\n[3/4] Training {N_STEPS} steps (K={K_FIXED} fixed, 66d masked input)...")
t0 = time.perf_counter()

for step in range(1, N_STEPS+1):
    X_np = np.zeros((BATCH, 66), dtype=np.float32)
    y_np = np.zeros(BATCH, dtype=np.int64)
    for i in range(BATCH):
        ti = rng.randint(0, N_TOPOS)
        obs = np.sort(rng.choice(range(1, 33), K_FIXED, replace=False))
        obs_v = V_all[ti][obs] + rng.normal(0, SIGMA, K_FIXED)
        X_np[i, obs] = obs_v         # 观测电压
        X_np[i, 33 + obs] = 1.0      # 节点掩码（编码节点身份）
        y_np[i] = ti

    X_t = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_np, dtype=torch.long).to(DEVICE)
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_t), y_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(); scheduler.step()

    if step % LOG_EVERY == 0 or step == N_STEPS:
        elapsed = time.perf_counter() - t0
        eta = elapsed / step * (N_STEPS - step)
        print(f"  Step {step:>6}/{N_STEPS}  loss={loss.item():.4f}"
              f"  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

print(f"  Done in {time.perf_counter()-t0:.1f}s")

# ── 评估 ────────────────────────────────────────────────────────
print(f"\n[4/4] Evaluation (500 test queries)...")

def ais_post(obs_v, obs_n, V_all, sigma):
    diff = (V_all[:, obs_n] - obs_v) / sigma
    ll = -0.5*np.sum(diff**2, axis=1); ll -= ll.max()
    w = np.exp(ll); return w / w.sum()

rng_te = np.random.RandomState(99)
acc1_ais, acc1_nre, acc3_nre, kls, H_nre, H_ais = [], [], [], [], [], []

for _ in range(500):
    ti = rng_te.randint(0, N_TOPOS)
    # 每次随机选K_FIXED个节点测试（与训练一致）
    obs = np.sort(rng_te.choice(range(1, 33), K_FIXED, replace=False))
    obs_v = V_all[ti][obs] + rng_te.normal(0, SIGMA * 0.3, K_FIXED)

    p_ais = ais_post(obs_v, obs, V_all, SIGMA)
    p_nre = model.posterior(obs_v, obs)

    acc1_ais.append(int(np.argmax(p_ais) == ti))
    acc1_nre.append(int(np.argmax(p_nre) == ti))
    top3 = np.argsort(p_nre)[::-1][:3]
    acc3_nre.append(int(ti in top3))
    kls.append(np.sum(p_ais * np.log((p_ais+1e-10)/(p_nre+1e-10))))
    H_nre.append(-np.sum(p_nre * np.log(p_nre + 1e-15)))
    H_ais.append(-np.sum(p_ais * np.log(p_ais + 1e-15)))

gap = np.mean(acc1_ais) - np.mean(acc1_nre)

print(f"\n  {'方法':>22}  {'top-1':>7}  {'top-3':>7}  {'H(post)':>8}")
print(f"  {'-'*50}")
print(f"  {'AIS (精确贝叶斯)':>22}  {np.mean(acc1_ais):>7.3f}  "
      f"{'---':>7}  {np.mean(H_ais):>8.3f}")
print(f"  {'Fixed-K NRE (本模型)':>22}  {np.mean(acc1_nre):>7.3f}  "
      f"{np.mean(acc3_nre):>7.3f}  {np.mean(H_nre):>8.3f}")
print(f"\n  精度差距 (AIS-NRE): {gap:+.3f}")
print(f"  KL(AIS||NRE):       {np.mean(kls):.4f}")
print(f"  NRE覆盖AIS精度的:   {np.mean(acc1_nre)/np.mean(acc1_ais)*100:.1f}%")

# 后验校准
coverage = []
for _ in range(500):
    ti = rng_te.randint(0, N_TOPOS)
    obs = np.sort(rng_te.choice(range(1,33), K_FIXED, replace=False))
    obs_v = V_all[ti][obs] + rng_te.normal(0, SIGMA*0.3, K_FIXED)
    p_nre = model.posterior(obs_v, obs)
    sorted_idx = np.argsort(p_nre)[::-1]
    cumsum = np.cumsum(p_nre[sorted_idx])
    n_set = np.searchsorted(cumsum, 0.90) + 1
    coverage.append(int(ti in sorted_idx[:n_set]))

cov = np.mean(coverage)
print(f"\n  后验校准 (90% CI实际覆盖率): {cov:.3f}  "
      f"{'良好' if abs(cov-0.9)<0.05 else '偏宽' if cov>0.9 else '偏窄'}")

# 保存
torch.save({'model_state': model.state_dict(), 'N_TOPOS': N_TOPOS,
            'K_FIXED': K_FIXED, 'SIGMA': SIGMA,
            'voltages': V_all, 'topos_raw': topos_raw}, SAVE_IP1)

verdict = ('PASS (gap<0.05)' if gap < 0.05 else
           'OK (gap<0.08)' if gap < 0.08 else 'LARGE (gap>=0.08)')
print(f"""
{'='*55}
IP1 FIXED-K MODEL v2 COMPLETE
  top-1 gap:  {gap:+.3f}  {verdict}
  KL:         {np.mean(kls):.4f}
  90% CI:     {cov:.3f}
  Saved: {SAVE_IP1}
{'='*55}
""")
