# -*- coding: utf-8 -*-
"""
IP1专用模型 v3：加深网络 + 更多训练步 + 学习率余弦衰减
目标：从78%提升到82%+
"""
import copy, time, warnings
import numpy as np
import torch
import torch.nn as nn
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_IP1 = r"<LOCAL_DESKTOP>\nre_ip1_v3.pt"
K_FIXED  = 12
N_STEPS  = 60000      # 比v2多2.4倍
BATCH    = 512
LR       = 3e-4
SIGMA    = 0.009
LOG_EVERY= 5000
np.random.seed(42); torch.manual_seed(42)
print(f"Device: {DEVICE}")

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

# 更深的网络：加残差连接
class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d))
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(x + self.layers(x))

class FixedKNRE_v3(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(66, 512), nn.LayerNorm(512), nn.GELU())
        self.res1 = ResBlock(512)
        self.res2 = ResBlock(512)
        self.res3 = ResBlock(512)
        self.down  = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, n))
    def forward(self, x):
        h = self.embed(x)
        h = self.res1(h); h = self.res2(h); h = self.res3(h)
        return self.down(h)

print("Building IEEE 33-bus network...")
net33,ne33,te33=build_ieee33()
topos_raw=enum_topos(ne33,te33)
N_TOPOS=len(topos_raw)
print(f"Topologies: {N_TOPOS}")

print("Running power flow for all topologies...")
voltages=[run_pf(net33,t) for t in topos_raw]
V_all=np.stack([v for v in voltages])
print(f"V_all shape: {V_all.shape}")

model=FixedKNRE_v3(N_TOPOS).to(DEVICE)
n_params=sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=N_STEPS,eta_min=1e-5)
loss_fn=nn.CrossEntropyLoss()

def ais_post(obs_v,obs_n,V_all,sigma):
    diff=(V_all[:,obs_n]-obs_v)/sigma
    ll=-0.5*np.sum(diff**2,axis=1); ll-=ll.max()
    w=np.exp(ll); return w/w.sum()

# 预生成更多样本
def gen_batch(rng, n):
    xs, ys = [], []
    for _ in range(n):
        ti = rng.randint(0, N_TOPOS)
        obs = np.sort(rng.choice(range(1, 33), K_FIXED, replace=False))
        obs_v = V_all[ti][obs] + rng.normal(0, SIGMA, K_FIXED)
        x = np.zeros(66, dtype=np.float32)
        x[obs] = obs_v; x[33 + obs] = 1.0
        xs.append(x); ys.append(ti)
    return (torch.tensor(np.array(xs), dtype=torch.float32).to(DEVICE),
            torch.tensor(ys, dtype=torch.long).to(DEVICE))

rng = np.random.RandomState(42)
model.train()
t0 = time.time()

print(f"\nTraining v3: {N_STEPS} steps, residual network, cosine LR...")
print(f"{'Step':>7}  {'Loss':>8}  {'LR':>9}  {'Elapsed':>8}")
print("-" * 40)

for step in range(1, N_STEPS + 1):
    xb, yb = gen_batch(rng, BATCH)
    logits = model(xb)
    loss = loss_fn(logits, yb)
    opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()

    if step % LOG_EVERY == 0:
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        print(f"  {step:>7}  {loss.item():>8.4f}  {lr_now:>9.2e}  {elapsed:>7.1f}s")

# 测试集评估
model.eval()
rng_te = np.random.RandomState(77)
acc1_ais, acc1_nre, cov90, kl_vals = [], [], [], []

print("\nEvaluating on 1000 test samples (sigma=0.009/3=0.003)...")
for _ in range(1000):
    ti = rng_te.randint(0, N_TOPOS)
    obs = np.sort(rng_te.choice(range(1, 33), K_FIXED, replace=False))
    obs_v = V_all[ti][obs] + rng_te.normal(0, SIGMA * 0.3, K_FIXED)
    p_ais = ais_post(obs_v, obs, V_all, SIGMA)
    acc1_ais.append(int(np.argmax(p_ais) == ti))
    x = np.zeros(66, dtype=np.float32); x[obs] = obs_v; x[33 + obs] = 1.0
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    p_nre = np.exp(logits - logits.max()); p_nre /= p_nre.sum()
    acc1_nre.append(int(np.argmax(p_nre) == ti))
    si = np.argsort(p_nre)[::-1]; cs = np.cumsum(p_nre[si])
    cov90.append(int(ti in si[:np.searchsorted(cs, 0.90) + 1]))
    kl_vals.append(np.sum(p_ais * np.log((p_ais + 1e-10) / (p_nre + 1e-10))))

acc_ais = np.mean(acc1_ais)
acc_nre = np.mean(acc1_nre)
gap = acc_ais - acc_nre
ci90 = np.mean(cov90)
kl_m = np.mean(kl_vals)

print(f"\n{'='*55}")
print(f"IP1 NRE v3 RESULTS")
print(f"  AIS accuracy:    {acc_ais:.3f}")
print(f"  NRE v3 accuracy: {acc_nre:.3f}  (gap={gap:+.3f})")
print(f"  90% CI coverage: {ci90:.3f}  (target=0.900)")
print(f"  KL(AIS||NRE):    {kl_m:.4f}")
print(f"{'='*55}")

torch.save({'model_state': model.state_dict(), 'N_TOPOS': N_TOPOS,
            'K_FIXED': K_FIXED, 'SIGMA': SIGMA, 'voltages': V_all,
            'topos_raw': topos_raw}, SAVE_IP1)
print(f"Saved: {SAVE_IP1}")
