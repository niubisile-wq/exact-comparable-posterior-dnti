# -*- coding: utf-8 -*-
"""
IP-C 正式实验：通信缺失鲁棒推断
对比：鲁棒NRE vs 朴素NRE vs 点估计DNN
missing_rate = 0%, 5%, 10%, 20%, 30%
N=500个测试样本，更可靠
"""
import copy, time, warnings
import numpy as np
import torch
import torch.nn as nn
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = r"<LOCAL_DESKTOP>\nre_model.pt"
OUT_PATH   = r"<LOCAL_DESKTOP>\ipc_result.txt"
N_TEST = 500
K_TEST = 15
SIGMA  = 0.009
print(f"Device: {DEVICE}")

# ── 网络构建（与训练时完全相同）──────────────────────────────────────

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
    return net, [(int(f),int(t)) for f,t,r,x in br], [(int(f),int(t)) for f,t,r,x in ti]

def enum_topos(ne, te, n=33):
    G = nx.Graph(); G.add_edges_from(ne)
    topos = [list(range(32))]; seen = {frozenset(range(32))}
    for ti2, tie in enumerate(te):
        path = nx.shortest_path(G, tie[0], tie[1])
        for i in range(len(path)-1):
            oe = frozenset([path[i], path[i+1]])
            ni = [j for j,e in enumerate(ne) if frozenset(e) != oe]
            key = frozenset(ni)
            if key in seen: continue
            edges = [ne[j] for j in ni] + [tie]
            Gt = nx.Graph(); Gt.add_nodes_from(range(n)); Gt.add_edges_from(edges)
            if nx.is_connected(Gt) and nx.is_tree(Gt):
                seen.add(key); topos.append(ni + [32+ti2])
    return topos

class MaskedNRE(nn.Module):
    def __init__(self, n_topos):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(66, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, n_topos))
    def forward(self, x): return self.net(x)
    def posterior(self, x_np):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x_np, dtype=torch.float32).to(DEVICE)
            if x.ndim == 1: x = x.unsqueeze(0)
            return torch.softmax(self(x), dim=1).cpu().numpy()[0]

# ── 加载 ────────────────────────────────────────────────────────────────
print("Loading network and model...")
net33, ne33, te33 = build_ieee33()
topos_raw = enum_topos(ne33, te33)
N_TOPOS = len(topos_raw)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
V_all = ckpt['voltages']
model = MaskedNRE(N_TOPOS).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"N_TOPOS={N_TOPOS}, V_all={V_all.shape}")

# AIS精确后验
def ais_posterior(obs_v, obs_nodes, V_all, sigma):
    diff = (V_all[:, obs_nodes] - obs_v) / sigma
    ll = -0.5 * np.sum(diff**2, axis=1)
    ll -= ll.max(); w = np.exp(ll)
    return w / w.sum()

def posterior_entropy(p):
    p = np.clip(p, 1e-10, 1.0)
    return -np.sum(p * np.log(p))

# ── 实验 ────────────────────────────────────────────────────────────────
miss_rates = [0.0, 0.05, 0.10, 0.20, 0.30]
rng = np.random.RandomState(999)

res_robust = {mr: [] for mr in miss_rates}
res_naive  = {mr: [] for mr in miss_rates}
res_ais    = {mr: [] for mr in miss_rates}
ent_robust = {mr: [] for mr in miss_rates}
ent_naive  = {mr: [] for mr in miss_rates}
top3_robust = {mr: [] for mr in miss_rates}

print(f"\nRunning {N_TEST} test samples x {len(miss_rates)} missing rates...")
t0 = time.time()

for idx in range(N_TEST):
    if idx % 100 == 0:
        print(f"  {idx}/{N_TEST}  t={time.time()-t0:.0f}s")

    ti = rng.randint(0, N_TOPOS)
    obs_all = np.sort(rng.choice(range(1, 33), K_TEST, replace=False))
    obs_v_all = V_all[ti][obs_all] + rng.normal(0, SIGMA * 0.3, K_TEST)

    for mr in miss_rates:
        if mr == 0.0:
            keep = np.ones(K_TEST, dtype=bool)
        else:
            keep = rng.rand(K_TEST) > mr
            if not np.any(keep): keep[0] = True

        obs_avail = obs_all[keep]
        obs_v_avail = obs_v_all[keep]

        # 鲁棒NRE：正确掩码（只声明实际可用的节点）
        x_robust = np.zeros(66, dtype=np.float32)
        x_robust[obs_avail] = obs_v_avail
        x_robust[33 + obs_avail] = 1.0

        # 朴素NRE：错误声称所有节点都可用，缺失节点填0值但mask仍=1
        x_naive = np.zeros(66, dtype=np.float32)
        x_naive[obs_all] = obs_v_all * keep  # 缺失位的值=0
        x_naive[33 + obs_all] = 1.0           # 错误地声明全部可观

        p_robust = model.posterior(x_robust)
        p_naive  = model.posterior(x_naive)

        # AIS只用可用测量
        p_ais = ais_posterior(obs_v_avail, obs_avail, V_all, SIGMA)

        res_robust[mr].append(int(np.argmax(p_robust) == ti))
        res_naive[mr].append(int(np.argmax(p_naive) == ti))
        res_ais[mr].append(int(np.argmax(p_ais) == ti))
        ent_robust[mr].append(posterior_entropy(p_robust))
        ent_naive[mr].append(posterior_entropy(p_naive))

        # top-3
        top3 = np.argsort(p_robust)[::-1][:3]
        top3_robust[mr].append(int(ti in top3))

# ── 输出 ────────────────────────────────────────────────────────────────
total_time = time.time() - t0
lines = []
lines.append("=" * 75)
lines.append("IP-C FORMAL EXPERIMENT: Communication-Missing Robust NRE")
lines.append(f"N_TEST={N_TEST}, K={K_TEST}, SIGMA={SIGMA}, time={total_time:.0f}s")
lines.append("=" * 75)
lines.append("")
lines.append(f"{'MissRate':>8}  {'Robust_acc':>10}  {'Naive_acc':>10}  {'AIS_acc':>8}  "
             f"{'Top3_rob':>9}  {'H_robust':>9}  {'H_naive':>9}  {'Delta_H':>8}")
lines.append("-" * 75)

for mr in miss_rates:
    r = np.mean(res_robust[mr])
    n = np.mean(res_naive[mr])
    a = np.mean(res_ais[mr])
    t3 = np.mean(top3_robust[mr])
    hr = np.mean(ent_robust[mr])
    hn = np.mean(ent_naive[mr])
    lines.append(f"{mr*100:>7.0f}%  {r:>10.3f}  {n:>10.3f}  {a:>8.3f}  "
                 f"{t3:>9.3f}  {hr:>9.3f}  {hn:>9.3f}  {hr-hn:>+8.3f}")

lines.append("")
lines.append("Key findings:")
mr10 = 0.10
lines.append(f"  At 10% missing: Robust={np.mean(res_robust[mr10]):.3f}  "
             f"Naive={np.mean(res_naive[mr10]):.3f}  "
             f"Delta={np.mean(res_robust[mr10])-np.mean(res_naive[mr10]):+.3f}")
mr30 = 0.30
lines.append(f"  At 30% missing: Robust={np.mean(res_robust[mr30]):.3f}  "
             f"Naive={np.mean(res_naive[mr30]):.3f}  "
             f"Delta={np.mean(res_robust[mr30])-np.mean(res_naive[mr30]):+.3f}")
lines.append("")
lines.append("Entropy analysis (H increases = posterior widens = correct UQ):")
for mr in [0.0, 0.10, 0.20, 0.30]:
    hr = np.mean(ent_robust[mr])
    hn = np.mean(ent_naive[mr])
    lines.append(f"  miss={mr*100:.0f}%: H_robust={hr:.3f}  H_naive={hn:.3f}  "
                 f"({'robust wider=correct' if hr>hn else 'robust narrower'})")
lines.append("=" * 75)

output = "\n".join(lines)
print(output)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write(output)
print(f"\nSaved: {OUT_PATH}")
