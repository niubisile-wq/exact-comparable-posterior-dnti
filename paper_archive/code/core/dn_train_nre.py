# -*- coding: utf-8 -*-
"""
Step 1: 训练统一掩码NRE模型
输入：66维 = 33节点电压 + 33位二值掩码（哪些节点有测量）
输出：32个拓扑的后验概率
支撑：IP1 / IP4 / IP-A / IP-C 四个实验
"""
import os, time, copy, warnings
import numpy as np
import torch
import torch.nn as nn
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
np.random.seed(42); torch.manual_seed(42)

# ── 超参数 ────────────────────────────────────────────────────────────
N_STEPS     = 15000      # 训练步数
BATCH_SIZE  = 512
LR          = 3e-4
SIGMA       = 0.009      # 测量噪声标准差
K_MIN, K_MAX = 5, 28     # 随机观测节点数范围
SAVE_PATH   = r"<LOCAL_DESKTOP>\nre_model.pt"
LOG_EVERY   = 1000

# ── 1. 构建IEEE 33-bus网络 ─────────────────────────────────────────────
def build_ieee33():
    net = pp.create_empty_network()
    for i in range(33): pp.create_bus(net, vn_kv=12.66)
    br = [
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
    return net, ne, te

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
            Gt = nx.Graph(); Gt.add_nodes_from(range(n))
            Gt.add_edges_from(edges)
            if nx.is_connected(Gt) and nx.is_tree(Gt):
                seen.add(key); topos.append(ni + [32+ti2])
    return topos

def run_pf(net_base, t_raw):
    net = copy.deepcopy(net_base)
    ns = {x for x in t_raw if x < 32}
    ts = {x-32 for x in t_raw if x >= 32}
    for li in range(37):
        active = (li in ns) if li < 32 else ((li-32) in ts)
        net.line.at[net.line.index[li], 'in_service'] = active
    try:
        pp.runpp(net, algorithm='bfsw', numba=False,
                 max_iteration=50, tolerance_mva=1e-8)
        if net.converged: return net.res_bus.vm_pu.values.copy()
    except: pass
    return None

# ── 2. 计算所有拓扑电压（只做一次） ──────────────────────────────────
print("\n[1/4] Building network and computing all topology voltages...")
t0 = time.perf_counter()
net33, ne33, te33 = build_ieee33()
topos_raw = enum_topos(ne33, te33)
N_TOPOS = len(topos_raw)
voltages = []
for t in topos_raw:
    V = run_pf(net33, t)
    voltages.append(V)
n_ok = sum(1 for v in voltages if v is not None)
V_all = np.stack([v for v in voltages if v is not None])  # (N_TOPOS, 33)
print(f"  {N_TOPOS} topologies, {n_ok} converged, time={time.perf_counter()-t0:.1f}s")
print(f"  Voltage cache shape: {V_all.shape}")

# ── 3. 定义掩码NRE模型 ────────────────────────────────────────────────
class MaskedNRE(nn.Module):
    """
    输入：[V_masked (33) | binary_mask (33)] = 66维
    V_masked[i] = V[i] 若节点i被观测，否则 = 0
    mask[i]     = 1    若节点i被观测，否则 = 0
    输出：N_TOPOS维 logits（拓扑后验的对数比率）
    """
    def __init__(self, n_topos):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(66, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, n_topos)
        )
    def forward(self, x):
        return self.net(x)

    def posterior(self, x_np):
        """返回拓扑后验概率数组"""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x_np, dtype=torch.float32).to(DEVICE)
            if x.ndim == 1: x = x.unsqueeze(0)
            return torch.softmax(self(x), dim=1).cpu().numpy()[0]

model = MaskedNRE(N_TOPOS).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n[2/4] Model: {n_params:,} parameters")

# ── 4. 训练循环（在线采样，不预生成数据集）────────────────────────────
print(f"\n[3/4] Training ({N_STEPS} steps, batch={BATCH_SIZE}, device={DEVICE})...")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR*5, total_steps=N_STEPS, pct_start=0.05)
criterion = nn.CrossEntropyLoss()

rng = np.random.RandomState(1)
t_start = time.perf_counter()
loss_history = []

def make_batch(rng, batch_size, V_all, N_TOPOS, sigma, K_min, K_max,
               missing_rate=0.0):
    """
    生成一批掩码训练样本。
    missing_rate: 在已选观测节点中随机丢弃的比例（模拟通信失败）。
    """
    X = np.zeros((batch_size, 66), dtype=np.float32)
    y = np.zeros(batch_size, dtype=np.int64)
    for i in range(batch_size):
        ti = rng.randint(0, N_TOPOS)
        V  = V_all[ti]
        # 随机选K个观测节点（不含bus 0=变电站）
        K  = rng.randint(K_min, K_max + 1)
        obs = rng.choice(range(1, 33), K, replace=False)
        # 通信缺失（IP-C）
        if missing_rate > 0:
            keep = rng.rand(K) > missing_rate
            obs  = obs[keep]
        if len(obs) == 0:
            obs = rng.choice(range(1, 33), 1, replace=False)
        # 填充输入
        X[i, obs] = V[obs] + rng.normal(0, sigma, len(obs))
        X[i, 33 + obs] = 1.0
        y[i] = ti
    return X, y

for step in range(1, N_STEPS + 1):
    # 80%正常训练，20%含通信缺失（IP-C数据增强）
    miss = 0.2 if rng.rand() < 0.2 else 0.0
    X_np, y_np = make_batch(rng, BATCH_SIZE, V_all, N_TOPOS,
                             SIGMA, K_MIN, K_MAX, missing_rate=miss)
    X_t = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_np, dtype=torch.long).to(DEVICE)

    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_t), y_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    loss_history.append(loss.item())

    if step % LOG_EVERY == 0 or step == N_STEPS:
        elapsed = time.perf_counter() - t_start
        avg_loss = np.mean(loss_history[-LOG_EVERY:])
        eta = elapsed / step * (N_STEPS - step)
        print(f"  Step {step:>6}/{N_STEPS}  loss={avg_loss:.4f}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

total_time = time.perf_counter() - t_start
print(f"  Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")

# ── 5. 评估 ──────────────────────────────────────────────────────────
print(f"\n[4/4] Evaluation...")
model.eval()

def ais_posterior(obs_v, obs_nodes, V_all, sigma):
    """精确贝叶斯后验（穷举）"""
    diff = (V_all[:, obs_nodes] - obs_v) / sigma
    ll = -0.5 * np.sum(diff**2, axis=1)
    ll -= ll.max(); w = np.exp(ll)
    return w / w.sum()

rng_te = np.random.RandomState(99)
results = {'K': [], 'acc_nre': [], 'acc_ais': [], 'kl': []}

for K in [5, 8, 12, 15, 20, 25]:
    acc_nre, acc_ais, kls = [], [], []
    for _ in range(200):
        ti = rng_te.randint(0, N_TOPOS)
        obs = np.sort(rng_te.choice(range(1,33), K, replace=False))
        obs_v = V_all[ti][obs] + rng_te.normal(0, SIGMA*0.3, K)

        # AIS（精确后验）
        p_ais = ais_posterior(obs_v, obs, V_all, SIGMA)
        acc_ais.append(int(np.argmax(p_ais) == ti))

        # NRE（掩码模型）
        x = np.zeros(66, dtype=np.float32)
        x[obs] = obs_v; x[33+obs] = 1.0
        p_nre = model.posterior(x)
        acc_nre.append(int(np.argmax(p_nre) == ti))

        kl = np.sum(p_ais * np.log((p_ais+1e-10)/(p_nre+1e-10)))
        kls.append(kl)

    results['K'].append(K)
    results['acc_nre'].append(np.mean(acc_nre))
    results['acc_ais'].append(np.mean(acc_ais))
    results['kl'].append(np.mean(kls))

print(f"\n  {'K':>4}  {'NRE_acc':>8}  {'AIS_acc':>8}  {'KL':>8}  {'gap':>6}")
print(f"  {'-'*40}")
for i in range(len(results['K'])):
    gap = results['acc_ais'][i] - results['acc_nre'][i]
    print(f"  {results['K'][i]:>4}  {results['acc_nre'][i]:>8.3f}  "
          f"{results['acc_ais'][i]:>8.3f}  {results['kl'][i]:>8.4f}  "
          f"{gap:>+6.3f}")

# 缺失数据测试（IP-C验证）
print(f"\n  Missing data test (IP-C):")
print(f"  {'miss%':>6}  {'naive_acc':>10}  {'robust_acc':>11}")

for miss_rate in [0.0, 0.10, 0.20, 0.30]:
    acc_naive, acc_robust = [], []
    for _ in range(300):
        ti = rng_te.randint(0, N_TOPOS)
        K_test = 15
        obs_all = np.sort(rng_te.choice(range(1,33), K_test, replace=False))
        obs_v_all = V_all[ti][obs_all] + rng_te.normal(0, SIGMA*0.3, K_test)

        # 模拟通信缺失
        keep = rng_te.rand(K_test) > miss_rate
        if not np.any(keep): keep[0] = True
        obs_avail = obs_all[keep]
        obs_v_avail = obs_v_all[keep]

        # 朴素NRE（用可用数据填充，不知道哪些缺失）
        x_naive = np.zeros(66, dtype=np.float32)
        x_naive[obs_avail] = obs_v_avail
        x_naive[33+obs_avail] = 1.0

        # 鲁棒NRE（正确掩码）
        x_robust = np.zeros(66, dtype=np.float32)
        x_robust[obs_avail] = obs_v_avail
        x_robust[33+obs_avail] = 1.0  # 同上（掩码已正确编码）

        # 对于naive：假装全部节点都有测量（用0填充缺失）
        x_naive2 = np.zeros(66, dtype=np.float32)
        x_naive2[obs_all] = obs_v_all * keep  # 缺失位填0但mask仍为1
        x_naive2[33+obs_all] = 1.0  # 错误地声称全部可观

        p_naive  = model.posterior(x_naive2)
        p_robust = model.posterior(x_robust)
        acc_naive.append(int(np.argmax(p_naive) == ti))
        acc_robust.append(int(np.argmax(p_robust) == ti))

    print(f"  {miss_rate*100:>5.0f}%  {np.mean(acc_naive):>10.3f}  "
          f"{np.mean(acc_robust):>11.3f}")

# 推断速度
x_test = np.zeros((1,66), dtype=np.float32); x_test[0,1:13]=0.95; x_test[0,34:46]=1.0
x_t = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(10000): _ = model(x_t)
nre_ms = (time.perf_counter()-t0)/10000*1000
print(f"\n  NRE inference: {nre_ms:.3f}ms/query")

# ── 6. 保存模型 ──────────────────────────────────────────────────────
torch.save({
    'model_state': model.state_dict(),
    'N_TOPOS': N_TOPOS,
    'SIGMA': SIGMA,
    'voltages': V_all,
    'topos_raw': topos_raw,
    'results': results,
}, SAVE_PATH)
print(f"\n  Model saved: {SAVE_PATH}")

# ── 最终汇报 ──────────────────────────────────────────────────────────
mean_kl = np.mean(results['kl'])
mean_gap = np.mean([results['acc_ais'][i]-results['acc_nre'][i]
                    for i in range(len(results['K']))])

print(f"""
{'='*55}
TRAINING COMPLETE
{'='*55}
  Training time:  {total_time:.1f}s ({total_time/60:.1f} min)
  Mean KL(AIS||NRE): {mean_kl:.4f}  {'EXCELLENT (<0.05)' if mean_kl<0.05 else 'OK' if mean_kl<0.1 else 'NEEDS MORE TRAINING'}
  Mean acc gap:      {mean_gap:.3f}  {'EXCELLENT (<0.02)' if mean_gap<0.02 else 'OK (<0.05)' if mean_gap<0.05 else 'LARGE'}
  Model saved to: {SAVE_PATH}

  Ready for:
    IP1 experiments  (posterior quality vs AIS)
    IP4 experiments  (H(K) curve, full K range)
    IP-A experiments (BOED sequential sensing)
    IP-C experiments (missing data robustness)
{'='*55}
""")
