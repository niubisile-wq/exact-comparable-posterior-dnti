# -*- coding: utf-8 -*-
"""
Step5-B: Fisher-OED传感器选址基线（频率派，对比IP-A的贝叶斯BOED）
方法：序贯贪婪最大化Fisher信息行列式（D-optimal design）
      每步选使FIM行列式增量最大的候选节点
两个网络：33-bus + 69-bus
"""
import warnings
import numpy as np
warnings.filterwarnings('ignore')

SAVE_DIR = r"<LOCAL_WORKSPACE>"
SIGMA    = 0.009; N_LF = 101; K_FIXED = 20; K_MAX = 12

def run_fisher_oed(net_name, N_BUS, boed_ref):
    print(f"\n{'='*55}")
    print(f"Fisher-OED vs BOED vs Random: {net_name}")

    if net_name == '33bus':
        ckpt = __import__('torch').load(f"{SAVE_DIR}\\nre_ipc_loadaware.pt",
                                         map_location='cpu', weights_only=False)
        V_library   = ckpt['V_library']
        lf_grid     = ckpt['lf_grid']
        N_TOPOS     = ckpt['N_TOPOS']
    else:
        dat = np.load(f"{SAVE_DIR}\\v_library_69bus.npz")
        V_library   = dat['V_library']
        lf_grid     = dat['lf_grid']
        N_TOPOS     = V_library.shape[0]

    CANDIDATES = list(range(1, N_BUS))
    lf_fixed_idx = N_LF // 2  # 在固定lf=1.0下计算Fisher信息

    # ── AIS后验（粗网格，供准确率评估用）──────────────────────────────────
    N_LF_C = 11
    lf_idx_c = np.round(np.linspace(0, N_LF-1, N_LF_C)).astype(int)
    V_lib_c  = V_library[:, lf_idx_c, :]

    def ais_posterior(obs_nodes, obs_vals):
        if not obs_nodes: return np.ones(N_TOPOS)/N_TOPOS
        diff = (V_lib_c[:, :, obs_nodes] - np.array(obs_vals)) / SIGMA
        ll = -0.5*np.sum(diff**2, axis=2); ll -= ll.max()
        w = np.exp(ll).sum(axis=1); return w/w.sum()

    def entropy(p): p=np.clip(p,1e-10,1); return -np.sum(p*np.log(p))

    # ── Fisher-OED核心：序贯D-optimal ─────────────────────────────────────
    # FIM(θ|x_S) = Σ_j∈S (dlog p(x_j|θ)/dθ)^T (dlog p(x_j|θ)/dθ) / σ²
    # 对于高斯观测：FIM_jj = (dV_j/dθ)^T (dV_j/dθ) / σ²
    # 实用近似：用拓扑电压差异矩阵近似Fisher信息
    # Jacobian近似：J[i,j] = (V_topo_i[node_j] - V_mean[node_j]) / σ
    # FIM ≈ J^T J / N_TOPOS

    V_fixed = V_library[:, lf_fixed_idx, :]  # (N_TOPOS, N_BUS)
    V_mean  = V_fixed.mean(axis=0, keepdims=True)  # (1, N_BUS)
    J_full  = (V_fixed - V_mean) / SIGMA            # (N_TOPOS, N_BUS) 近似Jacobian

    def fisher_score(obs_nodes, candidate):
        """当前已选obs_nodes情况下，加入candidate的FIM增量（log行列式增量近似）"""
        nodes = obs_nodes + [candidate]
        J_sub = J_full[:, nodes]          # (N_TOPOS, K+1)
        FIM   = J_sub.T @ J_sub           # (K+1, K+1)
        # D-optimal: maximize log|FIM|
        try:
            sign, ld = np.linalg.slogdet(FIM)
            return ld if sign > 0 else -np.inf
        except:
            return -np.inf

    def fisher_select(obs_nodes, candidates):
        """贪婪选使log|FIM|最大的下一个节点"""
        scores = [fisher_score(obs_nodes, c) for c in candidates]
        return candidates[int(np.argmax(scores))]

    # ── 评估 ──────────────────────────────────────────────────────────────
    N_TEST = 500 if net_name == '33bus' else 200
    rng_main = np.random.RandomState(2024)
    test_cases = []
    for _ in range(N_TEST):
        ti     = rng_main.randint(0, N_TOPOS)
        lf_idx = rng_main.randint(0, N_LF)
        full_v = V_library[ti, lf_idx, :] + rng_main.normal(0, SIGMA, N_BUS)
        test_cases.append((ti, lf_idx, full_v))

    acc_fisher  = np.zeros(K_MAX)
    acc_random  = np.zeros(K_MAX)
    H_fisher    = np.zeros(K_MAX)
    H_random    = np.zeros(K_MAX)

    import time; t0 = time.time()
    for ci, (true_ti, lf_idx, full_v) in enumerate(test_cases):
        if ci % 100 == 0:
            print(f"  [{net_name}] Case {ci}/{N_TEST}  {time.time()-t0:.0f}s", flush=True)

        # Fisher-OED
        sel, remain = [], list(CANDIDATES)
        sv = []
        for k in range(K_MAX):
            node = fisher_select(sel, remain)
            sel.append(node); sv.append(full_v[node]); remain.remove(node)
            p = ais_posterior(sel, sv)
            acc_fisher[k] += int(np.argmax(p)==true_ti)
            H_fisher[k]   += entropy(p)

        # Random
        rng_rand = np.random.RandomState(ci*17+3)
        perm = list(CANDIDATES); rng_rand.shuffle(perm)
        sel, sv = [], []
        for k in range(K_MAX):
            sel.append(perm[k]); sv.append(full_v[perm[k]])
            p = ais_posterior(sel, sv)
            acc_random[k] += int(np.argmax(p)==true_ti)
            H_random[k]   += entropy(p)

    acc_fisher /= N_TEST; acc_random /= N_TEST
    H_fisher   /= N_TEST; H_random   /= N_TEST

    print(f"\n  Fisher-OED vs Random vs BOED(ref):")
    print(f"  {'K':>3}  {'Fisher':>8}  {'Random':>8}  {'BOED(ref)':>10}")
    boed_vals = boed_ref[net_name]
    for k in range(K_MAX):
        boed_v = boed_vals[k] if k < len(boed_vals) else 'N/A'
        boed_str = f"{boed_v:.3f}" if isinstance(boed_v, float) else boed_v
        print(f"  {k+1:>3}  {acc_fisher[k]:>8.3f}  {acc_random[k]:>8.3f}  {boed_str:>10}")

    return acc_fisher, acc_random

# BOED参考值（来自boed_v2_result.txt和boed_69bus_result.txt）
boed_ref = {
    '33bus': [0.150, 0.278, 0.450, 0.574, 0.666, 0.718, 0.754, 0.782, 0.790, 0.796, 0.804, 0.812],
    '69bus': [0.095, 0.170, 0.315, 0.395, 0.500, 0.595, 0.670, 0.705, 0.755, 0.780, 0.790, 0.785],
}

all_fisher = {}
all_random = {}
for net_name, N_BUS in [('33bus', 33), ('69bus', 69)]:
    af, ar = run_fisher_oed(net_name, N_BUS, boed_ref)
    all_fisher[net_name] = af
    all_random[net_name] = ar

print(f"\n{'='*60}")
print("FISHER-OED vs BOED SUMMARY")
for net_name in ['33bus', '69bus']:
    af = all_fisher[net_name]; br = boed_ref[net_name]
    print(f"\n  {net_name}:")
    print(f"    K=4:  Fisher={af[3]:.3f}  BOED={br[3]:.3f}  delta={br[3]-af[3]:+.3f}")
    print(f"    K=7:  Fisher={af[6]:.3f}  BOED={br[6]:.3f}  delta={br[6]-af[6]:+.3f}")

with open(f"{SAVE_DIR}\\step5_fisher_result.txt", 'w', encoding='utf-8') as f:
    for net_name in ['33bus', '69bus']:
        af = all_fisher[net_name]; ar = all_random[net_name]; br = boed_ref[net_name]
        f.write(f"{net_name} Fisher-OED: K=4={af[3]:.3f} K=7={af[6]:.3f}\n")
        f.write(f"{net_name} Random:     K=4={ar[3]:.3f} K=7={ar[6]:.3f}\n")
        f.write(f"{net_name} BOED(ref):  K=4={br[3]:.3f} K=7={br[6]:.3f}\n\n")
print(f"Saved: {SAVE_DIR}\\step5_fisher_result.txt")
