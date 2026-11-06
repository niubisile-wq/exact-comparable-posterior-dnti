# -*- coding: utf-8 -*-
import copy, os, time, warnings
import numpy as np
import torch
import torch.nn as nn
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = os.path.join(os.environ['USERPROFILE'], 'Desktop', '配电网实验_临时')
OUT_PATH = os.path.join(SAVE_DIR, 'threephase_13bus_fixed_result.txt')
SEEDS = [42, 123, 456]
SIGMA = 0.005
LF_GRID = np.linspace(0.85, 1.15, 21, dtype=np.float32)
BATCH = 256
STEPS = 12000
LR = 3e-4
N_BUS = 13
BRANCHES = [
    (0,1,0.08,0.05),(1,2,0.10,0.06),(2,3,0.09,0.05),(3,4,0.11,0.07),
    (4,5,0.12,0.07),(5,6,0.10,0.06),(2,7,0.08,0.05),(7,8,0.09,0.05),
    (8,9,0.11,0.07),(9,10,0.12,0.07),(3,11,0.10,0.06),(11,12,0.09,0.05)
]
TIES = [(6,10,0.04,0.04),(5,12,0.04,0.04),(4,8,0.04,0.04)]

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d,d), nn.LayerNorm(d))
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(x + self.net(x))

class NRE3ph(nn.Module):
    def __init__(self, in_dim, n_topo):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU())
        self.res1 = ResBlock(256)
        self.res2 = ResBlock(256)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, n_topo))
    def forward(self, x):
        h = self.embed(x)
        h = self.res1(h)
        h = self.res2(h)
        return self.head(h)


def make_loads():
    loads = []
    p = np.zeros(N_BUS, dtype=np.float32)
    for bus in range(1, N_BUS):
        pa = 0.025 + 0.006 * ((bus * 2) % 5)
        pb = 0.022 + 0.005 * ((bus * 3) % 4)
        pc = 0.021 + 0.004 * ((bus * 5) % 4)
        qa = 0.36 * pa
        qb = 0.34 * pb
        qc = 0.33 * pc
        loads.append((bus, pa, pb, pc, qa, qb, qc))
        p[bus] = pa + pb + pc
    p /= max(np.max(p), 1e-9)
    return loads, p

BASE_LOADS, BASE_P = make_loads()
INSTALLED = np.unique(np.rint(np.linspace(1, N_BUS - 1, 8)).astype(int))
K = len(INSTALLED)


def build_net():
    net = pp.create_empty_network()
    for _ in range(N_BUS):
        pp.create_bus(net, vn_kv=12.66)
    pp.create_ext_grid(net, bus=0, vm_pu=1.0,
        s_sc_max_mva=1000, rx_max=0.1, rx_min=0.1,
        x0x_max=0.5, x0x_min=0.5, r0x0_max=0.1, r0x0_min=0.1)
    for f,t,r,x in BRANCHES:
        pp.create_line_from_parameters(net, f, t, 1.0, r, x, 0.0, 999.0,
            r0_ohm_per_km=r*3.0, x0_ohm_per_km=x*3.0, c0_nf_per_km=0.0, in_service=True)
    for f,t,r,x in TIES:
        pp.create_line_from_parameters(net, f, t, 1.0, r, x, 0.0, 999.0,
            r0_ohm_per_km=r*3.0, x0_ohm_per_km=x*3.0, c0_nf_per_km=0.0, in_service=False)
    for bus, pa, pb, pc, qa, qb, qc in BASE_LOADS:
        pp.create_asymmetric_load(net, bus=bus,
            p_a_mw=pa, p_b_mw=pb, p_c_mw=pc,
            q_a_mvar=qa, q_b_mvar=qb, q_c_mvar=qc)
    return net


def enum_topos():
    ne = [(f,t) for f,t,_,_ in BRANCHES]
    te = [(f,t) for f,t,_,_ in TIES]
    G = nx.Graph(); G.add_nodes_from(range(N_BUS)); G.add_edges_from(ne)
    topos = [list(range(len(ne)))]
    seen = {frozenset(topos[0])}
    for ti, tie in enumerate(te):
        path = nx.shortest_path(G, tie[0], tie[1])
        for i in range(len(path)-1):
            oe = frozenset((path[i], path[i+1]))
            ni = [j for j,e in enumerate(ne) if frozenset(e) != oe]
            topo = ni + [len(ne)+ti]
            key = frozenset(topo)
            if key in seen:
                continue
            Gt = nx.Graph(); Gt.add_nodes_from(range(N_BUS)); Gt.add_edges_from([ne[j] for j in ni] + [tie])
            if nx.is_tree(Gt):
                seen.add(key)
                topos.append(topo)
    return topos, ne, te


def run_pf(net_base, topo, ne, te, lf):
    net = copy.deepcopy(net_base)
    net.asymmetric_load['p_a_mw'] *= lf
    net.asymmetric_load['p_b_mw'] *= lf
    net.asymmetric_load['p_c_mw'] *= lf
    net.asymmetric_load['q_a_mvar'] *= lf
    net.asymmetric_load['q_b_mvar'] *= lf
    net.asymmetric_load['q_c_mvar'] *= lf
    n_ne = len(ne)
    active_ne = {x for x in topo if x < n_ne}
    active_te = {x - n_ne for x in topo if x >= n_ne}
    for li in range(n_ne):
        net.line.at[net.line.index[li], 'in_service'] = li in active_ne
    for li in range(len(te)):
        net.line.at[net.line.index[n_ne + li], 'in_service'] = li in active_te
    try:
        pp.runpp_3ph(net, numba=False, max_iteration=30, tolerance_va_degree=1e-5)
        if net.converged:
            return np.stack([net.res_bus_3ph.vm_a_pu.values, net.res_bus_3ph.vm_b_pu.values, net.res_bus_3ph.vm_c_pu.values], axis=1).astype(np.float32)
    except Exception:
        pass
    return None


def build_library(net, topos, ne, te):
    vlib = np.zeros((len(topos), len(LF_GRID), N_BUS, 3), dtype=np.float32)
    keep = np.ones(len(topos), dtype=bool)
    failed = 0
    for i, topo in enumerate(topos):
        ok = True
        for j, lf in enumerate(LF_GRID):
            v = run_pf(net, topo, ne, te, float(lf))
            if v is None:
                ok = False
                failed += 1
                break
            vlib[i, j] = v
        keep[i] = ok
    kept = [topos[i] for i in range(len(topos)) if keep[i]]
    return kept, vlib[keep], failed


def build_x(obs):
    x = np.zeros(N_BUS * 5, dtype=np.float32)
    x[INSTALLED] = (obs[:,0] - 1.0) / SIGMA
    x[N_BUS + INSTALLED] = (obs[:,1] - 1.0) / SIGMA
    x[2*N_BUS + INSTALLED] = (obs[:,2] - 1.0) / SIGMA
    x[3*N_BUS + INSTALLED] = 1.0
    x[4*N_BUS:5*N_BUS] = BASE_P
    return x


def infer(model, obs):
    x = build_x(obs)
    with torch.no_grad():
        logits = model(torch.tensor(x).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    p = np.exp(logits - logits.max()); p /= p.sum()
    return p.astype(np.float32)


def enum_post(vlib, obs, lf_idx):
    da = (vlib[:, lf_idx, :, 0][:, INSTALLED] - obs[:,0]) / SIGMA
    db = (vlib[:, lf_idx, :, 1][:, INSTALLED] - obs[:,1]) / SIGMA
    dc = (vlib[:, lf_idx, :, 2][:, INSTALLED] - obs[:,2]) / SIGMA
    ll = -0.5 * (np.sum(da*da, axis=1) + np.sum(db*db, axis=1) + np.sum(dc*dc, axis=1))
    q = np.exp(ll - ll.max()); q /= q.sum()
    return q.astype(np.float32)


def gen_batch(rng, vlib, n_topos):
    xs = np.zeros((BATCH, N_BUS * 5), dtype=np.float32)
    ys = np.zeros(BATCH, dtype=np.int64)
    for i in range(BATCH):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, len(LF_GRID))
        obs = vlib[ti, lf_idx, INSTALLED, :] + rng.normal(0.0, SIGMA, size=(K,3))
        xs[i] = build_x(obs)
        ys[i] = ti
    return torch.tensor(xs, dtype=torch.float32, device=DEVICE), torch.tensor(ys, dtype=torch.long, device=DEVICE)


def evaluate(model, vlib, n_topos):
    rng = np.random.RandomState(77)
    acc_e, acc_n, kls = [], [], []
    for _ in range(500):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, len(LF_GRID))
        obs = vlib[ti, lf_idx, INSTALLED, :] + rng.normal(0.0, SIGMA, size=(K,3))
        pe = enum_post(vlib, obs, lf_idx)
        pn = infer(model, obs)
        acc_e.append(int(np.argmax(pe) == ti))
        acc_n.append(int(np.argmax(pn) == ti))
        kls.append(max(0.0, float(np.sum(pe * np.log((pe + 1e-10) / (pn + 1e-10))))))
    return float(np.mean(acc_e)), float(np.mean(acc_n)), float(np.mean(kls))


def main():
    net = build_net()
    raw_topos, ne, te = enum_topos()
    kept_topos, vlib, failed = build_library(net, raw_topos, ne, te)
    n_topos = len(kept_topos)
    t0 = time.time()
    for topo in kept_topos:
        _ = run_pf(net, topo, ne, te, 1.0)
    enum_ms = (time.time() - t0) / n_topos * 1000.0
    lines = [f'Device={DEVICE}', f'Benchmark=13bus_fixed n_bus={N_BUS} K={K} raw_topos={len(raw_topos)} kept_topos={n_topos} failed_pf={failed}']
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        model = NRE3ph(N_BUS * 5, n_topos).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=1e-5)
        loss_fn = nn.CrossEntropyLoss()
        rng = np.random.RandomState(seed)
        t_train = time.time()
        model.train()
        for _ in range(STEPS):
            xb, yb = gen_batch(rng, vlib, n_topos)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        train_sec = time.time() - t_train
        exact, nre, kl = evaluate(model, vlib, n_topos)
        dummy_obs = np.ones((K,3), dtype=np.float32)
        t_inf = time.time()
        for _ in range(500):
            _ = infer(model, dummy_obs)
        nre_ms = (time.time() - t_inf) / 500.0 * 1000.0
        row = {'seed': seed, 'exact': exact, 'nre': nre, 'gap': exact - nre, 'kl': kl, 'enum_ms': enum_ms, 'nre_ms': nre_ms, 'speedup': enum_ms / max(nre_ms, 1e-9), 'train_sec': train_sec}
        rows.append(row)
        lines.append('seed={seed} exact={exact:.4f} nre={nre:.4f} gap={gap:.4f} kl={kl:.4f} enum_ms={enum_ms:.3f} nre_ms={nre_ms:.3f} speedup={speedup:.1f}x train_sec={train_sec:.1f}'.format(**row))
        with open(OUT_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        torch.save({'model_state': model.state_dict(), 'seed': seed, 'name': '13bus_fixed', 'n_topos': n_topos}, os.path.join(SAVE_DIR, f'nre_3ph_13bus_fixed_seed{seed}.pt'))
    lines.append('mean_exact={:.4f} mean_nre={:.4f} mean_gap={:.4f} mean_kl={:.4f} mean_speedup={:.1f}x'.format(
        np.mean([r['exact'] for r in rows]), np.mean([r['nre'] for r in rows]), np.mean([r['gap'] for r in rows]), np.mean([r['kl'] for r in rows]), np.mean([r['speedup'] for r in rows])
    ))
    lines.append('Boundary')
    lines.append('- Synthetic unbalanced three-phase 13-bus fixed-deployment benchmark.')
    lines.append('- Serves as a middle-scale bridge between the earlier 10-bus extension and the new 37-bus benchmark.')
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(OUT_PATH)

if __name__ == '__main__':
    main()
