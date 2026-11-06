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
SEEDS = [42, 123, 456]
SIGMA = 0.005
LF_GRID = np.linspace(0.85, 1.15, 25, dtype=np.float32)
BATCH = 256

CONFIGS = {
    '13bus': {
        'n_bus': 13,
        'k': 7,
        'steps': 18000,
        'lr': 3e-4,
        'branches': [
            (0,1,0.08,0.05),(1,2,0.10,0.06),(2,3,0.09,0.05),(3,4,0.11,0.07),
            (4,5,0.12,0.07),(5,6,0.10,0.06),(2,7,0.08,0.05),(7,8,0.09,0.05),
            (8,9,0.11,0.07),(9,10,0.12,0.07),(3,11,0.10,0.06),(11,12,0.09,0.05)
        ],
        'ties': [(6,10,0.04,0.04),(5,12,0.04,0.04),(4,8,0.04,0.04)],
    },
    '37bus': {
        'n_bus': 37,
        'k': 18,
        'steps': 22000,
        'lr': 2.5e-4,
        'branches': [
            (0,1,0.09,0.06),(1,2,0.10,0.06),(2,3,0.11,0.07),(3,4,0.10,0.06),
            (4,5,0.12,0.08),(5,6,0.10,0.06),(6,7,0.11,0.07),(7,8,0.10,0.06),
            (8,9,0.12,0.08),(9,10,0.11,0.07),(10,11,0.10,0.06),(11,12,0.12,0.08),
            (12,13,0.11,0.07),(13,14,0.10,0.06),(14,15,0.12,0.08),(15,16,0.10,0.06),
            (16,17,0.11,0.07),(3,18,0.09,0.06),(18,19,0.10,0.06),(19,20,0.11,0.07),
            (20,21,0.10,0.06),(21,22,0.12,0.08),(6,23,0.09,0.06),(23,24,0.11,0.07),
            (24,25,0.10,0.06),(25,26,0.12,0.08),(9,27,0.09,0.06),(27,28,0.11,0.07),
            (28,29,0.10,0.06),(12,30,0.09,0.06),(30,31,0.10,0.06),(31,32,0.11,0.07),
            (15,33,0.09,0.06),(33,34,0.10,0.06),(34,35,0.11,0.07),(35,36,0.10,0.06)
        ],
        'ties': [(17,22,0.04,0.04),(14,26,0.04,0.04),(10,29,0.04,0.04),(21,36,0.04,0.04)],
    },
}

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
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


def make_loads(n_bus):
    loads = []
    for bus in range(1, n_bus):
        pa = 0.020 + 0.004 * ((bus * 2) % 7)
        pb = 0.018 + 0.005 * ((bus * 3) % 6)
        pc = 0.019 + 0.0045 * ((bus * 5) % 5)
        qa = 0.35 * pa
        qb = 0.37 * pb
        qc = 0.33 * pc
        loads.append((bus, pa, pb, pc, qa, qb, qc))
    return loads


def build_3ph_net(cfg):
    n_bus = cfg['n_bus']
    net = pp.create_empty_network()
    for _ in range(n_bus):
        pp.create_bus(net, vn_kv=12.66)
    pp.create_ext_grid(net, bus=0, vm_pu=1.0,
        s_sc_max_mva=1000, rx_max=0.1, rx_min=0.1,
        x0x_max=0.5, x0x_min=0.5, r0x0_max=0.1, r0x0_min=0.1)
    for f, t, r, x in cfg['branches']:
        pp.create_line_from_parameters(net, f, t, 1.0, r, x, 0.0, 999.0,
            r0_ohm_per_km=r*3.0, x0_ohm_per_km=x*3.0, c0_nf_per_km=0.0, in_service=True)
    for f, t, r, x in cfg['ties']:
        pp.create_line_from_parameters(net, f, t, 1.0, r, x, 0.0, 999.0,
            r0_ohm_per_km=r*3.0, x0_ohm_per_km=x*3.0, c0_nf_per_km=0.0, in_service=False)
    loads = make_loads(n_bus)
    for bus, pa, pb, pc, qa, qb, qc in loads:
        pp.create_asymmetric_load(net, bus=bus,
            p_a_mw=pa, p_b_mw=pb, p_c_mw=pc,
            q_a_mvar=qa, q_b_mvar=qb, q_c_mvar=qc)
    ne = [(int(f), int(t)) for f, t, _, _ in cfg['branches']]
    te = [(int(f), int(t)) for f, t, _, _ in cfg['ties']]
    return net, ne, te


def enum_topos(ne, te, n):
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(ne)
    topos = [list(range(len(ne)))]
    seen = {frozenset(topos[0])}
    for ti, tie in enumerate(te):
        try:
            path = nx.shortest_path(G, tie[0], tie[1])
        except Exception:
            continue
        for i in range(len(path) - 1):
            oe = frozenset((path[i], path[i + 1]))
            ni = [j for j, e in enumerate(ne) if frozenset(e) != oe]
            topo = ni + [len(ne) + ti]
            key = frozenset(topo)
            if key in seen:
                continue
            Gt = nx.Graph(); Gt.add_nodes_from(range(n)); Gt.add_edges_from([ne[j] for j in ni] + [tie])
            if nx.is_tree(Gt):
                seen.add(key)
                topos.append(topo)
    return topos


def run_pf_3ph(net_base, topo, ne, te, lf):
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
            return np.stack([
                net.res_bus_3ph.vm_a_pu.values,
                net.res_bus_3ph.vm_b_pu.values,
                net.res_bus_3ph.vm_c_pu.values,
            ], axis=1).astype(np.float32)
    except Exception:
        pass
    return None


def build_library(net, ne, te, topos, n_bus):
    vlib = np.zeros((len(topos), len(LF_GRID), n_bus, 3), dtype=np.float32)
    keep = np.ones(len(topos), dtype=bool)
    failed = 0
    for i, topo in enumerate(topos):
        ok = True
        for j, lf in enumerate(LF_GRID):
            v = run_pf_3ph(net, topo, ne, te, float(lf))
            if v is None:
                ok = False
                failed += 1
                break
            vlib[i, j] = v
        keep[i] = ok
    kept_topos = [topos[i] for i in range(len(topos)) if keep[i]]
    return kept_topos, vlib[keep], failed


def infer(model, installed, obs_v3, n_bus):
    x = np.zeros(n_bus * 4, dtype=np.float32)
    x[installed] = obs_v3[:, 0]
    x[n_bus + installed] = obs_v3[:, 1]
    x[2 * n_bus + installed] = obs_v3[:, 2]
    x[3 * n_bus + installed] = 1.0
    with torch.no_grad():
        logits = model(torch.tensor(x).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return p.astype(np.float32)


def enum_post(vlib, installed, obs_v3, lf_idx):
    da = (vlib[:, lf_idx, :, 0][:, installed] - obs_v3[:, 0]) / SIGMA
    db = (vlib[:, lf_idx, :, 1][:, installed] - obs_v3[:, 1]) / SIGMA
    dc = (vlib[:, lf_idx, :, 2][:, installed] - obs_v3[:, 2]) / SIGMA
    ll = -0.5 * (np.sum(da * da, axis=1) + np.sum(db * db, axis=1) + np.sum(dc * dc, axis=1))
    q = np.exp(ll - ll.max())
    q /= q.sum()
    return q.astype(np.float32)


def gen_batch(rng, vlib, n_bus, k, n_topos, batch):
    xs = np.zeros((batch, n_bus * 4), dtype=np.float32)
    ys = np.zeros(batch, dtype=np.int64)
    for i in range(batch):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, len(LF_GRID))
        installed = np.sort(rng.choice(np.arange(1, n_bus), k, replace=False))
        obs = vlib[ti, lf_idx, installed, :] + rng.normal(0.0, SIGMA, size=(k, 3))
        xs[i, installed] = obs[:, 0]
        xs[i, n_bus + installed] = obs[:, 1]
        xs[i, 2 * n_bus + installed] = obs[:, 2]
        xs[i, 3 * n_bus + installed] = 1.0
        ys[i] = ti
    return torch.tensor(xs, dtype=torch.float32, device=DEVICE), torch.tensor(ys, dtype=torch.long, device=DEVICE)


def evaluate(model, vlib, n_bus, k, n_topos, eval_n=600):
    rng = np.random.RandomState(77)
    acc_e, acc_n, kls = [], [], []
    for _ in range(eval_n):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, len(LF_GRID))
        installed = np.sort(rng.choice(np.arange(1, n_bus), k, replace=False))
        obs = vlib[ti, lf_idx, installed, :] + rng.normal(0.0, SIGMA, size=(k, 3))
        p_e = enum_post(vlib, installed, obs, lf_idx)
        p_n = infer(model, installed, obs, n_bus)
        acc_e.append(int(np.argmax(p_e) == ti))
        acc_n.append(int(np.argmax(p_n) == ti))
        kls.append(max(0.0, float(np.sum(p_e * np.log((p_e + 1e-10) / (p_n + 1e-10))))))
    return float(np.mean(acc_e)), float(np.mean(acc_n)), float(np.mean(kls))


def run_suite(name, cfg):
    n_bus = cfg['n_bus']
    net, ne, te = build_3ph_net(cfg)
    raw_topos = enum_topos(ne, te, n_bus)
    kept_topos, vlib, failed = build_library(net, ne, te, raw_topos, n_bus)
    if len(kept_topos) == 0:
        raise RuntimeError(f'{name}: no feasible topologies')
    t0 = time.time()
    for topo in kept_topos:
        _ = run_pf_3ph(net, topo, ne, te, 1.0)
    enum_ms = (time.time() - t0) / len(kept_topos) * 1000.0
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = NRE3ph(n_bus * 4, len(kept_topos)).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['steps'], eta_min=1e-5)
        loss_fn = nn.CrossEntropyLoss()
        rng = np.random.RandomState(seed)
        t_train = time.time()
        model.train()
        for _ in range(cfg['steps']):
            xb, yb = gen_batch(rng, vlib, n_bus, cfg['k'], len(kept_topos), BATCH)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        train_sec = time.time() - t_train
        exact, nre, kl = evaluate(model, vlib, n_bus, cfg['k'], len(kept_topos))
        dummy_obs = np.ones((cfg['k'], 3), dtype=np.float32)
        installed = np.arange(1, cfg['k'] + 1)
        t_inf = time.time()
        for _ in range(800):
            _ = infer(model, installed, dummy_obs, n_bus)
        nre_ms = (time.time() - t_inf) / 800.0 * 1000.0
        rows.append({
            'seed': seed,
            'exact': exact,
            'nre': nre,
            'gap': exact - nre,
            'kl': kl,
            'enum_ms': enum_ms,
            'nre_ms': nre_ms,
            'speedup': enum_ms / max(nre_ms, 1e-9),
            'train_sec': train_sec,
        })
        torch.save({'model_state': model.state_dict(), 'seed': seed, 'name': name, 'n_topos': len(kept_topos)},
            os.path.join(SAVE_DIR, f'nre_3ph_{name}_seed{seed}.pt'))
    return {
        'name': name,
        'n_bus': n_bus,
        'k': cfg['k'],
        'raw_topos': len(raw_topos),
        'kept_topos': len(kept_topos),
        'failed_pf': failed,
        'rows': rows,
    }


def main():
    lines = [f'Device={DEVICE}']
    for name in ['13bus', '37bus']:
        suite = run_suite(name, CONFIGS[name])
        lines.append(f"Benchmark={suite['name']} n_bus={suite['n_bus']} K={suite['k']} raw_topos={suite['raw_topos']} kept_topos={suite['kept_topos']} failed_pf={suite['failed_pf']}")
        for r in suite['rows']:
            lines.append('seed={seed} exact={exact:.4f} nre={nre:.4f} gap={gap:.4f} kl={kl:.4f} enum_ms={enum_ms:.3f} nre_ms={nre_ms:.3f} speedup={speedup:.1f}x train_sec={train_sec:.1f}'.format(**r))
        lines.append('mean_exact={:.4f} mean_nre={:.4f} mean_gap={:.4f} mean_kl={:.4f} mean_speedup={:.1f}x'.format(
            np.mean([r['exact'] for r in suite['rows']]),
            np.mean([r['nre'] for r in suite['rows']]),
            np.mean([r['gap'] for r in suite['rows']]),
            np.mean([r['kl'] for r in suite['rows']]),
            np.mean([r['speedup'] for r in suite['rows']]),
        ))
        lines.append('')
    lines.append('Boundary')
    lines.append('- These are synthetic unbalanced three-phase feeder benchmarks extending the earlier 10-bus pilot.')
    lines.append('- They materially strengthen reviewer-facing three-phase evidence beyond a toy example but do not replace a utility-grade field feeder benchmark.')
    out_path = os.path.join(SAVE_DIR, 'threephase_benchmark_suite_result.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(out_path)

if __name__ == '__main__':
    main()
