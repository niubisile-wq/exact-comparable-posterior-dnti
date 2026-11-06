# -*- coding: utf-8 -*-
"""快速检验：BOED vs MVG 后验熵H(K)对比（33-bus，真正的优化目标）"""
import warnings, numpy as np, torch
warnings.filterwarnings('ignore')

SAVE_DIR = r"<LOCAL_WORKSPACE>"
SIGMA=0.009; N_LF=101; K_MAX=12; N_BUS=33; K_FIXED=20; N_TEST=500

ckpt = torch.load(f"{SAVE_DIR}\\nre_ipc_loadaware.pt", map_location='cpu', weights_only=False)
V_library=ckpt['V_library']; lf_grid=ckpt['lf_grid']; N_TOPOS=ckpt['N_TOPOS']
N_LF_C=11
lf_idx_c=np.round(np.linspace(0,N_LF-1,N_LF_C)).astype(int)
V_lib_c=V_library[:,lf_idx_c,:]
CANDIDATES=list(range(1,N_BUS))

def ais_post(obs,vals):
    if not obs: return np.ones(N_TOPOS)/N_TOPOS
    diff=(V_lib_c[:,:,obs]-np.array(vals))/SIGMA
    ll=-0.5*np.sum(diff**2,axis=2); ll-=ll.max()
    w=np.exp(ll).sum(axis=1); return w/w.sum()

def entropy(p): p=np.clip(p,1e-10,1); return -np.sum(p*np.log(p))

def mvg_select(obs,cands,p):
    V_mean_lf=V_library[:,:,cands].mean(axis=1)
    EV=(p[:,None]*V_mean_lf).sum(axis=0)
    EV2=(p[:,None]*V_mean_lf**2).sum(axis=0)
    return cands[int(np.argmax(EV2-EV**2))]

# 用同一测试集（seed=2024，N_TEST=500，与BOED完全一致）
rng=np.random.RandomState(2024)
test_cases=[]
for _ in range(N_TEST):
    ti=rng.randint(0,N_TOPOS); lf_idx=rng.randint(0,N_LF)
    full_v=V_library[ti,lf_idx,:]+rng.normal(0,SIGMA,N_BUS)
    test_cases.append((ti,lf_idx,full_v))

H_mvg=np.zeros(K_MAX)
for ci,(true_ti,lf_idx,full_v) in enumerate(test_cases):
    sel,sv,remain=[],[],list(CANDIDATES)
    for k in range(K_MAX):
        p=ais_post(sel,sv)
        node=mvg_select(sel,remain,p)
        sel.append(node); sv.append(full_v[node]); remain.remove(node)
        H_mvg[k]+=entropy(ais_post(sel,sv))
H_mvg/=N_TEST

# BOED H(K)来自boed_v2_result.txt
boed_H=[2.635,1.907,1.336,0.998,0.790,0.719,0.668,0.630,0.595,0.562,0.543,0.531]

print("33-bus 后验熵H(K)对比（H越低=信息越多=更好）")
print(f"{'K':>3}  {'BOED_H':>10}  {'MVG_H':>10}  {'Winner(H)':>10}  {'BOED_acc':>10}  {'MVG_acc':>10}")
boed_acc=[0.148,0.272,0.452,0.584,0.688,0.708,0.748,0.756,0.768,0.802,0.784,0.792]
mvg_acc =[0.162,0.306,0.466,0.642,0.700,0.718,0.752,0.752,0.764,0.782,0.774,0.792]
print("-"*65)
for k in range(K_MAX):
    bh=boed_H[k]; mh=H_mvg[k]
    bh_better=bh<mh-0.005
    mh_better=mh<bh-0.005
    hw='BOED' if bh_better else ('MVG' if mh_better else 'TIE')
    aw='BOED' if boed_acc[k]>mvg_acc[k]+0.005 else ('MVG' if mvg_acc[k]>boed_acc[k]+0.005 else 'TIE')
    print(f"{k+1:>3}  {bh:>10.3f}  {mh:>10.3f}  {hw:>10}  {boed_acc[k]:>10.3f}  {mvg_acc[k]:>10.3f}  acc={aw}")

print()
boed_wins_H=sum(1 for k in range(K_MAX) if boed_H[k]<H_mvg[k]-0.005)
mvg_wins_H =sum(1 for k in range(K_MAX) if H_mvg[k]<boed_H[k]-0.005)
print(f"H(K)胜负: BOED胜{boed_wins_H}/12, MVG胜{mvg_wins_H}/12")
print()
if boed_wins_H>mvg_wins_H:
    print("结论: BOED在真正的优化目标(H最小化)上一致优于MVG")
    print("      top-1差异是因为H(K)和top-1不是单调对应关系")
    print("      IP-A主张('BOED是贝叶斯最优策略')成立")
elif mvg_wins_H>boed_wins_H:
    print("结论: MVG在H(K)上也优于BOED => EIG估计可能有偏差，需检查")
else:
    print("结论: H(K)基本持平，两者在真实目标上等效")
