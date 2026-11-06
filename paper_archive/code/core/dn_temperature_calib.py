# -*- coding: utf-8 -*-
"""温度校准：在验证集上搜索最优T，修复CI过宽问题"""
import copy, warnings
import numpy as np
import torch
import torch.nn as nn
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_IN  = r"<LOCAL_DESKTOP>\nre_ip1_v2.pt"
SAVE_OUT = r"<LOCAL_DESKTOP>\nre_ip1_calibrated.pt"
K_FIXED  = 12
SIGMA    = 0.009
np.random.seed(0); torch.manual_seed(0)

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

class FixedKMaskedNRE(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(66,512),nn.LayerNorm(512),nn.GELU(),
            nn.Linear(512,512),nn.LayerNorm(512),nn.GELU(),
            nn.Linear(512,256),nn.LayerNorm(256),nn.GELU(),
            nn.Linear(256,128),nn.GELU(),nn.Linear(128,n))
    def forward(self,x): return self.net(x)

print("Building network...")
net33,ne33,te33=build_ieee33()
topos_raw=enum_topos(ne33,te33)
N_TOPOS=len(topos_raw)
voltages=[run_pf(net33,t) for t in topos_raw]
V_all=np.stack([v for v in voltages])

print(f"Loading model from {SAVE_IN}...")
ckpt=torch.load(SAVE_IN,map_location=DEVICE,weights_only=False)
model=FixedKMaskedNRE(N_TOPOS).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()

def ais_post(obs_v,obs_n,V_all,sigma):
    diff=(V_all[:,obs_n]-obs_v)/sigma
    ll=-0.5*np.sum(diff**2,axis=1); ll-=ll.max()
    w=np.exp(ll); return w/w.sum()

# ── 收集验证集logits ──────────────────────────────────────────────
print("Collecting validation logits (2000 samples)...")
rng_val=np.random.RandomState(123)
val_logits,val_labels=[],[]
with torch.no_grad():
    for _ in range(2000):
        ti=rng_val.randint(0,N_TOPOS)
        obs=np.sort(rng_val.choice(range(1,33),K_FIXED,replace=False))
        obs_v=V_all[ti][obs]+rng_val.normal(0,SIGMA,K_FIXED)
        x=np.zeros(66,dtype=np.float32)
        x[obs]=obs_v; x[33+obs]=1.0
        xt=torch.tensor(x,dtype=torch.float32).unsqueeze(0).to(DEVICE)
        logits=model(xt).cpu().numpy()[0]
        val_logits.append(logits)
        val_labels.append(ti)
val_logits=np.array(val_logits)
val_labels=np.array(val_labels)

# ── 温度搜索：NLL最小化（等价于ECE最小化）──────────────────────
print("\nSearching optimal temperature T...")
val_L_t=torch.tensor(val_logits,dtype=torch.float32)
val_y_t=torch.tensor(val_labels,dtype=torch.long)

best_T,best_nll=1.0,np.inf
T_candidates=np.linspace(0.1,3.0,200)

print(f"  {'T':>6}  {'NLL':>8}  {'top-1':>7}  {'90%CI':>7}  {'H_mean':>8}")
print(f"  {'-'*45}")

results=[]
rng_te=np.random.RandomState(99)
for T in T_candidates:
    with torch.no_grad():
        probs_t=torch.softmax(val_L_t/T,dim=1)
        nll=nn.CrossEntropyLoss()(val_L_t/T,val_y_t).item()
    probs=probs_t.numpy()
    acc=np.mean(np.argmax(probs,axis=1)==val_labels)
    # 90% CI覆盖率
    cov90=[]
    for i in range(len(val_labels)):
        sorted_idx=np.argsort(probs[i])[::-1]
        cumsum=np.cumsum(probs[i][sorted_idx])
        n_set=np.searchsorted(cumsum,0.90)+1
        cov90.append(int(val_labels[i] in sorted_idx[:n_set]))
    cov=np.mean(cov90)
    H_mean=float(-np.mean(np.sum(probs*np.log(probs+1e-15),axis=1)))
    results.append((T,nll,acc,cov,H_mean))
    if nll<best_nll:
        best_nll=nll; best_T=T

# 打印关键T值
key_Ts=[0.3,0.5,0.7,0.8,0.9,1.0,1.2,1.5,2.0]
for T,nll,acc,cov,H in results:
    if any(abs(T-kt)<0.01 for kt in key_Ts):
        print(f"  {T:>6.2f}  {nll:>8.4f}  {acc:>7.3f}  {cov:>7.3f}  {H:>8.3f}")

print(f"\n  Best T by NLL: {best_T:.3f}  (NLL={best_nll:.4f})")

# 找校准最好的T（90% CI最接近0.90）
calib_results=[(abs(cov-0.90),T,acc,cov) for T,nll,acc,cov,H in results]
calib_results.sort()
best_calib_T=calib_results[0][1]
best_calib_acc=calib_results[0][2]
best_calib_cov=calib_results[0][3]
print(f"  Best T by calibration: {best_calib_T:.3f}"
      f"  (90%CI={best_calib_cov:.3f}, acc={best_calib_acc:.3f})")

# 最终选T：在校准好的T中选acc最高的
good_calib=[(acc,T,cov) for diff,T,acc,cov in calib_results if diff<0.05]
if good_calib:
    good_calib.sort(reverse=True)
    final_T=good_calib[0][1]
    final_acc=good_calib[0][0]
    final_cov=good_calib[0][2]
    print(f"  Final T (best acc with good calib): {final_T:.3f}"
          f"  acc={final_acc:.3f}  90%CI={final_cov:.3f}")
else:
    final_T=best_calib_T
    final_acc=best_calib_acc
    final_cov=best_calib_cov

# ── 测试集最终评估 ────────────────────────────────────────────────
print(f"\nFinal evaluation on test set (T={final_T:.3f})...")
rng_te=np.random.RandomState(77)
acc1_ais,acc1_nre_raw,acc1_nre_cal,cov90_raw,cov90_cal,kl_cal=[],[],[],[],[],[]

for _ in range(1000):
    ti=rng_te.randint(0,N_TOPOS)
    obs=np.sort(rng_te.choice(range(1,33),K_FIXED,replace=False))
    obs_v=V_all[ti][obs]+rng_te.normal(0,SIGMA*0.3,K_FIXED)
    p_ais=ais_post(obs_v,obs,V_all,SIGMA)
    acc1_ais.append(int(np.argmax(p_ais)==ti))
    x=np.zeros(66,dtype=np.float32); x[obs]=obs_v; x[33+obs]=1.0
    with torch.no_grad():
        logits=model(torch.tensor(x,dtype=torch.float32).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    p_raw=np.exp(logits-logits.max()); p_raw/=p_raw.sum()
    p_cal=np.exp(logits/final_T-np.max(logits/final_T)); p_cal/=p_cal.sum()
    acc1_nre_raw.append(int(np.argmax(p_raw)==ti))
    acc1_nre_cal.append(int(np.argmax(p_cal)==ti))
    for p,lst in [(p_raw,cov90_raw),(p_cal,cov90_cal)]:
        si=np.argsort(p)[::-1]; cs=np.cumsum(p[si])
        lst.append(int(ti in si[:np.searchsorted(cs,0.90)+1]))
    kl_cal.append(np.sum(p_ais*np.log((p_ais+1e-10)/(p_cal+1e-10))))

print(f"\n  {'方法':>22}  {'top-1':>7}  {'90%CI':>7}  {'KL':>8}")
print(f"  {'-'*50}")
print(f"  {'AIS':>22}  {np.mean(acc1_ais):>7.3f}  {'---':>7}  {'---':>8}")
print(f"  {'NRE (T=1.0, raw)':>22}  {np.mean(acc1_nre_raw):>7.3f}  "
      f"{np.mean(cov90_raw):>7.3f}  {'---':>8}")
print(f"  {f'NRE (T={final_T:.2f}, calib)':>22}  {np.mean(acc1_nre_cal):>7.3f}  "
      f"{np.mean(cov90_cal):>7.3f}  {np.mean(kl_cal):>8.4f}")

gap_raw=np.mean(acc1_ais)-np.mean(acc1_nre_raw)
gap_cal=np.mean(acc1_ais)-np.mean(acc1_nre_cal)
print(f"\n  精度差距: raw={gap_raw:+.3f}  calibrated={gap_cal:+.3f}")
print(f"  校准改善: CI从{np.mean(cov90_raw):.3f}→{np.mean(cov90_cal):.3f}"
      f" (目标0.900)")

torch.save({'model_state':model.state_dict(),'N_TOPOS':N_TOPOS,
            'K_FIXED':K_FIXED,'SIGMA':SIGMA,'temperature':final_T,
            'voltages':V_all,'topos_raw':topos_raw}, SAVE_OUT)
print(f"\n  Calibrated model saved: {SAVE_OUT}")
print(f"\n{'='*55}")
print(f"TEMPERATURE CALIBRATION COMPLETE")
print(f"  Optimal T = {final_T:.3f}")
print(f"  top-1 accuracy: {np.mean(acc1_nre_cal):.3f}  (gap={gap_cal:+.3f})")
print(f"  90% CI coverage: {np.mean(cov90_cal):.3f}  (target=0.900)")
print(f"  KL(AIS||NRE_cal): {np.mean(kl_cal):.4f}")
print(f"{'='*55}")
