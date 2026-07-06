# -*- coding: utf-8 -*-
"""继续训练NRE模型：从已保存的checkpoint继续训15000步"""
import copy, time, warnings
import numpy as np
import torch
import torch.nn as nn
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH  = r"<LOCAL_DESKTOP>\nre_model.pt"
EXTRA_STEPS = 15000
BATCH_SIZE  = 512
LR_RESUME   = 5e-5   # 降低lr继续微调
LOG_EVERY   = 1000
np.random.seed(7); torch.manual_seed(7)

# ── 重建网络（与训练脚本一致）────────────────────────────────────────
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
    ne=[(int(f),int(t)) for f,t,r,x in br]
    te=[(int(f),int(t)) for f,t,r,x in ti]
    return net,ne,te

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

class MaskedNRE(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(66,512),nn.LayerNorm(512),nn.GELU(),
            nn.Linear(512,512),nn.LayerNorm(512),nn.GELU(),
            nn.Linear(512,256),nn.LayerNorm(256),nn.GELU(),
            nn.Linear(256,128),nn.GELU(),nn.Linear(128,n))
    def forward(self,x): return self.net(x)
    def posterior(self,x_np):
        self.eval()
        with torch.no_grad():
            x=torch.tensor(x_np,dtype=torch.float32).to(DEVICE)
            if x.ndim==1: x=x.unsqueeze(0)
            return torch.softmax(self(x),dim=1).cpu().numpy()[0]

def make_batch(rng,bs,V_all,N,sigma,kmin,kmax,miss=0.0):
    X=np.zeros((bs,66),dtype=np.float32); y=np.zeros(bs,dtype=np.int64)
    for i in range(bs):
        ti=rng.randint(0,N); V=V_all[ti]
        K=rng.randint(kmin,kmax+1)
        obs=rng.choice(range(1,33),K,replace=False)
        if miss>0:
            keep=rng.rand(K)>miss; obs=obs[keep]
            if len(obs)==0: obs=rng.choice(range(1,33),1)
        X[i,obs]=V[obs]+rng.normal(0,sigma,len(obs))
        X[i,33+obs]=1.0; y[i]=ti
    return X,y

def ais_post(obs_v,obs_n,V_all,sigma):
    diff=(V_all[:,obs_n]-obs_v)/sigma
    ll=-0.5*np.sum(diff**2,axis=1); ll-=ll.max()
    w=np.exp(ll); return w/w.sum()

# ── 加载 ─────────────────────────────────────────────────────────────
print(f"Device: {DEVICE}")
print("Building network...")
net33,ne33,te33=build_ieee33()
topos_raw=enum_topos(ne33,te33)
N_TOPOS=len(topos_raw)
voltages=[run_pf(net33,t) for t in topos_raw]
V_all=np.stack([v for v in voltages])
SIGMA=0.009; K_MIN,K_MAX=5,28

print(f"Loading checkpoint from {SAVE_PATH}...")
ckpt=torch.load(SAVE_PATH,map_location=DEVICE,weights_only=False)
model=MaskedNRE(N_TOPOS).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
print(f"  Loaded. N_TOPOS={N_TOPOS}")

# ── 继续训练 ─────────────────────────────────────────────────────────
optimizer=torch.optim.AdamW(model.parameters(),lr=LR_RESUME,weight_decay=1e-4)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EXTRA_STEPS)
criterion=nn.CrossEntropyLoss()
rng=np.random.RandomState(42)

print(f"\nContinuing training ({EXTRA_STEPS} more steps, lr={LR_RESUME})...")
t_start=time.perf_counter()
for step in range(1,EXTRA_STEPS+1):
    miss=0.2 if rng.rand()<0.2 else 0.0
    X_np,y_np=make_batch(rng,BATCH_SIZE,V_all,N_TOPOS,SIGMA,K_MIN,K_MAX,miss)
    X_t=torch.tensor(X_np,dtype=torch.float32).to(DEVICE)
    y_t=torch.tensor(y_np,dtype=torch.long).to(DEVICE)
    model.train(); optimizer.zero_grad()
    loss=criterion(model(X_t),y_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    optimizer.step(); scheduler.step()
    if step%LOG_EVERY==0 or step==EXTRA_STEPS:
        elapsed=time.perf_counter()-t_start
        eta=elapsed/step*(EXTRA_STEPS-step)
        print(f"  Step {step:>6}/{EXTRA_STEPS}  loss={loss.item():.4f}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

# ── 评估 ─────────────────────────────────────────────────────────────
print("\nEvaluating...")
model.eval()
rng_te=np.random.RandomState(99)
print(f"\n  {'K':>4}  {'NRE_acc':>8}  {'AIS_acc':>8}  {'KL':>8}  {'gap':>6}")
print(f"  {'-'*40}")
gaps=[]
for K in [5,8,12,15,20,25]:
    acc_nre,acc_ais,kls=[],[],[]
    for _ in range(300):
        ti=rng_te.randint(0,N_TOPOS)
        obs=np.sort(rng_te.choice(range(1,33),K,replace=False))
        obs_v=V_all[ti][obs]+rng_te.normal(0,SIGMA*0.3,K)
        p_ais=ais_post(obs_v,obs,V_all,SIGMA)
        acc_ais.append(int(np.argmax(p_ais)==ti))
        x=np.zeros(66,dtype=np.float32); x[obs]=obs_v; x[33+obs]=1.0
        p_nre=model.posterior(x)
        acc_nre.append(int(np.argmax(p_nre)==ti))
        kls.append(np.sum(p_ais*np.log((p_ais+1e-10)/(p_nre+1e-10))))
    gap=np.mean(acc_ais)-np.mean(acc_nre); gaps.append(gap)
    print(f"  {K:>4}  {np.mean(acc_nre):>8.3f}  {np.mean(acc_ais):>8.3f}  "
          f"{np.mean(kls):>8.4f}  {gap:>+6.3f}")

print(f"\n  Missing data (K=15):")
print(f"  {'miss%':>6}  {'naive':>8}  {'robust':>8}")
for mr in [0.0,0.10,0.20,0.30]:
    an,ar=[],[]
    for _ in range(300):
        ti=rng_te.randint(0,N_TOPOS)
        obs=np.sort(rng_te.choice(range(1,33),15,replace=False))
        obs_v=V_all[ti][obs]+rng_te.normal(0,SIGMA*0.3,15)
        keep=rng_te.rand(15)>mr
        if not np.any(keep): keep[0]=True
        oa=obs[keep]; ov=obs_v[keep]
        # naive: 掩码全1但缺失值填0
        xn=np.zeros(66,dtype=np.float32)
        xn[obs]=obs_v*keep; xn[33+obs]=1.0
        # robust: 正确掩码
        xr=np.zeros(66,dtype=np.float32)
        xr[oa]=ov; xr[33+oa]=1.0
        an.append(int(np.argmax(model.posterior(xn))==ti))
        ar.append(int(np.argmax(model.posterior(xr))==ti))
    print(f"  {mr*100:>5.0f}%  {np.mean(an):>8.3f}  {np.mean(ar):>8.3f}")

# 保存
torch.save({'model_state':model.state_dict(),'N_TOPOS':N_TOPOS,
            'SIGMA':SIGMA,'voltages':V_all,'topos_raw':topos_raw},SAVE_PATH)

mean_gap=np.mean(gaps)
total=time.perf_counter()-t_start
print(f"""
{'='*50}
CONTINUED TRAINING COMPLETE
  Time: {total:.0f}s ({total/60:.1f} min)
  Mean accuracy gap: {mean_gap:.3f}
  {'EXCELLENT (<0.05)' if mean_gap<0.05 else 'OK (<0.08)' if mean_gap<0.08 else 'still large'}
  Model saved: {SAVE_PATH}
{'='*50}
""")
