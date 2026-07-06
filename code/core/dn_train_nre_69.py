# -*- coding: utf-8 -*-
"""
IEEE 69-bus NRE训练（可变K掩码模型）
支持：IP1精度对比 + IP-C通信缺失鲁棒推断
"""
import copy, time, warnings
import numpy as np
import torch
import torch.nn as nn
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH = r"<LOCAL_WORKSPACE>\nre_model_69.pt"
OUT_PATH  = r"<LOCAL_WORKSPACE>\nre_69_train_out.txt"
N_STEPS   = 25000
BATCH     = 512
LR        = 3e-4
SIGMA     = 0.009
K_MIN, K_MAX = 5, 55   # 69-bus有68个非slack节点，取5-55
LOG_EVERY = 2500
print(f"Device: {DEVICE}")

def build_ieee69():
    net = pp.create_empty_network()
    for i in range(69): pp.create_bus(net, vn_kv=12.66)
    branch_data = [
        (1,2,0.0005,0.0012,0,0),(2,3,0.0005,0.0012,0,0),(3,4,0.0015,0.0036,0,0),(4,5,0.0251,0.0294,0,0),
        (5,6,0.3660,0.1864,2.6,2.2),(6,7,0.3811,0.1941,40.4,30.0),(7,8,0.0922,0.0470,75.0,54.0),
        (8,9,0.0493,0.0251,30.0,22.0),(9,10,0.8190,0.2707,28.0,19.0),(10,11,0.1872,0.0619,145.0,104.0),
        (11,12,0.7114,0.2351,145.0,104.0),(12,13,1.0300,0.3400,8.0,5.5),(13,14,1.0440,0.3450,8.0,5.5),
        (14,15,1.0580,0.3496,0.0,0.0),(15,16,0.1966,0.0650,45.5,30.0),(16,17,0.3744,0.1238,60.0,35.0),
        (17,18,0.0047,0.0016,60.0,35.0),(18,19,0.3276,0.1083,0.0,0.0),(19,20,0.2106,0.0690,1.0,0.6),
        (20,21,0.3416,0.1129,114.0,81.0),(21,22,0.0140,0.0046,5.3,3.5),(22,23,0.1591,0.0526,0.0,0.0),
        (23,24,0.3463,0.1145,28.0,20.0),(24,25,0.7488,0.2475,0.0,0.0),(25,26,0.3089,0.1021,14.0,10.0),
        (26,27,0.1732,0.0572,14.0,10.0),(3,28,0.0044,0.0108,26.0,18.6),(28,29,0.0640,0.1565,26.0,18.6),
        (29,30,0.3978,0.1315,0.0,0.0),(30,31,0.0702,0.0232,0.0,0.0),(31,32,0.3510,0.1160,0.0,0.0),
        (32,33,0.8390,0.2816,14.0,10.0),(33,34,1.7080,0.5646,19.5,14.0),(34,35,1.4740,0.4873,6.0,4.0),
        (35,36,0.0044,0.0108,26.0,18.6),(36,37,0.0640,0.1565,26.0,18.6),(37,38,0.1053,0.1230,0.0,0.0),
        (38,39,0.0304,0.0355,24.0,17.0),(39,40,0.0018,0.0021,24.0,17.0),(40,41,0.7283,0.8509,1.2,1.0),
        (41,42,0.3100,0.3623,0.0,0.0),(42,43,0.0410,0.0478,6.0,4.3),(43,44,0.0092,0.0116,0.0,0.0),
        (44,45,0.1089,0.1373,39.2,26.3),(45,46,0.0009,0.0012,39.2,26.3),(4,47,0.0034,0.0084,0.0,0.0),
        (47,48,0.0851,0.2083,79.0,56.4),(48,49,0.2898,0.7091,384.7,274.5),(49,50,0.0822,0.2011,384.7,274.5),
        (8,51,0.0928,0.0473,40.5,28.3),(51,52,0.3319,0.1114,3.6,2.7),(9,53,0.1740,0.0886,4.35,3.5),
        (53,54,0.2030,0.1034,26.4,19.0),(54,55,0.2842,0.1447,24.0,17.2),(55,56,0.2813,0.1433,0.0,0.0),
        (56,57,1.5900,0.5337,0.0,0.0),(57,58,0.7837,0.2630,0.0,0.0),(58,59,0.3042,0.1006,100.0,72.0),
        (59,60,0.3861,0.1172,0.0,0.0),(60,61,0.5075,0.2585,1244.0,888.0),(61,62,0.0974,0.0496,32.0,23.0),
        (62,63,0.1450,0.0738,0.0,0.0),(63,64,0.7105,0.3619,227.0,162.0),(64,65,1.0410,0.5302,59.0,42.0),
        (11,66,0.2012,0.0611,18.0,13.0),(66,67,0.0047,0.0014,18.0,13.0),
        (12,68,0.7394,0.2444,28.0,20.0),(68,69,0.0047,0.0014,28.0,20.0),
    ]
    tie_data = [(11,43,0.5,0.5),(13,21,0.5,0.5),(15,46,0.5,0.5),(50,59,0.5,0.5),(27,65,0.5,0.5)]
    ne, te = [], []
    for f1,t1,r,x,p,q in branch_data:
        f0,t0=f1-1,t1-1
        pp.create_line_from_parameters(net,f0,t0,1,r,x,0,9999,in_service=True)
        if p>0 or q>0: pp.create_load(net,t0,p/1000,q/1000)
        ne.append((f0,t0))
    for f1,t1,r,x in tie_data:
        f0,t0=f1-1,t1-1
        pp.create_line_from_parameters(net,f0,t0,1,r,x,0,9999,in_service=False)
        te.append((f0,t0))
    pp.create_ext_grid(net,0,vm_pu=1.0)
    return net, ne, te

def enum_topos(ne, te, n=69):
    G=nx.Graph(); G.add_edges_from(ne)
    topos=[list(range(len(ne)))]; seen={frozenset(range(len(ne)))}
    for ti2,tie in enumerate(te):
        try: path=nx.shortest_path(G,tie[0],tie[1])
        except: continue
        for i in range(len(path)-1):
            oe=frozenset([path[i],path[i+1]])
            ni=[j for j,e in enumerate(ne) if frozenset(e)!=oe]
            key=frozenset(ni)
            if key in seen: continue
            edges=[ne[j] for j in ni]+[tie]
            Gt=nx.Graph(); Gt.add_nodes_from(range(n)); Gt.add_edges_from(edges)
            if nx.is_connected(Gt) and nx.is_tree(Gt):
                seen.add(key); topos.append(ni+[len(ne)+ti2])
    return topos

def run_pf(net_base, topo_raw, ne, te):
    net=copy.deepcopy(net_base)
    n_ne=len(ne)
    act_ne={x for x in topo_raw if x<n_ne}
    act_te={x-n_ne for x in topo_raw if x>=n_ne}
    for li in range(n_ne):
        net.line.at[net.line.index[li],'in_service']=(li in act_ne)
    for li in range(len(te)):
        net.line.at[net.line.index[n_ne+li],'in_service']=(li in act_te)
    try:
        pp.runpp(net,algorithm='bfsw',numba=False,max_iteration=100,tolerance_mva=1e-6)
        if net.converged: return net.res_bus.vm_pu.values.copy()
    except: pass
    return None

# 138维输入：69节点电压 + 69位掩码
class MaskedNRE69(nn.Module):
    def __init__(self, n_topos):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(138, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, n_topos))
    def forward(self, x): return self.net(x)
    def posterior(self, x_np):
        self.eval()
        with torch.no_grad():
            x=torch.tensor(x_np,dtype=torch.float32).to(DEVICE)
            if x.ndim==1: x=x.unsqueeze(0)
            return torch.softmax(self(x),dim=1).cpu().numpy()[0]

print("Building IEEE 69-bus...")
net69,ne69,te69=build_ieee69()
topos=enum_topos(ne69,te69)
N_TOPOS=len(topos)
print(f"Topologies: {N_TOPOS}")

print("Computing voltage profiles...")
t0=time.time()
voltages=[run_pf(net69,t,ne69,te69) for t in topos]
ok=[v for v in voltages if v is not None]
V_all=np.stack(ok)
N_TOPOS=len(V_all)
print(f"Converged: {N_TOPOS}/{len(topos)}  ({time.time()-t0:.1f}s)")

model=MaskedNRE69(N_TOPOS).to(DEVICE)
n_params=sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR*5,total_steps=N_STEPS,pct_start=0.05)
loss_fn=nn.CrossEntropyLoss()
rng=np.random.RandomState(42)

def make_batch(rng, bs):
    X=np.zeros((bs,138),dtype=np.float32); y=np.zeros(bs,dtype=np.int64)
    for i in range(bs):
        ti=rng.randint(0,N_TOPOS)
        K=rng.randint(K_MIN,K_MAX+1)
        obs=rng.choice(range(1,69),K,replace=False)
        miss=0.2 if rng.rand()<0.2 else 0.0
        if miss>0:
            keep=rng.rand(K)>miss
            if not keep.any(): keep[0]=True
            obs=obs[keep]
        X[i,obs]=V_all[ti][obs]+rng.normal(0,SIGMA,len(obs))
        X[i,69+obs]=1.0; y[i]=ti
    return X,y

print(f"\nTraining: {N_STEPS} steps, batch={BATCH}...")
t_start=time.time()
log_lines=[]
model.train()
for step in range(1,N_STEPS+1):
    X_np,y_np=make_batch(rng,BATCH)
    X_t=torch.tensor(X_np,dtype=torch.float32).to(DEVICE)
    y_t=torch.tensor(y_np,dtype=torch.long).to(DEVICE)
    opt.zero_grad(); loss=loss_fn(model(X_t),y_t); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt.step(); scheduler.step()
    if step%LOG_EVERY==0 or step==N_STEPS:
        e=time.time()-t_start
        msg=f"  Step {step:>6}/{N_STEPS}  loss={loss.item():.4f}  elapsed={e:.0f}s"
        print(msg); log_lines.append(msg)

# 评估
model.eval()
rng_te=np.random.RandomState(99)
results={}
for K in [8,12,20]:
    accs,accs_ais=[],[]
    for _ in range(300):
        ti=rng_te.randint(0,N_TOPOS)
        obs=np.sort(rng_te.choice(range(1,69),K,replace=False))
        obs_v=V_all[ti][obs]+rng_te.normal(0,SIGMA*0.3,K)
        diff=(V_all[:,obs]-obs_v)/SIGMA; ll=-0.5*np.sum(diff**2,axis=1)
        ll-=ll.max(); w=np.exp(ll); p_ais=w/w.sum()
        accs_ais.append(int(np.argmax(p_ais)==ti))
        x=np.zeros(138,dtype=np.float32); x[obs]=obs_v; x[69+obs]=1.0
        p_nre=model.posterior(x)
        accs.append(int(np.argmax(p_nre)==ti))
    results[K]=(np.mean(accs),np.mean(accs_ais))

# IP-C测试
miss_res={}
for mr in [0.0,0.10,0.20,0.30]:
    r,n=[],[]
    for _ in range(300):
        ti=rng_te.randint(0,N_TOPOS); K=20
        obs=np.sort(rng_te.choice(range(1,69),K,replace=False))
        obs_v=V_all[ti][obs]+rng_te.normal(0,SIGMA*0.3,K)
        keep=rng_te.rand(K)>mr if mr>0 else np.ones(K,dtype=bool)
        if not keep.any(): keep[0]=True
        av=obs[keep]; vv=obs_v[keep]
        xr=np.zeros(138,dtype=np.float32); xr[av]=vv; xr[69+av]=1.0
        xn=np.zeros(138,dtype=np.float32)
        xn[obs]=obs_v*keep; xn[69+obs]=1.0
        r.append(int(np.argmax(model.posterior(xr))==ti))
        n.append(int(np.argmax(model.posterior(xn))==ti))
    miss_res[mr]=(np.mean(r),np.mean(n))

# AIS speed
t0=time.time()
x_t=torch.zeros(1,138,dtype=torch.float32).to(DEVICE)
with torch.no_grad():
    for _ in range(10000): model(x_t)
nre_ms=(time.time()-t0)/10000*1000

output_lines=[]
output_lines.append("="*60)
output_lines.append("IEEE 69-bus NRE Training Complete")
output_lines.append("="*60)
output_lines.append(f"Topologies: {N_TOPOS}, Model params: {n_params:,}")
output_lines.append(f"NRE inference: {nre_ms:.3f}ms/query")
output_lines.append("")
output_lines.append("IP1 Accuracy (NRE vs AIS):")
for K,(nre,ais) in results.items():
    output_lines.append(f"  K={K:2d}: NRE={nre:.3f}  AIS={ais:.3f}  gap={ais-nre:+.3f}")
output_lines.append("")
output_lines.append("IP-C Missing Data (K=20):")
for mr,(rb,nv) in miss_res.items():
    output_lines.append(f"  miss={mr*100:.0f}%: Robust={rb:.3f}  Naive={nv:.3f}  Delta={rb-nv:+.3f}")
output_lines.append("="*60)

out="\n".join(output_lines)
print(out)
with open(OUT_PATH,'w',encoding='utf-8') as f:
    f.write(out)
torch.save({'model_state':model.state_dict(),'N_TOPOS':N_TOPOS,
            'voltages':V_all,'topos_raw':topos,'SIGMA':SIGMA},SAVE_PATH)
print(f"Saved: {SAVE_PATH}")
