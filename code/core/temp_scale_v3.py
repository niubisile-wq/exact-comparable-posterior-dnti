import copy, warnings
import numpy as np
import torch
import torch.nn as nn
import pandapower as pp
import networkx as nx
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIGMA  = 0.009
K_TEST = 12

def build_ieee33():
    net = pp.create_empty_network()
    for i in range(33): pp.create_bus(net, vn_kv=12.66)
    br=[(0,1,0.0922,0.0470),(1,2,0.4930,0.2511),(2,3,0.3660,0.1864),(3,4,0.3811,0.1941),(4,5,0.8190,0.7070),(5,6,0.1872,0.6188),(6,7,0.7114,0.2351),(7,8,1.0300,0.7400),(8,9,1.0440,0.7400),(9,10,0.1966,0.0650),(10,11,0.3744,0.1238),(11,12,1.4680,1.1550),(12,13,0.5416,0.7129),(13,14,0.5910,0.5260),(14,15,0.7463,0.5450),(15,16,1.2890,1.7210),(16,17,0.7320,0.5740),(1,18,0.1640,0.1565),(18,19,1.5042,1.3554),(19,20,0.4095,0.4784),(20,21,0.7089,0.9373),(2,22,0.4512,0.3083),(22,23,0.8980,0.7091),(23,24,0.8960,0.7011),(5,25,0.2030,0.1034),(25,26,0.2842,0.1447),(26,27,1.0590,0.9337),(27,28,0.8042,0.7006),(28,29,0.5075,0.2585),(29,30,0.9744,0.9630),(30,31,0.3105,0.3619),(31,32,0.3410,0.5302)]
    ti=[(7,20,0.089,0.089),(8,14,0.059,0.059),(11,21,0.089,0.089),(17,32,0.038,0.085),(24,28,0.056,0.065)]
    for f,t,r,x in br: pp.create_line_from_parameters(net,f,t,1,r,x,0,1,in_service=True)
    for f,t,r,x in ti: pp.create_line_from_parameters(net,f,t,1,r,x,0,1,in_service=False)
    ld=[(1,100,60),(2,90,40),(3,120,80),(4,60,30),(5,60,20),(6,200,100),(7,200,100),(8,60,20),(9,60,20),(10,45,30),(11,60,35),(12,60,35),(13,120,80),(14,60,10),(15,60,20),(16,60,20),(17,90,40),(18,90,40),(19,90,40),(20,90,40),(21,90,40),(22,90,50),(23,420,200),(24,420,200),(25,60,25),(26,60,25),(27,60,20),(28,120,70),(29,200,600),(30,150,70),(31,210,100),(32,60,40)]
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
        net.line.at[net.line.index[li],"in_service"]=active
    try:
        pp.runpp(net,algorithm="bfsw",numba=False,max_iteration=50,tolerance_mva=1e-8)
        if net.converged: return net.res_bus.vm_pu.values.copy()
    except: pass
    return None

class ResBlock(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(d,d),nn.LayerNorm(d),nn.GELU(),nn.Linear(d,d),nn.LayerNorm(d))
        self.act=nn.GELU()
    def forward(self,x): return self.act(x+self.layers(x))

class FixedKNRE_v3(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(66,512),nn.LayerNorm(512),nn.GELU())
        self.res1=ResBlock(512); self.res2=ResBlock(512); self.res3=ResBlock(512)
        self.down=nn.Sequential(nn.Linear(512,256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,n))
    def forward(self,x):
        h=self.embed(x); h=self.res1(h); h=self.res2(h); h=self.res3(h); return self.down(h)

print("Loading v3 model...")
net33,ne33,te33=build_ieee33()
topos_raw=enum_topos(ne33,te33)
N_TOPOS=len(topos_raw)
voltages=[run_pf(net33,t) for t in topos_raw]
V_all=np.stack(voltages); N_BUS=V_all.shape[1]

MODEL_PATH = r"<LOCAL_WORKSPACE>\nre_ip1_v3.pt"
ckpt=torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model=FixedKNRE_v3(N_TOPOS).to(DEVICE)
model.load_state_dict(ckpt["model_state"]); model.eval()
print("Model loaded OK")

def ais_post(obs_v,obs_n):
    diff=(V_all[:,obs_n]-obs_v)/SIGMA
    ll=-0.5*np.sum(diff**2,axis=1); ll-=ll.max()
    w=np.exp(ll); return w/w.sum()

def make_x(obs,obs_v):
    x=np.zeros(66,dtype=np.float32); x[obs]=obs_v; x[33+obs]=1.0
    return torch.tensor(x,dtype=torch.float32).unsqueeze(0).to(DEVICE)

# 验证集收集 logits（seed=99）
print("Collecting validation logits (N=2000, seed=99)...")
rng_val=np.random.RandomState(99)
val_logits, val_labels = [], []
for _ in range(2000):
    ti=rng_val.randint(0,N_TOPOS)
    obs=np.sort(rng_val.choice(range(1,N_BUS),K_TEST,replace=False))
    obs_v=V_all[ti][obs]+rng_val.normal(0,SIGMA*0.3,K_TEST)
    with torch.no_grad():
        logits=model(make_x(obs,obs_v)).cpu().numpy()[0]
    val_logits.append(logits); val_labels.append(ti)
val_logits=np.array(val_logits); val_labels=np.array(val_labels)

# 搜索最优 T
print("Grid-searching optimal temperature T...")
best_T, best_acc = 1.0, 0.0
T_grid=np.concatenate([np.arange(0.1,1.0,0.02), np.arange(1.0,5.1,0.1)])
for T in T_grid:
    preds=np.argmax(val_logits/T,axis=1)
    acc=np.mean(preds==val_labels)
    if acc>best_acc: best_acc=acc; best_T=round(float(T),2)

baseline_val=np.mean(np.argmax(val_logits,axis=1)==val_labels)
print(f"Val  T=1.00: {baseline_val:.3f}")
print(f"Val  T={best_T:.2f}: {best_acc:.3f}  (delta={best_acc-baseline_val:+.3f})")

# 测试集评估（seed=77，与v3原始评估相同）
print("\nEvaluating on test set (N=1000, seed=77, sigma_obs=0.003)...")
rng_te=np.random.RandomState(77)
acc_ais,acc_raw,acc_cal,kl_raw,kl_cal,ci_raw,ci_cal=[],[],[],[],[],[],[]
for _ in range(1000):
    ti=rng_te.randint(0,N_TOPOS)
    obs=np.sort(rng_te.choice(range(1,N_BUS),K_TEST,replace=False))
    obs_v=V_all[ti][obs]+rng_te.normal(0,SIGMA*0.3,K_TEST)
    p_ais=ais_post(obs_v,obs)
    acc_ais.append(int(np.argmax(p_ais)==ti))
    with torch.no_grad():
        logits=model(make_x(obs,obs_v)).cpu().numpy()[0]
    p_raw=np.exp(logits-logits.max()); p_raw/=p_raw.sum()
    acc_raw.append(int(np.argmax(p_raw)==ti))
    kl_raw.append(np.sum(p_ais*np.log((p_ais+1e-10)/(p_raw+1e-10))))
    si=np.argsort(p_raw)[::-1]; cs=np.cumsum(p_raw[si])
    ci_raw.append(int(ti in si[:np.searchsorted(cs,0.90)+1]))
    lg_c=logits/best_T; p_cal=np.exp(lg_c-lg_c.max()); p_cal/=p_cal.sum()
    acc_cal.append(int(np.argmax(p_cal)==ti))
    kl_cal.append(np.sum(p_ais*np.log((p_ais+1e-10)/(p_cal+1e-10))))
    si2=np.argsort(p_cal)[::-1]; cs2=np.cumsum(p_cal[si2])
    ci_cal.append(int(ti in si2[:np.searchsorted(cs2,0.90)+1]))

print(f"\n{'='*58}")
print(f"Temperature Scaling Results  (v3 base, K=12, sigma_obs=0.003)")
print(f"  Optimal T (from val): {best_T}")
print(f"  {'':22s}  top-1    KL(AIS||NRE)  CI-90")
print(f"  AIS (exact posterior): {np.mean(acc_ais):.3f}   --            --")
print(f"  NRE raw  (T=1.0):      {np.mean(acc_raw):.3f}   {np.mean(kl_raw):.4f}        {np.mean(ci_raw):.3f}")
print(f"  NRE cal  (T={best_T:.2f}):   {np.mean(acc_cal):.3f}   {np.mean(kl_cal):.4f}        {np.mean(ci_cal):.3f}")
print(f"  Delta top-1: {np.mean(acc_cal)-np.mean(acc_raw):+.3f}")
print(f"  Delta KL:    {np.mean(kl_cal)-np.mean(kl_raw):+.4f}")
print(f"{'='*58}")