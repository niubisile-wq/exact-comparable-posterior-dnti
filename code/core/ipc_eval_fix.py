import copy, warnings
import numpy as np
import torch
import torch.nn as nn
import pandapower as pp
import networkx as nx
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_FIXED = 20; SIGMA = 0.009; LF_MIN, LF_MAX = 0.8, 1.2; N_LF = 101; N_BUS = 33
MISSING_RATES = [0.0, 0.05, 0.10, 0.20, 0.30]

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

def run_pf_scaled(net_base,t_raw,lf=1.0):
    net=copy.deepcopy(net_base)
    net.load["p_mw"]=net.load["p_mw"]*lf; net.load["q_mvar"]=net.load["q_mvar"]*lf
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
        self.net=nn.Sequential(nn.Linear(d,d),nn.LayerNorm(d),nn.GELU(),nn.Linear(d,d),nn.LayerNorm(d))
        self.act=nn.GELU()
    def forward(self,x): return self.act(x+self.net(x))

class LoadAwareNRE(nn.Module):
    def __init__(self,n_topo,n_bus=33):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(n_bus*3,512),nn.LayerNorm(512),nn.GELU())
        self.res1=ResBlock(512); self.res2=ResBlock(512); self.res3=ResBlock(512)
        self.head=nn.Sequential(nn.Linear(512,256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,n_topo))
    def forward(self,x):
        h=self.embed(x); h=self.res1(h); h=self.res2(h); h=self.res3(h); return self.head(h)

net33,ne33,te33=build_ieee33()
topos_raw=enum_topos(ne33,te33)
N_TOPOS=len(topos_raw)
base_P_per_bus=np.zeros(N_BUS)
for _,row in net33.load.iterrows(): base_P_per_bus[int(row.bus)]+=row["p_mw"]
base_P_norm=base_P_per_bus/(base_P_per_bus.max()+1e-8)
lf_grid=np.linspace(LF_MIN,LF_MAX,N_LF)

print("Precomputing library...")
V_library=np.zeros((N_TOPOS,N_LF,N_BUS),dtype=np.float32)
for i,topo in enumerate(topos_raw):
    for j,lf in enumerate(lf_grid):
        V=run_pf_scaled(net33,topo,lf)
        V_library[i,j,:]=V if V is not None else (V_library[i,max(j-1,0),:] if j>0 else 1.0)
print("Done.")

rob_ckpt=torch.load(r"<LOCAL_WORKSPACE>\nre_ipc_loadaware.pt",map_location=DEVICE,weights_only=False)
nai_ckpt=torch.load(r"<LOCAL_WORKSPACE>\nre_ip1_v5a.pt",map_location=DEVICE,weights_only=False)
rob_model=LoadAwareNRE(N_TOPOS,N_BUS).to(DEVICE); rob_model.load_state_dict(rob_ckpt["model_state"]); rob_model.eval()
nai_model=LoadAwareNRE(N_TOPOS,N_BUS).to(DEVICE); nai_model.load_state_dict(nai_ckpt["model_state"]); nai_model.eval()

def infer(model,reported,obs_v,lf):
    x=np.zeros(N_BUS*3,dtype=np.float32)
    if len(reported)>0:
        x[reported]=obs_v; x[N_BUS+reported]=1.0; x[2*N_BUS+reported]=base_P_norm[reported]*lf
    with torch.no_grad():
        logits=model(torch.tensor(x).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    p=np.exp(logits-logits.max()); p/=p.sum(); return p

def ais_post(reported,obs_v,lf_idx):
    if len(reported)==0: return np.ones(N_TOPOS)/N_TOPOS
    diff=(V_library[:,lf_idx,:][:,reported]-obs_v)/SIGMA
    ll=-0.5*np.sum(diff**2,axis=1); ll-=ll.max(); w=np.exp(ll); return w/w.sum()

print(f"\nFIXED evaluation: both models receive only REPORTED nodes")
print(f"{'Miss%':>6}  {'AIS':>7}  {'Robust':>8}  {'Naive':>8}  {'Delta(R-N)':>12}")
print("-"*50)
for miss_rate in MISSING_RATES:
    n_miss=int(K_FIXED*miss_rate)
    acc_ais,acc_rob,acc_nai=[],[],[]
    rng=np.random.RandomState(77)
    for _ in range(1000):
        ti=rng.randint(0,N_TOPOS); lf_idx=rng.randint(0,N_LF); lf=lf_grid[lf_idx]
        V=V_library[ti,lf_idx,:]
        installed=np.sort(rng.choice(range(1,N_BUS),K_FIXED,replace=False))
        if n_miss>0:
            miss_idx=rng.choice(len(installed),n_miss,replace=False)
            reported=np.delete(installed,miss_idx)
        else:
            reported=installed
        obs_v=V[reported]+rng.normal(0,SIGMA,len(reported))
        p_ais=ais_post(reported,obs_v,lf_idx)
        acc_ais.append(int(np.argmax(p_ais)==ti))
        p_rob=infer(rob_model,reported,obs_v,lf)
        acc_rob.append(int(np.argmax(p_rob)==ti))
        p_nai=infer(nai_model,reported,obs_v,lf)  # 同样只传reported
        acc_nai.append(int(np.argmax(p_nai)==ti))
    a,r,n=np.mean(acc_ais),np.mean(acc_rob),np.mean(acc_nai)
    print(f"  {miss_rate*100:>4.0f}%  {a:>7.3f}  {r:>8.3f}  {n:>8.3f}  {r-n:>+12.3f}")
print()
print("Gate: miss=10% and miss=30% Delta(R-N) > 10pp?")