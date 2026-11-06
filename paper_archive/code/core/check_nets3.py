import pandapower as pp
import pandapower.networks as pn
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

for name, builder in [
    ("simple_mv_open_ring_net", lambda: pn.simple_mv_open_ring_net()),
    ("mv_oberrhein",            lambda: pn.mv_oberrhein()),
]:
    try:
        net = builder()
        open_l = net.line[~net.line.in_service]
        active_l = net.line[net.line.in_service]
        print(f"\n{name}: buses={len(net.bus)}, active_lines={len(active_l)}, open_lines(tie)={len(open_l)}")
        if len(open_l) > 0:
            print(f"  Tie switches: {list(open_l[['from_bus','to_bus']].itertuples(index=False,name=None))}")
        # 跑潮流
        try:
            pp.runpp(net, algorithm="bfsw", numba=False, max_iteration=100)
            print(f"  Converged: {net.converged}")
        except Exception as e:
            print(f"  PF error: {e}")
    except Exception as e:
        print(f"{name}: ERROR - {e}")
