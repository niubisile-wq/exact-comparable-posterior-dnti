import pandapower as pp
import pandapower.networks as pn
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

print("=== 检查可用网络 ===")
# 列出pandapower里有tie switch的配电网
networks_to_try = [
    ("case33bw", lambda: pn.case33bw()),
    ("cigre_mv", lambda: pn.create_cigre_network_mv()),
    ("case69",   lambda: pn.case69()),
]

for name, builder in networks_to_try:
    try:
        net = builder()
        n_bus = len(net.bus)
        n_line = len(net.line)
        n_tie = len(net.line[~net.line.in_service])
        n_active = len(net.line[net.line.in_service])
        print(f"\n{name}: buses={n_bus}, active_lines={n_active}, tie_switches={n_tie}")
        
        # 尝试跑潮流
        try:
            pp.runpp(net, algorithm="bfsw", numba=False, max_iteration=50)
            print(f"  Power flow: {'converged' if net.converged else 'FAILED'}")
        except Exception as e:
            print(f"  Power flow error: {e}")
        
        # 如果有tie switch，枚举拓扑数量
        if n_tie > 0:
            print(f"  Tie switches: {list(net.line[~net.line.in_service][['from_bus','to_bus']].itertuples(index=False, name=None))}")
    except Exception as e:
        print(f"\n{name}: NOT AVAILABLE - {e}")

print("\n=== CIGRE MV详细检查 ===")
try:
    net = pn.create_cigre_network_mv()
    print(f"buses={len(net.bus)}, lines={len(net.line)}")
    print(f"switches: {len(net.switch) if hasattr(net,'switch') else 'N/A'}")
    # 检查open lines
    open_lines = net.line[~net.line.in_service]
    print(f"open lines (tie switches): {len(open_lines)}")
    if len(open_lines) > 0:
        print(open_lines[['from_bus','to_bus','in_service']].to_string())
except Exception as e:
    print(f"CIGRE MV error: {e}")
