import pandapower as pp
import pandapower.networks as pn
import inspect, warnings
warnings.filterwarnings("ignore")

# 列出所有case函数
all_funcs = [name for name, obj in inspect.getmembers(pn) if callable(obj) and not name.startswith("_")]
case_funcs = [f for f in all_funcs if f.startswith("case")]
print("All case networks:", case_funcs)
print()

# 逐个检查有tie switch的
print("Networks with tie switches (open lines):")
for fname in case_funcs:
    try:
        net = getattr(pn, fname)()
        open_l = len(net.line[~net.line.in_service])
        if open_l >= 2:
            print(f"  {fname}: buses={len(net.bus)}, open_lines={open_l}")
    except: pass
