import pandapower.networks as pn
import inspect

# 列出所有网络函数
funcs = [name for name, obj in inspect.getmembers(pn) if callable(obj) and not name.startswith("_")]
print("Available networks:", [f for f in funcs if any(k in f.lower() for k in ["case","cigre","rural","mv","lv","kerber","kb"])])

# 试几个可能有tie switch的
candidates = ["case_ieee30","case34sa","case_b4gs","kb_extrem_landnetz_freileitung_1"]
for c in candidates:
    try:
        net = getattr(pn, c)()
        open_l = len(net.line[~net.line.in_service])
        print(f"{c}: buses={len(net.bus)}, open_lines={open_l}")
    except: pass

# 试Kerber网络
for fname in funcs:
    if "kerber" in fname.lower() or "kb" in fname.lower():
        try:
            net = getattr(pn, fname)()
            open_l = len(net.line[~net.line.in_service])
            if open_l > 0:
                print(f"{fname}: buses={len(net.bus)}, tie_switches={open_l}")
        except: pass
