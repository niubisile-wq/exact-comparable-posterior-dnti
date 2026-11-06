import json, os
path = r"<LOCAL_WORKSPACE>\digital-twin-dataset\sample_dataset\topology\network_files\circuit3\2023-08-01T00h00m00.000000s.json"
with open(path, encoding="utf-8") as f:
    d = json.load(f)
print("Top-level keys:", list(d.keys())[:10])
for k in list(d.keys())[:5]:
    v = d[k]
    if isinstance(v, list): print(f"  {k}: list of {len(v)}")
    elif isinstance(v, dict): print(f"  {k}: dict with keys {list(v.keys())[:5]}")
    else: print(f"  {k}: {v}")
