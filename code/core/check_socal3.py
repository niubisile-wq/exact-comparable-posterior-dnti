import json
path = r"<LOCAL_WORKSPACE>\digital-twin-dataset\sample_dataset\topology\network_files\circuit3\2023-08-01T00h00m00.000000s.json"
with open(path, encoding="utf-8") as f:
    d = json.load(f)

# 看Line结构
print("=== Line sample (first 2) ===")
for line in d["Line"][:2]:
    print(line)

print("\n=== Bus sample (first 3) ===")
for bus in d["Bus"][:3]:
    print(bus)

print("\n=== SwitchMultiPosition (all, NO状态=tie switch) ===")
no_count = 0
for sw in d["SwitchMultiPosition"]:
    statuses = [t["status"] for t in sw["tbus"]]
    if "NO" in statuses:
        no_count += 1
        print(f"  {sw['name']}: {sw['fbus']} -> {[t['name'] for t in sw['tbus']]} status={statuses}")
print(f"Total with NO (open) positions: {no_count}")
