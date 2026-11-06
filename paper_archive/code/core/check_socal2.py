import json, pandas as pd, os, glob
path = r"<LOCAL_WORKSPACE>\digital-twin-dataset\sample_dataset\topology\network_files\circuit3\2023-08-01T00h00m00.000000s.json"
with open(path, encoding="utf-8") as f:
    d = json.load(f)

# 网络规模
print(f"Bus: {len(d['Bus'])}, Line: {len(d['Line'])}, Switch: {len(d['Switch'])}")
print(f"SwitchMultiPosition (tie switches): {len(d['SwitchMultiPosition'])}")
print(f"CB (circuit breakers): {len(d['CB'])}")
print(f"Transformer: {len(d['Transformer'])}")
print()

# 看一个Switch的结构
sw = d['SwitchMultiPosition'][0]
print("SwitchMultiPosition sample:", {k:v for k,v in sw.items() if k != 'positions'})

# 读电路断路器状态CSV
cb_dir = r"<LOCAL_WORKSPACE>\digital-twin-dataset\sample_dataset\topology\parameter_timeseries\circuit3"
csvs = glob.glob(cb_dir + "\*.csv")
print(f"\nCircuit breaker CSV files: {len(csvs)}")
if csvs:
    df = pd.read_csv(csvs[0])
    print(f"Sample CB file columns: {list(df.columns)}")
    print(df.head(3).to_string())
