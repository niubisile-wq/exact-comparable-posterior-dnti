import pandas as pd

# 读配置（线路参数）数据
config_path = r"<LOCAL_WORKSPACE>\feeder123\feeder123\config data.xls"
df_c = pd.read_excel(config_path, header=None)
print("=== Config Data (line impedances) ===")
print(df_c.to_string())

# 读负荷数据
load_path = r"<LOCAL_WORKSPACE>\feeder123\feeder123\spot loads data.xls"
df_l = pd.read_excel(load_path, header=None)
print(f"\n=== Spot Load Data: {len(df_l)} rows ===")
print(df_l.head(15).to_string())
