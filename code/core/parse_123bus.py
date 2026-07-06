import pandas as pd

sw_path = r"<LOCAL_WORKSPACE>\feeder123\feeder123\switch data.xls"
df_sw = pd.read_excel(sw_path, header=None)
print("=== Switch Data (all rows) ===")
print(df_sw.to_string())

line_path = r"<LOCAL_WORKSPACE>\feeder123\feeder123\line data.xls"
df_line = pd.read_excel(line_path, header=None)
print(f"\n=== Line Data: {len(df_line)} rows ===")
print(df_line.head(8).to_string())
