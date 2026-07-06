import pandas as pd

cond_path = r"<LOCAL_WORKSPACE>\common123\Conductor Data.xls"
df = pd.read_excel(cond_path, header=None)
print("=== Conductor Data ===")
print(df.to_string())
