import re
path = r"<LOCAL_WORKSPACE>\ElectricalSystemsDataForReconfiguration\SystemData_119.txt"
with open(path, encoding="utf-8") as f:
    content = f.read()
branch_lines = []
for line in content.split("\n"):
    s = line.strip()
    if re.match(r"^\d+\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+", s):
        nums = re.findall(r"[-+]?\d*\.?\d+", s)
        if len(nums) >= 5:
            branch_lines.append((int(nums[0]),int(nums[1]),int(nums[2]),float(nums[3]),float(nums[4])))

# 找R=0且X=0的支路
zero_branches = [(b[0],b[1],b[2],b[3],b[4]) for b in branch_lines if b[3]==0 and b[4]==0]
print(f"Branches with R=0, X=0: {len(zero_branches)}")
for b in zero_branches:
    print(f"  FB={b[0]} TB={b[1]} Br#{b[2]} R={b[3]} X={b[4]}")
    
# 也找R很小的
small_branches = [(b[0],b[1],b[2],b[3],b[4]) for b in branch_lines if b[3]<0.001 or b[4]<0.001]
print(f"\nBranches with R<0.001 or X<0.001: {len(small_branches)}")
for b in small_branches[:10]:
    print(f"  FB={b[0]} TB={b[1]} Br#{b[2]} R={b[3]} X={b[4]}")
