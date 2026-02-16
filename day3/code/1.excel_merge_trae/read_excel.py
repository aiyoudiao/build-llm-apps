import pandas as pd
import os

# Define file paths
base_dir = '/Users/zakj/Desktop/block_mac/do/2026/AI 大模型应用/build-llm-apps/day3/code/1.excel_merge_trae'
file1 = os.path.join(base_dir, '员工基本信息表.xlsx')
file2 = os.path.join(base_dir, '员工绩效表.xlsx')

# Read and print first 5 rows of file 1
print(f"Reading {file1}...")
try:
    df1 = pd.read_excel(file1)
    print("Top 5 rows of 员工基本信息表.xlsx:")
    print(df1.head(5))
except Exception as e:
    print(f"Error reading {file1}: {e}")

print("\n" + "="*50 + "\n")

# Read and print first 5 rows of file 2
print(f"Reading {file2}...")
try:
    df2 = pd.read_excel(file2)
    print("Top 5 rows of 员工绩效表.xlsx:")
    print(df2.head(5))
except Exception as e:
    print(f"Error reading {file2}: {e}")
