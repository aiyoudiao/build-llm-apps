import pandas as pd
import os

# Define file paths
base_dir = '/Users/zakj/Desktop/block_mac/do/2026/AI 大模型应用/build-llm-apps/day3/code/1.excel_merge_trae'
file_info = os.path.join(base_dir, '员工基本信息表.xlsx')
file_perf = os.path.join(base_dir, '员工绩效表.xlsx')
output_file = os.path.join(base_dir, '员工2024年绩效表.xlsx')

try:
    # 1. Read the Excel files
    print(f"Reading {file_info}...")
    df_info = pd.read_excel(file_info)
    
    print(f"Reading {file_perf}...")
    df_perf = pd.read_excel(file_perf)

    # 2. Filter performance data for 2024
    print("Filtering for 2024 performance data...")
    df_perf_2024 = df_perf[df_perf['年度'] == 2024].copy()
    
    if df_perf_2024.empty:
        print("Warning: No performance data found for 2024.")
    else:
        print(f"Found {len(df_perf_2024)} records for 2024.")

    # 3. Pivot the data to have quarters as columns
    # We want one row per employee, with columns for each quarter's score
    print("Pivoting performance data...")
    df_pivot = df_perf_2024.pivot(index='员工ID', columns='季度', values='绩效评分')
    
    # Rename columns to be more descriptive (e.g., 1 -> 2024Q1)
    df_pivot.columns = [f'2024Q{col}' for col in df_pivot.columns]
    
    # Reset index so '员工ID' becomes a column again for merging
    df_pivot.reset_index(inplace=True)

    # 4. Merge the dataframes
    print("Merging dataframes...")
    df_merged = pd.merge(df_info, df_pivot, on='员工ID', how='left')

    # 5. Save to new Excel file
    print(f"Saving merged data to {output_file}...")
    df_merged.to_excel(output_file, index=False)
    
    print("Successfully created 员工2024年绩效表.xlsx")
    print("\nPreview of the merged data:")
    print(df_merged.head())

except Exception as e:
    print(f"An error occurred: {e}")
