import pandas as pd
import os

# Define file paths
base_dir = '/Users/zakj/Desktop/block_mac/do/2026/AI 大模型应用/build-llm-apps/day3/code/1.excel_merge_trae'
file_info = os.path.join(base_dir, '员工基本信息表.xlsx')
file_perf = os.path.join(base_dir, '员工绩效表.xlsx')
output_file = os.path.join(base_dir, '员工Q4绩效表.xlsx')

try:
    # 1. Read the Excel files
    print(f"Reading {file_info}...")
    df_info = pd.read_excel(file_info)
    
    print(f"Reading {file_perf}...")
    df_perf = pd.read_excel(file_perf)

    # 2. Filter performance data for 2024 Q4
    print("Filtering for 2024 Q4 performance data...")
    df_perf_q4 = df_perf[(df_perf['年度'] == 2024) & (df_perf['季度'] == 4)]
    
    # Check if we have data after filtering
    if df_perf_q4.empty:
        print("Warning: No performance data found for 2024 Q4.")
    else:
        print(f"Found {len(df_perf_q4)} records for 2024 Q4.")

    # 3. Merge the dataframes on '员工ID'
    # We use a left join to keep all employees from the info table, 
    # even if they don't have a Q4 score (though ideally they should).
    print("Merging dataframes...")
    df_merged = pd.merge(df_info, df_perf_q4[['员工ID', '绩效评分']], on='员工ID', how='left')
    
    # Rename '绩效评分' to clearly indicate it's for Q4
    df_merged.rename(columns={'绩效评分': '2024Q4绩效评分'}, inplace=True)

    # 4. Save to new Excel file
    print(f"Saving merged data to {output_file}...")
    df_merged.to_excel(output_file, index=False)
    
    print("Successfully created 员工Q4绩效表.xlsx")
    print("\nPreview of the merged data:")
    print(df_merged.head())

except Exception as e:
    print(f"An error occurred: {e}")
