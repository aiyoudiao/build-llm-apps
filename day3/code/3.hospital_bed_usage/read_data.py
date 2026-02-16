import pandas as pd
import os

# 定义文件路径
file_path = 'hospital_bed_usage_data.xlsx'

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found.")
else:
    try:
        # 读取Excel文件的前20行数据
        # nrows=20 只读取前20行
        df = pd.read_excel(file_path, nrows=20)
        
        # 打印数据
        print("前20行数据如下:")
        print(df)
        
        # 如果需要更详细的信息，也可以打印列名
        print("\n列名:")
        print(df.columns.tolist())
        
    except Exception as e:
        print(f"发生错误: {e}")
