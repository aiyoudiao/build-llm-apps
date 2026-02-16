import pandas as pd

# 设置显示选项，以显示所有列并确保中文字符对齐/显示正确
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

file_path = '香港各区疫情数据_20250322.xlsx'

try:
    # 读取 Excel 文件
    df = pd.read_excel(file_path)
    
    # 获取前 20 行数据
    first_20_rows = df.head(20)
    
    print("前 20 行数据：")
    print(first_20_rows)
    
except Exception as e:
    print(f"读取文件时发生错误: {e}")
