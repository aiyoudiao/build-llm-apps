#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
读取香港各区疫情数据 Excel 文件的前20行数据
"""

import pandas as pd

def read_excel_first_20_rows():
    """
    读取 Excel 文件的前20行数据
    """
    # Excel 文件路径
    excel_file = "香港各区疫情数据_20250322.xlsx"
    
    try:
        # 读取 Excel 文件
        df = pd.read_excel(excel_file)
        
        # 获取前20行数据
        first_20_rows = df.head(20)
        
        # 显示数据信息
        print("=" * 80)
        print(f"Excel 文件: {excel_file}")
        print(f"总行数: {len(df)}")
        print(f"总列数: {len(df.columns)}")
        print("=" * 80)
        print("\n前20行数据:")
        print("=" * 80)
        print(first_20_rows.to_string())
        print("=" * 80)
        
        # 显示列名
        print("\n列名:")
        print("-" * 80)
        for i, col in enumerate(df.columns, 1):
            print(f"{i}. {col}")
        print("-" * 80)
        
        return first_20_rows
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{excel_file}'")
        return None
    except Exception as e:
        print(f"错误: 读取文件时出现问题 - {e}")
        return None

if __name__ == "__main__":
    read_excel_first_20_rows()
