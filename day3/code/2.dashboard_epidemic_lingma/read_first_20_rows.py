#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取香港各区疫情数据Excel文件的前20行数据
"""

import pandas as pd
import os

def read_first_20_rows():
    """
    读取Excel文件的前20行数据并显示
    
    Returns:
        pandas.DataFrame: 包含前20行数据的DataFrame
    """
    # Excel文件路径
    file_path = "香港各区疫情数据_20250322.xlsx"
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return None
    
    try:
        # 读取Excel文件的前20行数据
        print("正在读取Excel文件...")
        df = pd.read_excel(file_path, nrows=20)
        
        # 显示基本信息
        print("=" * 50)
        print("文件基本信息:")
        print(f"文件名: {file_path}")
        print(f"总行数: {len(df)}")
        print(f"列数: {len(df.columns)}")
        print("=" * 50)
        
        # 显示列名
        print("\n列名:")
        for i, column in enumerate(df.columns, 1):
            print(f"{i}. {column}")
        
        # 显示前20行数据
        print("\n前20行数据:")
        print("-" * 50)
        print(df.to_string(index=False))
        print("-" * 50)
        
        # 显示数据类型信息
        print("\n各列数据类型:")
        print(df.dtypes)
        
        return df
        
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return None

def main():
    """
    主函数
    """
    print("香港各区疫情数据读取程序")
    print("=" * 30)
    
    # 读取前20行数据
    data = read_first_20_rows()
    
    if data is not None:
        print(f"\n成功读取 {len(data)} 行数据")
        print("程序执行完毕！")

if __name__ == "__main__":
    main()