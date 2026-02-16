# -*- coding: utf-8 -*-
"""
读取员工基本信息表和员工绩效表的前5行数据
"""

import os
import pandas as pd


def main():
    # 文件路径（与脚本同目录）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_basic = os.path.join(base_dir, "员工基本信息表.xlsx")
    file_perf = os.path.join(base_dir, "员工绩效表.xlsx")

    # 读取员工基本信息表，前5行
    df_basic = pd.read_excel(file_basic, engine="openpyxl", nrows=5)
    print("=== 员工基本信息表（前5行）===")
    print(df_basic.to_string())
    print()

    # 读取员工绩效表，前5行
    df_perf = pd.read_excel(file_perf, engine="openpyxl", nrows=5)
    print("=== 员工绩效表（前5行）===")
    print(df_perf.to_string())


if __name__ == "__main__":
    main()
