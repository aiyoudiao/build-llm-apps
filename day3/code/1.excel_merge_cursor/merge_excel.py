# -*- coding: utf-8 -*-
"""
将员工基本信息表与 2024 年第 4 季度绩效合并，保存为 员工基本信息及绩效表.xlsx
"""

import os
import pandas as pd


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_basic = os.path.join(base_dir, "员工基本信息表.xlsx")
    file_perf = os.path.join(base_dir, "员工绩效表.xlsx")
    file_out = os.path.join(base_dir, "员工基本信息及绩效表.xlsx")

    # 读取员工基本信息表（全部）
    df_basic = pd.read_excel(file_basic, engine="openpyxl")
    # 读取员工绩效表（全部）
    df_perf = pd.read_excel(file_perf, engine="openpyxl")

    # 筛选 2024 年第 4 季度绩效，只保留 员工ID 和 绩效评分
    df_q4 = df_perf[(df_perf["年度"] == 2024) & (df_perf["季度"] == 4)][["员工ID", "绩效评分"]].copy()
    df_q4 = df_q4.rename(columns={"绩效评分": "2024年第4季度绩效"})

    # 以基本信息表为基准，左连接 2024Q4 绩效（无绩效的员工该列为空）
    df_merged = df_basic.merge(df_q4, on="员工ID", how="left")

    # 保存到新 Excel
    df_merged.to_excel(file_out, index=False, engine="openpyxl")
    print(f"已合并并保存到: {file_out}")
    print("前5行预览:")
    print(df_merged.head().to_string())


if __name__ == "__main__":
    main()
