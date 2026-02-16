# -*- coding: utf-8 -*-
"""
将员工基本信息表与 2024 年每一季度绩效合并，保存为 员工2024年绩效表.xlsx
"""

import os
import pandas as pd


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_basic = os.path.join(base_dir, "员工基本信息表.xlsx")
    file_perf = os.path.join(base_dir, "员工绩效表.xlsx")
    file_out = os.path.join(base_dir, "员工2024年绩效表.xlsx")

    # 读取员工基本信息表（全部）
    df_basic = pd.read_excel(file_basic, engine="openpyxl")
    # 读取员工绩效表（全部）
    df_perf = pd.read_excel(file_perf, engine="openpyxl")

    # 筛选 2024 年绩效
    df_2024 = df_perf[df_perf["年度"] == 2024][["员工ID", "季度", "绩效评分"]].copy()
    # 透视：每个员工一行，列为 第1季度、第2季度、第3季度、第4季度 的绩效评分
    df_pivot = df_2024.pivot(index="员工ID", columns="季度", values="绩效评分").reset_index()
    # 列名改为中文季度描述
    quarter_names = {1: "2024年第1季度绩效", 2: "2024年第2季度绩效", 3: "2024年第3季度绩效", 4: "2024年第4季度绩效"}
    df_pivot = df_pivot.rename(columns=quarter_names)

    # 以基本信息表为基准，左连接 2024 年各季度绩效（无绩效的单元格为空）
    df_merged = df_basic.merge(df_pivot, on="员工ID", how="left")
    # 确保列顺序：基本信息 + 第1～4季度
    quarter_cols = ["2024年第1季度绩效", "2024年第2季度绩效", "2024年第3季度绩效", "2024年第4季度绩效"]
    df_merged = df_merged[[c for c in df_merged.columns if c not in quarter_cols] + [q for q in quarter_cols if q in df_merged.columns]]

    # 保存到新 Excel
    df_merged.to_excel(file_out, index=False, engine="openpyxl")
    print(f"已合并并保存到: {file_out}")
    print("前5行预览:")
    print(df_merged.head().to_string())


if __name__ == "__main__":
    main()
