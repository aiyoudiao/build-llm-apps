#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制新增确诊人数图表，横坐标为报告日期
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# 设置中文字体，避免中文显示乱码
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_new_cases():
    """
    绘制新增确诊人数图表
    """
    # Excel 文件路径
    excel_file = "香港各区疫情数据_20250322.xlsx"
    
    try:
        # 读取 Excel 文件
        df = pd.read_excel(excel_file)
        
        # 确保报告日期是日期类型
        df['报告日期'] = pd.to_datetime(df['报告日期'])
        
        # 按报告日期汇总新增确诊人数（所有区的总和）
        daily_new_cases = df.groupby('报告日期')['新增确诊'].sum().reset_index()
        daily_new_cases = daily_new_cases.sort_values('报告日期')
        
        # 创建图表
        plt.figure(figsize=(14, 7))
        plt.plot(daily_new_cases['报告日期'], daily_new_cases['新增确诊'], 
                linewidth=2, color='#e74c3c', marker='o', markersize=3)
        
        # 设置标题和标签
        plt.title('香港新增确诊人数趋势', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('报告日期', fontsize=12)
        plt.ylabel('新增确诊人数', fontsize=12)
        
        # 设置日期格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(daily_new_cases)//10)))
        plt.xticks(rotation=45, ha='right')
        
        # 添加网格
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        output_file = '新增确诊人数趋势图.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图表已保存为: {output_file}")
        
        # 显示图表
        plt.show()
        
        # 显示统计信息
        print("\n" + "=" * 80)
        print("数据统计:")
        print("=" * 80)
        print(f"日期范围: {daily_new_cases['报告日期'].min().strftime('%Y-%m-%d')} 至 {daily_new_cases['报告日期'].max().strftime('%Y-%m-%d')}")
        print(f"总天数: {len(daily_new_cases)} 天")
        print(f"累计新增确诊: {daily_new_cases['新增确诊'].sum():,} 人")
        print(f"日均新增确诊: {daily_new_cases['新增确诊'].mean():.2f} 人")
        print(f"单日最高新增: {daily_new_cases['新增确诊'].max():,} 人")
        print(f"单日最高新增日期: {daily_new_cases.loc[daily_new_cases['新增确诊'].idxmax(), '报告日期'].strftime('%Y-%m-%d')}")
        print("=" * 80)
        
        return daily_new_cases
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{excel_file}'")
        return None
    except Exception as e:
        print(f"错误: 处理文件时出现问题 - {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    plot_new_cases()
