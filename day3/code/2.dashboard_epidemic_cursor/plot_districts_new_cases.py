#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制不同区的新增确诊人数折线图，横坐标为报告日期
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# 设置中文字体，避免中文显示乱码
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_districts_new_cases():
    """
    绘制不同区的新增确诊人数折线图
    """
    # Excel 文件路径
    excel_file = "香港各区疫情数据_20250322.xlsx"
    
    try:
        # 读取 Excel 文件
        df = pd.read_excel(excel_file)
        
        # 确保报告日期是日期类型
        df['报告日期'] = pd.to_datetime(df['报告日期'])
        
        # 获取所有区名称
        districts = df['地区名称'].unique()
        districts = sorted(districts)
        
        # 创建图表
        plt.figure(figsize=(16, 9))
        
        # 定义颜色列表，为每个区分配不同颜色
        colors = plt.cm.tab20(np.linspace(0, 1, len(districts)))
        
        # 为每个区绘制折线图
        for i, district in enumerate(districts):
            # 筛选该区的数据
            district_data = df[df['地区名称'] == district].copy()
            district_data = district_data.sort_values('报告日期')
            
            # 绘制折线
            plt.plot(district_data['报告日期'], district_data['新增确诊'], 
                    linewidth=1.5, label=district, color=colors[i], alpha=0.8)
        
        # 设置标题和标签
        plt.title('香港各区新增确诊人数趋势', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('报告日期', fontsize=14)
        plt.ylabel('新增确诊人数', fontsize=14)
        
        # 设置日期格式
        date_range = df['报告日期'].max() - df['报告日期'].min()
        days = date_range.days
        if days > 100:
            interval = max(1, days // 15)
        else:
            interval = max(1, days // 10)
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        plt.xticks(rotation=45, ha='right')
        
        # 添加图例
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
        
        # 添加网格
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        output_file = '各区新增确诊人数趋势图.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图表已保存为: {output_file}")
        
        # 显示图表
        plt.show()
        
        # 显示统计信息
        print("\n" + "=" * 80)
        print("各区数据统计:")
        print("=" * 80)
        
        district_stats = []
        for district in districts:
            district_data = df[df['地区名称'] == district]
            total_cases = district_data['新增确诊'].sum()
            avg_cases = district_data['新增确诊'].mean()
            max_cases = district_data['新增确诊'].max()
            max_date = district_data.loc[district_data['新增确诊'].idxmax(), '报告日期']
            
            district_stats.append({
                '地区': district,
                '累计新增': total_cases,
                '日均新增': avg_cases,
                '单日最高': max_cases,
                '最高日期': max_date
            })
        
        # 按累计新增排序
        district_stats.sort(key=lambda x: x['累计新增'], reverse=True)
        
        print(f"{'地区':<12} {'累计新增':>12} {'日均新增':>12} {'单日最高':>12} {'最高日期':>12}")
        print("-" * 80)
        for stat in district_stats:
            print(f"{stat['地区']:<12} {stat['累计新增']:>12,} {stat['日均新增']:>12.2f} "
                  f"{stat['单日最高']:>12,} {stat['最高日期'].strftime('%Y-%m-%d'):>12}")
        
        print("=" * 80)
        
        return df
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{excel_file}'")
        return None
    except Exception as e:
        print(f"错误: 处理文件时出现问题 - {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    plot_districts_new_cases()
