#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制不同区域的新增确诊人数趋势图（折线图）
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates

# 设置中文字体支持（修复乱码问题）
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'Microsoft YaHei', 'SimHei', 'Noto Sans CJK', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

def read_covid_data():
    """
    读取疫情数据
    
    Returns:
        pandas.DataFrame: 疫情数据
    """
    file_path = "香港各区疫情数据_20250322.xlsx"
    
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return None
    
    try:
        df = pd.read_excel(file_path)
        df['报告日期'] = pd.to_datetime(df['报告日期'])
        print(f"成功读取数据，共 {len(df)} 行")
        return df
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return None

def plot_regions_new_cases():
    """
    绘制不同区域的新增确诊趋势图
    """
    # 读取数据
    df = read_covid_data()
    if df is None:
        return
    
    # 获取所有区域名称
    regions = df['地区名称'].unique()
    print(f"共有 {len(regions)} 个区域: {regions.tolist()}")
    
    # 创建图形
    plt.figure(figsize=(14, 10))
    
    # 定义颜色方案
    colors = plt.cm.tab10.colors  # 使用tab10调色板（10种颜色）
    
    # 为每个区域绘制趋势线
    for i, region in enumerate(regions):
        # 过滤该区域的数据
        region_data = df[df['地区名称'] == region]
        
        # 按日期汇总新增确诊
        daily_region = region_data.groupby('报告日期')['新增确诊'].sum().sort_index()
        
        # 绘制折线图
        if len(daily_region) > 0:
            plt.plot(daily_region.index, daily_region.values,
                    linewidth=2, marker='o', markersize=3,
                    label=region, color=colors[i % len(colors)])
    
    # 设置图表属性
    plt.title('香港各区域新增确诊人数趋势对比', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('报告日期', fontsize=12)
    plt.ylabel('新增确诊人数', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    # 格式化日期显示
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    # 添加网格和调整布局
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('各区域新增确诊趋势图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 各区域新增确诊趋势图已保存为 '各区域新增确诊趋势图.png'")
    
    # 打印统计信息
    print("\n=== 区域统计信息 ===")
    region_stats = df.groupby('地区名称')['新增确诊'].sum().sort_values(ascending=False)
    print("新增确诊最多的前10个区域:")
    for i, (region, total) in enumerate(region_stats.head(10).items(), 1):
        print(f"{i:2d}. {region:<8}: {total:>8} 例")

def plot_top_regions_comparison():
    """
    绘制新增确诊最多的前5个区域对比图（更清晰的版本）
    """
    df = read_covid_data()
    if df is None:
        return
    
    # 找出累计新增确诊最多的前5个区域
    top_regions = df.groupby('地区名称')['新增确诊'].sum().nlargest(5).index.tolist()
    print(f"\n新增确诊前5的区域: {top_regions}")
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 为前5区域绘制趋势线
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, region in enumerate(top_regions):
        region_data = df[df['地区名称'] == region].groupby('报告日期')['新增确诊'].sum().sort_index()
        plt.plot(region_data.index, region_data.values,
                linewidth=2, marker='o', markersize=4,
                label=region, color=colors[i])
    
    # 设置图表属性
    plt.title('香港新增确诊前5区域趋势对比', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('报告日期', fontsize=12)
    plt.ylabel('新增确诊人数', fontsize=12)
    plt.legend(loc='upper left')
    
    # 格式化日期显示
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    # 添加网格和调整布局
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('前5区域新增确诊对比图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 前5区域对比图已保存为 '前5区域新增确诊对比图.png'")

def main():
    """
    主函数
    """
    print("香港各区域新增确诊趋势分析")
    print("=" * 30)
    
    # 绘制所有区域趋势图
    print("\n正在绘制所有区域新增确诊趋势图...")
    plot_regions_new_cases()
    
    # 绘制前5区域对比图
    print("\n正在绘制前5区域新增确诊对比图...")
    plot_top_regions_comparison()
    
    print("\n✓ 图表生成完成！")

if __name__ == "__main__":
    main()