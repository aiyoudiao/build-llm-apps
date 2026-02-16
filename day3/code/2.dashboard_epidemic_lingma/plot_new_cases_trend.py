#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制香港各区疫情新增确诊人数趋势图
横坐标：报告日期，纵坐标：新增确诊人数
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.dates as mdates

# 设置中文字体支持（修复乱码）
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'Microsoft YaHei', 'SimHei', 'Noto Sans CJK', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

def read_covid_data():
    """
    读取完整的疫情数据
    
    Returns:
        pandas.DataFrame: 疫情数据
    """
    file_path = "香港各区疫情数据_20250322.xlsx"
    
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return None
    
    try:
        # 读取完整数据
        df = pd.read_excel(file_path)
        print(f"成功读取数据，共 {len(df)} 行")
        return df
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return None

def plot_new_cases_trend():
    """
    绘制新增确诊人数趋势图
    """
    # 读取数据
    df = read_covid_data()
    if df is None:
        return
    
    # 数据预处理
    # 将报告日期转换为datetime格式
    df['报告日期'] = pd.to_datetime(df['报告日期'])
    
    # 按日期汇总新增确诊人数（所有区域合计）
    daily_total = df.groupby('报告日期')['新增确诊'].sum().reset_index()
    daily_total = daily_total.sort_values('报告日期')
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制折线图
    plt.plot(daily_total['报告日期'], daily_total['新增确诊'], 
             marker='o', linewidth=2, markersize=4, color='#2E86AB')
    
    # 设置标题和标签
    plt.title('香港疫情新增确诊人数趋势图', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('报告日期', fontsize=12)
    plt.ylabel('新增确诊人数', fontsize=12)
    
    # 格式化x轴日期显示
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 设置y轴从0开始
    plt.ylim(bottom=0)
    
    # 添加数据标签（每隔几天标注一次）
    for i in range(0, len(daily_total), 3):
        if i < len(daily_total):
            plt.annotate(str(daily_total.iloc[i]['新增确诊']), 
                        (daily_total.iloc[i]['报告日期'], daily_total.iloc[i]['新增确诊']),
                        textcoords="offset points", xytext=(0,10), ha='center',
                        fontsize=8, color='red')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('新增确诊趋势图.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 '新增确诊趋势图.png'")
    
    # 显示图形
    plt.show()
    
    # 打印统计信息
    print("\n数据统计信息:")
    print(f"数据时间范围: {daily_total['报告日期'].min().strftime('%Y-%m-%d')} 到 {daily_total['报告日期'].max().strftime('%Y-%m-%d')}")
    print(f"总天数: {len(daily_total)} 天")
    print(f"最高单日新增: {daily_total['新增确诊'].max()} 例")
    print(f"最低单日新增: {daily_total['新增确诊'].min()} 例")
    print(f"平均每日新增: {daily_total['新增确诊'].mean():.1f} 例")

def plot_top_regions():
    """
    绘制新增确诊人数最多的前5个区域趋势图
    """
    # 读取数据
    df = read_covid_data()
    if df is None:
        return
    
    # 数据预处理
    df['报告日期'] = pd.to_datetime(df['报告日期'])
    
    # 找出累计新增确诊最多的前5个区域
    top_regions = df.groupby('地区名称')['新增确诊'].sum().nlargest(5).index.tolist()
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 为每个区域绘制趋势线
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, region in enumerate(top_regions):
        region_data = df[df['地区名称'] == region].groupby('报告日期')['新增确诊'].sum().reset_index()
        region_data = region_data.sort_values('报告日期')
        
        plt.plot(region_data['报告日期'], region_data['新增确诊'], 
                marker='o', linewidth=2, markersize=3, 
                label=region, color=colors[i])
    
    # 设置图表属性
    plt.title('香港新增确诊人数前5区域趋势对比', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('报告日期', fontsize=12)
    plt.ylabel('新增确诊人数', fontsize=12)
    plt.legend(loc='upper left')
    
    # 格式化日期显示
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    
    # 添加网格和调整布局
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.tight_layout()
    
    # 保存和显示
    plt.savefig('区域新增确诊趋势对比.png', dpi=300, bbox_inches='tight')
    print("区域对比图已保存为 '区域新增确诊趋势对比.png'")
    plt.show()

def main():
    """
    主函数
    """
    print("香港疫情新增确诊趋势分析")
    print("=" * 30)
    
    # 绘制总体趋势图
    print("\n正在绘制总体新增确诊趋势图...")
    plot_new_cases_trend()
    
    # 绘制区域对比图
    print("\n正在绘制区域新增确诊对比图...")
    plot_top_regions()
    
    print("\n图表生成完成！")

if __name__ == "__main__":
    main()