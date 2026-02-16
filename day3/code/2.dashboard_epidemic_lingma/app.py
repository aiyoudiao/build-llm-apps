#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask + ECharts 疫情数据可视化大屏
"""

from flask import Flask, render_template, jsonify
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# 读取数据
def load_covid_data():
    """加载疫情数据"""
    file_path = "香港各区疫情数据_20250322.xlsx"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    df = pd.read_excel(file_path)
    df['报告日期'] = pd.to_datetime(df['报告日期'])
    return df

# 全局数据变量
covid_data = None

# 全局数据变量
covid_data = None

def init_data():
    """初始化数据"""
    global covid_data
    try:
        covid_data = load_covid_data()
        print("✅ 数据加载成功")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")

# 在应用启动时初始化数据
init_data()

@app.route('/')
def index():
    """首页 - 可视化大屏"""
    return render_template('index.html')

@app.route('/api/summary')
def get_summary():
    """获取数据概览"""
    if covid_data is None:
        return jsonify({'error': '数据未加载'}), 500
    
    total_days = len(covid_data['报告日期'].unique())
    total_cases = int(covid_data['新增确诊'].sum())
    max_daily_cases = int(covid_data['新增确诊'].max())
    regions_count = len(covid_data['地区名称'].unique())
    
    return jsonify({
        'total_days': total_days,
        'total_cases': total_cases,
        'max_daily_cases': max_daily_cases,
        'regions_count': regions_count,
        'data_range': {
            'start': covid_data['报告日期'].min().strftime('%Y-%m-%d'),
            'end': covid_data['报告日期'].max().strftime('%Y-%m-%d')
        }
    })

@app.route('/api/daily_trend')
def get_daily_trend():
    """获取每日趋势数据"""
    if covid_data is None:
        return jsonify({'error': '数据未加载'}), 500
    
    daily_trend = covid_data.groupby('报告日期')['新增确诊'].sum().sort_index()
    trend_data = [{
        'date': date.strftime('%Y-%m-%d'),
        'cases': int(value)
    } for date, value in daily_trend.items()]
    
    return jsonify(trend_data)

@app.route('/api/region_stats')
def get_region_stats():
    """获取区域统计"""
    if covid_data is None:
        return jsonify({'error': '数据未加载'}), 500
    
    region_stats = covid_data.groupby('地区名称')['新增确诊'].sum().sort_values(ascending=False)
    stats_data = [{
        'region': region,
        'cases': int(value),
        'percentage': round(value / region_stats.sum() * 100, 1)
    } for region, value in region_stats.items()]
    
    return jsonify(stats_data)

@app.route('/api/risk_distribution')
def get_risk_distribution():
    """获取风险等级分布"""
    if covid_data is None:
        return jsonify({'error': '数据未加载'}), 500
    
    risk_dist = covid_data['风险等级'].value_counts().to_dict()
    return jsonify(risk_dist)

@app.route('/api/top_regions_trend')
def get_top_regions_trend():
    """获取前5区域趋势"""
    if covid_data is None:
        return jsonify({'error': '数据未加载'}), 500
    
    top_regions = covid_data.groupby('地区名称')['新增确诊'].sum().nlargest(5).index.tolist()
    trend_data = {}
    
    for region in top_regions:
        region_data = covid_data[covid_data['地区名称'] == region]
        daily_region = region_data.groupby('报告日期')['新增确诊'].sum().sort_index()
        trend_data[region] = [{
            'date': date.strftime('%Y-%m'),
            'cases': int(value)
        } for date, value in daily_region.items()]
    
    return jsonify(trend_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)