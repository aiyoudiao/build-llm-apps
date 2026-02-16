#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flask 应用：香港疫情数据可视化大屏
"""

from flask import Flask, render_template, jsonify
import pandas as pd
import json
from datetime import datetime

app = Flask(__name__)

# 映射：Excel 中的简体地区名称 -> 政府 GeoJSON 中使用的繁体「地區」名称
# 注意：部分地區在 GeoJSON 中不带「區」字，如「九龍城」「油尖旺」「深水埗」等
DISTRICT_NAME_MAP = {
    "中西区": "中西區",
    "湾仔区": "灣仔",
    "东区": "東區",
    "南区": "南區",
    "油尖旺区": "油尖旺",
    "深水埗区": "深水埗",
    "九龙城区": "九龍城",
    "黄大仙区": "黃大仙",
    "观塘区": "觀塘",
    "葵青区": "葵青",
    "荃湾区": "荃灣",
    "屯门区": "屯門",
    "元朗区": "元朗",
    "北区": "北區",
    "大埔区": "大埔",
    "沙田区": "沙田",
    "西贡区": "西貢",
    "离岛区": "離島",
}

# 读取数据
def load_data():
    """
    加载 Excel 数据文件
    """
    file_path = '香港各区疫情数据_20250322.xlsx'
    try:
        df = pd.read_excel(file_path)
        df['报告日期'] = pd.to_datetime(df['报告日期'])
        return df
    except Exception as e:
        print(f"错误: 加载数据时出现问题 - {e}")
        return None

@app.route('/')
def index():
    """
    主页面路由
    """
    return render_template('index.html')

@app.route('/api/summary')
def get_summary():
    """
    获取数据概览统计
    """
    df = load_data()
    if df is None:
        return jsonify({'error': '数据加载失败'}), 500
    
    # 获取最新日期的数据
    latest_date = df['报告日期'].max()
    latest_data = df[df['报告日期'] == latest_date]
    
    summary = {
        'date': latest_date.strftime('%Y-%m-%d'),
        'cumulative': int(latest_data['累计确诊'].sum()),
        'active': int(latest_data['现存确诊'].sum()),
        'recovered': int(latest_data['累计康复'].sum()),
        'deaths': int(latest_data['累计死亡'].sum()),
        'new_cases': int(latest_data['新增确诊'].sum())
    }
    
    return jsonify(summary)

@app.route('/api/trend')
def get_trend():
    """
    获取全港每日新增确诊趋势数据
    """
    df = load_data()
    if df is None:
        return jsonify({'error': '数据加载失败'}), 500
    
    # 按日期汇总新增确诊
    daily_data = df.groupby('报告日期')['新增确诊'].sum().reset_index()
    daily_data = daily_data.sort_values('报告日期')
    
    dates = [d.strftime('%Y-%m-%d') for d in daily_data['报告日期']]
    new_cases = daily_data['新增确诊'].tolist()
    
    return jsonify({
        'dates': dates,
        'new_cases': new_cases
    })

@app.route('/api/districts/ranking')
def get_districts_ranking():
    """
    获取各区累计确诊排行数据
    """
    df = load_data()
    if df is None:
        return jsonify({'error': '数据加载失败'}), 500
    
    # 获取最新日期的数据
    latest_date = df['报告日期'].max()
    latest_data = df[df['报告日期'] == latest_date]
    
    # 按累计确诊排序
    district_data = latest_data[['地区名称', '累计确诊']].sort_values('累计确诊', ascending=False)
    
    districts = district_data['地区名称'].tolist()
    cumulative_cases = district_data['累计确诊'].tolist()
    
    return jsonify({
        'districts': districts,
        'cumulative_cases': cumulative_cases
    })

@app.route('/api/districts/trend')
def get_districts_trend():
    """
    获取重点地区（Top 5）新增确诊趋势对比数据
    """
    df = load_data()
    if df is None:
        return jsonify({'error': '数据加载失败'}), 500
    
    # 计算各区累计新增确诊总数
    district_totals = df.groupby('地区名称')['新增确诊'].sum().sort_values(ascending=False)
    top5_districts = district_totals.head(5).index.tolist()
    
    # 获取日期列表
    dates = sorted(df['报告日期'].unique())
    date_strs = [d.strftime('%Y-%m-%d') for d in dates]
    
    # 为每个区准备数据
    series_data = []
    for district in top5_districts:
        district_df = df[df['地区名称'] == district].sort_values('报告日期')
        # 创建日期到新增确诊的映射
        date_dict = dict(zip(district_df['报告日期'], district_df['新增确诊']))
        new_cases = [date_dict.get(d, 0) for d in dates]
        series_data.append({
            'name': district,
            'data': new_cases
        })
    
    return jsonify({
        'dates': date_strs,
        'series': series_data
    })

@app.route('/api/risk/distribution')
def get_risk_distribution():
    """
    获取各区风险等级分布数据（最新日期）
    """
    df = load_data()
    if df is None:
        return jsonify({'error': '数据加载失败'}), 500
    
    # 获取最新日期的数据
    latest_date = df['报告日期'].max()
    latest_data = df[df['报告日期'] == latest_date]
    
    # 统计各风险等级的数量
    risk_counts = latest_data['风险等级'].value_counts().to_dict()
    
    # 转换为 ECharts 需要的格式
    data = [{'name': k, 'value': int(v)} for k, v in risk_counts.items()]
    
    return jsonify({
        'data': data
    })


@app.route('/api/map/districts')
def get_map_districts():
    """
    获取地图展示用的各区疫情数据
    - 颜色默认使用整个时间段内“累计新增确诊”（所有天的新增确诊之和）
    - 其他指标（现存确诊、累计确诊、累计康复、累计死亡）使用最新日期的快照
    """
    df = load_data()
    if df is None:
        return jsonify({'error': '数据加载失败'}), 500

    # 1. 整个时间段：按区统计“累计新增确诊”（所有天的新增确诊之和）
    total_new_by_district = (
        df.groupby('地区名称')['新增确诊']
        .sum()
        .rename('total_new_cases')
    )

    # 2. 最新日期快照：现存确诊、累计确诊、累计康复、累计死亡
    latest_date = df['报告日期'].max()
    latest_data = df[df['报告日期'] == latest_date]

    latest_grouped = (
        latest_data.groupby('地区名称')[['现存确诊', '累计确诊', '累计康复', '累计死亡']]
        .sum()
    )

    # 3. 合并：全周期累计新增确诊 + 最新快照
    merged = latest_grouped.join(total_new_by_district, on='地区名称').reset_index()

    items = []
    for _, row in merged.iterrows():
        original_name = row['地区名称']
        geo_name = DISTRICT_NAME_MAP.get(original_name, original_name)
        items.append({
            # 与 GeoJSON 中 properties.地區 对齐
            "name": geo_name,
            # 地图颜色默认使用整个时间段的“累计新增确诊”
            "total_new_cases": int(row['total_new_cases']),
            # 最新日期快照指标
            "active": int(row['现存确诊']),
            "cumulative": int(row['累计确诊']),
            "recovered": int(row['累计康复']),
            "deaths": int(row['累计死亡']),
            # 额外保留原始简体名称，方便 tooltip 显示
            "district": original_name,
        })

    return jsonify({
        "date": latest_date.strftime('%Y-%m-%d'),
        "items": items
    })

if __name__ == '__main__':
    print("=" * 80)
    print("香港疫情数据可视化大屏")
    print("=" * 80)
    print("访问地址: http://127.0.0.1:5002")
    print("=" * 80)
    app.run(debug=True, host='0.0.0.0', port=5002)
