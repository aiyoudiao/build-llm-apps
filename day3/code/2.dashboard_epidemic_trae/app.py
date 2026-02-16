from flask import Flask, render_template, jsonify
import pandas as pd
import json

app = Flask(__name__)

# 读取数据
def load_data():
    file_path = '香港各区疫情数据_20250322.xlsx'
    try:
        df = pd.read_excel(file_path)
        df['报告日期'] = pd.to_datetime(df['报告日期'])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/summary')
def get_summary():
    df = load_data()
    if df is None:
        return jsonify({})
    
    # 获取最新日期的数据
    latest_date = df['报告日期'].max()
    latest_df = df[df['报告日期'] == latest_date]
    
    # 计算总计
    total_cumulative = int(latest_df['累计确诊'].sum())
    total_active = int(latest_df['现存确诊'].sum())
    total_recovered = int(latest_df['累计康复'].sum())
    total_deaths = int(latest_df['累计死亡'].sum())
    
    # 获取最新日期字符串
    latest_date_str = latest_date.strftime('%Y-%m-%d')
    
    return jsonify({
        'date': latest_date_str,
        'cumulative': total_cumulative,
        'active': total_active,
        'recovered': total_recovered,
        'deaths': total_deaths
    })

@app.route('/api/trend')
def get_trend():
    df = load_data()
    if df is None:
        return jsonify({})
    
    # 按日期汇总
    daily_trend = df.groupby('报告日期')[['新增确诊', '新增康复', '新增死亡']].sum().reset_index()
    
    dates = daily_trend['报告日期'].dt.strftime('%Y-%m-%d').tolist()
    new_cases = daily_trend['新增确诊'].tolist()
    new_recovered = daily_trend['新增康复'].tolist()
    new_deaths = daily_trend['新增死亡'].tolist()
    
    return jsonify({
        'dates': dates,
        'new_cases': new_cases,
        'new_recovered': new_recovered,
        'new_deaths': new_deaths
    })

@app.route('/api/district_stats')
def get_district_stats():
    df = load_data()
    if df is None:
        return jsonify({})
    
    # 获取最新日期的数据
    latest_date = df['报告日期'].max()
    latest_df = df[df['报告日期'] == latest_date].copy()
    
    # 按累计确诊排序
    top_cumulative = latest_df.sort_values('累计确诊', ascending=False).head(10)
    
    districts = top_cumulative['地区名称'].tolist()
    cumulative_cases = top_cumulative['累计确诊'].tolist()
    incidence_rates = top_cumulative['发病率(每10万人)'].tolist()
    
    return jsonify({
        'districts': districts,
        'cumulative_cases': cumulative_cases,
        'incidence_rates': incidence_rates
    })

@app.route('/api/risk_distribution')
def get_risk_distribution():
    df = load_data()
    if df is None:
        return jsonify({})
    
    # 获取最新日期的数据
    latest_date = df['报告日期'].max()
    latest_df = df[df['报告日期'] == latest_date]
    
    # 统计各风险等级的地区数量
    risk_counts = latest_df['风险等级'].value_counts().reset_index()
    risk_counts.columns = ['risk_level', 'count']
    
    data = []
    for _, row in risk_counts.iterrows():
        data.append({'name': row['risk_level'], 'value': int(row['count'])})
        
    return jsonify(data)

@app.route('/api/district_trend_top5')
def get_district_trend_top5():
    df = load_data()
    if df is None:
        return jsonify({})
        
    # 找出累计确诊最多的前5个地区
    latest_date = df['报告日期'].max()
    latest_df = df[df['报告日期'] == latest_date]
    top5_districts = latest_df.sort_values('累计确诊', ascending=False).head(5)['地区名称'].tolist()
    
    # 获取这5个地区的所有日期数据
    top5_df = df[df['地区名称'].isin(top5_districts)]
    
    # 整理成 Echarts 需要的格式
    # 系列列表
    series_list = []
    dates = sorted(df['报告日期'].unique())
    dates_str = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in dates]
    
    for district in top5_districts:
        district_data = top5_df[top5_df['地区名称'] == district].sort_values('报告日期')
        # 确保日期对齐（虽然此数据集比较规整，但为了健壮性）
        # 这里简化处理，假设每天都有数据
        data = district_data['新增确诊'].tolist()
        series_list.append({
            'name': district,
            'type': 'line',
            'smooth': True,
            'data': data
        })
        
    return jsonify({
        'dates': dates_str,
        'series': series_list,
        'districts': top5_districts
    })

@app.route('/api/map_data')
def get_map_data():
    df = load_data()
    if df is None:
        return jsonify([])
    
    # 简体到繁体的映射 (根据 GeoJSON 实际内容调整)
    district_map = {
        '中西区': '中西區',
        '湾仔区': '灣仔',      # GeoJSON 中无"區"
        '东区': '東區',
        '南区': '南區',
        '油尖旺区': '油尖旺',  # GeoJSON 中无"區"
        '深水埗区': '深水埗',  # GeoJSON 中无"區"
        '九龙城区': '九龍城',  # GeoJSON 中无"區"
        '黄大仙区': '黃大仙',  # GeoJSON 中无"區"
        '观塘区': '觀塘',      # GeoJSON 中无"區"
        '葵青区': '葵青',      # GeoJSON 中无"區"
        '荃湾区': '荃灣',      # GeoJSON 中无"區"
        '屯门区': '屯門',      # GeoJSON 中无"區"
        '元朗区': '元朗',      # GeoJSON 中无"區"
        '北区': '北區',
        '大埔区': '大埔',      # GeoJSON 中无"區"
        '沙田区': '沙田',      # GeoJSON 中无"區"
        '西贡区': '西貢',      # GeoJSON 中无"區"
        '离岛区': '離島'       # GeoJSON 中无"區"且为繁体
    }
    
    # 获取每个地区最新日期的数据 (防止不同地区更新日期不一致)
    latest_df = df.sort_values('报告日期').groupby('地区名称').last().reset_index()
    
    data = []
    for _, row in latest_df.iterrows():
        district_simp = row['地区名称']
        # 使用映射，如果没有匹配则使用原名
        district_trad = district_map.get(district_simp, district_simp)
        
        data.append({
            'name': district_trad,
            'value': int(row['累计确诊']),
            'active': int(row['现存确诊']),
            'recovered': int(row['累计康复']),
            'deaths': int(row['累计死亡']),
            'cumulative': int(row['累计确诊'])
        })
        
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
