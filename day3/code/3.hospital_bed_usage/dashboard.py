import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# 设置页面配置
st.set_page_config(
    page_title="医院病床使用监控大屏",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# 加载数据函数
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# 主程序
def main():
    st.title("🏥 医院病床使用监控大屏")
    
    file_path = 'hospital_bed_usage_data.xlsx'
    df = load_data(file_path)
    
    if df is None:
        st.warning("数据文件不存在，请检查文件路径。")
        return

    # --- 侧边栏：配置 ---
    st.sidebar.header("配置选项")
    
    # 1. 视图模式选择
    view_mode = st.sidebar.radio(
        "数据视图模式",
        ["最新实时数据", "历史统计分析 (与Excel一致)"],
        index=1, # 默认选中历史统计，以响应用户需求
        help="实时数据：仅显示最近一次更新的数据。\n历史统计：计算所有历史数据的加权平均使用率，与 Excel 报表逻辑一致。"
    )
    
    # 获取时间信息
    latest_timestamp = df['timestamp'].max()
    min_timestamp = df['timestamp'].min()
    
    if view_mode == "最新实时数据":
        st.sidebar.info(f"当前显示时间: {latest_timestamp}")
        # 过滤数据
        current_df = df[df['timestamp'] == latest_timestamp]
    else:
        st.sidebar.info(f"统计时间范围:\n{min_timestamp} 至\n{latest_timestamp}")
        current_df = df # 使用全量数据

    st.divider()
    
    # 2. 筛选条件
    st.sidebar.subheader("数据筛选")
    
    # 医院筛选
    all_hospitals = sorted(df['hospital_name'].unique())
    selected_hospitals = st.sidebar.multiselect(
        "选择医院",
        all_hospitals,
        default=all_hospitals
    )
    
    # 区域筛选
    all_districts = sorted(df['hospital_district'].unique())
    selected_districts = st.sidebar.multiselect(
        "选择区域",
        all_districts,
        default=all_districts
    )

    # 应用筛选
    filtered_df = current_df[
        (current_df['hospital_name'].isin(selected_hospitals)) & 
        (current_df['hospital_district'].isin(selected_districts))
    ]
    
    if filtered_df.empty:
        st.warning("没有符合筛选条件的数据。")
        return

    # --- 数据聚合逻辑 ---
    # 根据视图模式，计算用于展示的数据
    if view_mode == "最新实时数据":
        # 实时模式下，filtered_df 已经是单时刻快照
        # KPI 计算
        total_beds_kpi = filtered_df['total_beds'].sum()
        occupied_beds_kpi = filtered_df['occupied_beds'].sum()
        available_beds_kpi = filtered_df['available_beds'].sum()
        
        # 图表数据准备 (直接使用，不需要再聚合，或者按需简单聚合)
        chart_df = filtered_df
        
        # 热力图数据
        heatmap_data = chart_df.pivot_table(
            index='hospital_name', 
            columns='department_name', 
            values='occupancy_rate', 
            aggfunc='mean' # 单时刻 mean 等于本身
        )
        
    else:
        # 历史统计模式下，需要进行聚合计算
        # KPI 计算 (使用平均值来代表"常态")
        # 注意：简单 sum 会导致数字巨大且无物理意义。
        # 我们计算平均每时刻的床位数
        num_timestamps = filtered_df['timestamp'].nunique()
        total_beds_kpi = filtered_df['total_beds'].sum() / num_timestamps
        occupied_beds_kpi = filtered_df['occupied_beds'].sum() / num_timestamps
        available_beds_kpi = filtered_df['available_beds'].sum() / num_timestamps
        
        # 核心：计算加权平均使用率 (与 Excel 逻辑一致)
        # 先按维度聚合 sum
        grouped = filtered_df.groupby(['hospital_name', 'department_name', 'hospital_district'])[['total_beds', 'occupied_beds', 'available_beds']].sum().reset_index()
        # 再计算率
        grouped['occupancy_rate'] = (grouped['occupied_beds'] / grouped['total_beds'] * 100).round(2)
        
        # 为了其他图表，我们也需要保留一些维度
        chart_df = grouped
        
        # 热力图数据
        heatmap_data = chart_df.pivot(
            index='hospital_name', 
            columns='department_name', 
            values='occupancy_rate'
        )

    # 计算整体使用率
    avg_occupancy_kpi = (occupied_beds_kpi / total_beds_kpi * 100) if total_beds_kpi > 0 else 0

    # --- 第一行：关键指标 (KPIs) ---
    st.markdown("### 📊 核心指标概览")
    col1, col2, col3, col4 = st.columns(4)
    
    kpi_suffix = " (平均)" if view_mode == "历史统计分析 (与Excel一致)" else ""
    
    col1.metric(f"总病床数{kpi_suffix}", f"{total_beds_kpi:,.0f}")
    col2.metric(f"已用病床数{kpi_suffix}", f"{occupied_beds_kpi:,.0f}")
    col3.metric(f"空闲病床数{kpi_suffix}", f"{available_beds_kpi:,.0f}", delta_color="normal")
    col4.metric(f"整体使用率{kpi_suffix}", f"{avg_occupancy_kpi:.2f}%", delta=f"{avg_occupancy_kpi-85:.1f}% (基准85%)", delta_color="inverse")
    
    st.divider()

    # --- 第二行：图表展示 ---
    
    # 1. 占用率分析
    st.subheader("📈 各医院及科室病床使用率")
    
    fig_heatmap = px.imshow(
        heatmap_data,
        labels=dict(x="科室", y="医院", color="使用率(%)"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="YlOrRd",
        text_auto=".1f",
        aspect="auto"
    )
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader(f"🏥 各医院空闲病床数分布{kpi_suffix}")
        
        if view_mode == "最新实时数据":
             avail_by_hospital = filtered_df.groupby('hospital_name')['available_beds'].sum().reset_index()
        else:
             # 统计模式下，chart_df 已经是聚合后的数据 (Sum of all time)，所以需要除以时间点数还原为"平均空闲"
             # 或者直接使用 chart_df 中的 available_beds (这是 Sum)，并在图表中说明是累计或者重新计算平均
             # 为了直观，我们重新计算平均空闲
             avail_by_hospital = filtered_df.groupby('hospital_name')['available_beds'].sum().reset_index()
             avail_by_hospital['available_beds'] = avail_by_hospital['available_beds'] / num_timestamps
        
        avail_by_hospital = avail_by_hospital.sort_values('available_beds', ascending=False)
        
        fig_bar = px.bar(
            avail_by_hospital,
            x='available_beds',
            y='hospital_name',
            orientation='h',
            text_auto='.0f',
            title="各医院空闲病床排行榜",
            labels={'available_beds': f'空闲病床数{kpi_suffix}', 'hospital_name': '医院'},
            color='available_beds',
            color_continuous_scale='Greens'
        )
        fig_bar.update_layout(height=500)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        st.subheader("🗺️ 病床分布情况 (区域 -> 科室)")
        
        # 旭日图数据准备
        if view_mode == "最新实时数据":
            sunburst_df = filtered_df
        else:
            # 统计模式下，使用聚合后的 chart_df (Total Sum)，展示比例关系是没问题的
            sunburst_df = chart_df
        
        fig_sunburst = px.sunburst(
            sunburst_df,
            path=['hospital_district', 'hospital_name', 'department_name'],
            values='total_beds',
            color='occupancy_rate',
            color_continuous_scale='RdBu_r',
            title="区域-医院-科室 床位分布与使用率(颜色)",
            hover_data=['occupied_beds']
        )
        fig_sunburst.update_layout(height=500)
        st.plotly_chart(fig_sunburst, use_container_width=True)

    # --- 第三行：更多细节 ---
    st.subheader("📋 详细数据明细")
    with st.expander("查看详细数据表"):
        if view_mode == "最新实时数据":
             display_df = filtered_df[['hospital_name', 'department_name', 'ward_name', 'total_beds', 'occupied_beds', 'available_beds', 'occupancy_rate', 'special_status']]
        else:
             display_df = chart_df[['hospital_name', 'department_name', 'total_beds', 'occupied_beds', 'available_beds', 'occupancy_rate']]
             st.info("注：统计模式下显示的是汇总/平均数据，不显示具体病房(ward)维度的细节。")
             
        st.dataframe(
            display_df,
            use_container_width=True
        )

if __name__ == "__main__":
    main()
