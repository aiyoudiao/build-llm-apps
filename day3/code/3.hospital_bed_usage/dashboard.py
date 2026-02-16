import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# 设置页面配置
st.set_page_config(
    page_title="医院病床使用实时监控大屏",
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
    st.title("🏥 医院病床使用实时监控大屏")
    
    file_path = 'hospital_bed_usage_data.xlsx'
    df = load_data(file_path)
    
    if df is None:
        st.warning("数据文件不存在，请检查文件路径。")
        return

    # --- 侧边栏：过滤器 ---
    st.sidebar.header("筛选条件")
    
    # 获取最新的时间戳作为"实时"数据
    latest_timestamp = df['timestamp'].max()
    st.sidebar.info(f"当前数据更新时间: {latest_timestamp}")
    
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

    # 数据过滤
    # 1. 首先只取最新的时间戳数据，模拟"实时"状态
    current_df = df[df['timestamp'] == latest_timestamp]
    
    # 2. 应用侧边栏筛选
    filtered_df = current_df[
        (current_df['hospital_name'].isin(selected_hospitals)) & 
        (current_df['hospital_district'].isin(selected_districts))
    ]
    
    if filtered_df.empty:
        st.warning("没有符合筛选条件的数据。")
        return

    # --- 第一行：关键指标 (KPIs) ---
    st.markdown("### 📊 核心指标概览")
    col1, col2, col3, col4 = st.columns(4)
    
    total_beds = filtered_df['total_beds'].sum()
    occupied_beds = filtered_df['occupied_beds'].sum()
    available_beds = filtered_df['available_beds'].sum()
    avg_occupancy = (occupied_beds / total_beds * 100) if total_beds > 0 else 0
    
    col1.metric("总病床数", f"{total_beds:,.0f}")
    col2.metric("已用病床数", f"{occupied_beds:,.0f}")
    col3.metric("空闲病床数", f"{available_beds:,.0f}", delta_color="normal")
    col4.metric("整体使用率", f"{avg_occupancy:.2f}%", delta=f"{avg_occupancy-85:.1f}% (基准85%)", delta_color="inverse")
    
    st.divider()

    # --- 第二行：图表展示 ---
    
    # 1. 占用率分析
    st.subheader("📈 各医院及科室病床使用率")
    
    # 计算各医院各科室的平均使用率（其实在单时刻就是当前使用率）
    heatmap_data = filtered_df.pivot_table(
        index='hospital_name', 
        columns='department_name', 
        values='occupancy_rate', 
        aggfunc='mean'
    )
    
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
        st.subheader("🏥 各医院空闲病床数分布")
        # 按医院汇总空闲病床
        avail_by_hospital = filtered_df.groupby('hospital_name')['available_beds'].sum().reset_index().sort_values('available_beds', ascending=False)
        
        fig_bar = px.bar(
            avail_by_hospital,
            x='available_beds',
            y='hospital_name',
            orientation='h',
            text='available_beds',
            title="各医院空闲病床排行榜",
            labels={'available_beds': '空闲病床数', 'hospital_name': '医院'},
            color='available_beds',
            color_continuous_scale='Greens'
        )
        fig_bar.update_layout(height=500)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        st.subheader("🗺️ 病床分布情况 (区域 -> 科室)")
        # 使用旭日图展示层级分布: 区域 -> 医院 -> 科室 -> 总床位
        # 为了避免图表过于拥挤，我们展示 区域 -> 科室 的分布，或者 区域 -> 医院
        # 这里展示 区域 -> 医院 -> 科室 的总床位分布
        
        fig_sunburst = px.sunburst(
            filtered_df,
            path=['hospital_district', 'department_name'],
            values='total_beds',
            color='occupancy_rate',
            color_continuous_scale='RdBu_r',
            title="不同区域及科室的病床分布与使用率(颜色)",
            hover_data=['available_beds']
        )
        fig_sunburst.update_layout(height=500)
        st.plotly_chart(fig_sunburst, use_container_width=True)

    # --- 第三行：更多细节 ---
    st.subheader("📋 详细数据明细")
    with st.expander("查看详细数据表"):
        st.dataframe(
            filtered_df[['hospital_name', 'department_name', 'ward_name', 'total_beds', 'occupied_beds', 'available_beds', 'occupancy_rate', 'special_status']],
            use_container_width=True
        )

if __name__ == "__main__":
    main()
