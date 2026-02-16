import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
import platform

# 设置中文字体
system_name = platform.system()
if system_name == "Darwin":  # macOS
    font_path = '/System/Library/Fonts/PingFang.ttc'
    # 如果 PingFang 不存在，尝试其他常用字体
    try:
        font_prop = FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    except:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
elif system_name == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] # Linux 常用

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

file_path = '香港各区疫情数据_20250322.xlsx'

try:
    # 读取 Excel 文件
    df = pd.read_excel(file_path)
    
    # 确保报告日期是 datetime 类型
    df['报告日期'] = pd.to_datetime(df['报告日期'])
    
    # 按日期汇总新增确诊人数 (因为原始数据是分区的，需要汇总得到全港每日数据)
    daily_cases = df.groupby('报告日期')['新增确诊'].sum().reset_index()
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 绘制折线图
    plt.plot(daily_cases['报告日期'], daily_cases['新增确诊'], marker='o', markersize=2, linestyle='-', linewidth=1, color='#1f77b4')
    
    # 设置标题和标签
    plt.title('香港每日新增确诊人数趋势', fontsize=16)
    plt.xlabel('报告日期', fontsize=12)
    plt.ylabel('新增确诊人数', fontsize=12)
    
    # 格式化 x 轴日期显示
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()  # 自动旋转日期标签以防重叠
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    output_file = 'epidemic_trend.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存为: {output_file}")
    
    # 显示前几行汇总数据供检查
    print("\n每日汇总数据前5行:")
    print(daily_cases.head())

except Exception as e:
    print(f"发生错误: {e}")
