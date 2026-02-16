import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
import seaborn as sns
import platform

# 设置中文字体
system_name = platform.system()
if system_name == "Darwin":  # macOS
    font_path = '/System/Library/Fonts/PingFang.ttc'
    try:
        font_prop = FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    except:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
elif system_name == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']

plt.rcParams['axes.unicode_minus'] = False

file_path = '香港各区疫情数据_20250322.xlsx'

try:
    # 读取 Excel 文件
    df = pd.read_excel(file_path)
    
    # 确保报告日期是 datetime 类型
    df['报告日期'] = pd.to_datetime(df['报告日期'])
    
    # 创建图表
    plt.figure(figsize=(14, 8))  # 增加图表宽度以容纳图例
    
    # 使用 seaborn 绘制多条折线图，hue 参数指定按地区区分颜色
    sns.lineplot(data=df, x='报告日期', y='新增确诊', hue='地区名称', marker='o', markersize=3, linewidth=1.5)
    
    # 设置标题和标签
    plt.title('香港各区每日新增确诊人数趋势', fontsize=18)
    plt.xlabel('报告日期', fontsize=14)
    plt.ylabel('新增确诊人数', fontsize=14)
    
    # 格式化 x 轴日期显示
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 调整图例位置到图表外部
    plt.legend(title='地区名称', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # 保存图表，bbox_inches='tight' 确保图例不会被裁剪
    output_file = 'epidemic_trend_by_district.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存为: {output_file}")
    
    # 打印不同地区的列表供确认
    districts = df['地区名称'].unique()
    print(f"包含的地区 ({len(districts)}个): {', '.join(districts)}")

except Exception as e:
    print(f"发生错误: {e}")
