import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform

# 设置中文字体
system_name = platform.system()
if system_name == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
elif system_name == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_bed_usage(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        # 读取数据
        df = pd.read_excel(file_path)
        
        # 按医院和科室分组，并聚合数据
        # 我们需要对 total_beds 和 occupied_beds 求和，然后重新计算使用率
        grouped = df.groupby(['hospital_name', 'department_name'])[['total_beds', 'occupied_beds']].sum().reset_index()
        
        # 计算使用率
        # 注意：这里计算的是该时间段内的平均使用率
        grouped['occupancy_rate'] = (grouped['occupied_beds'] / grouped['total_beds'] * 100).round(2)
        
        print("各医院及科室病床使用率统计（前20行）：")
        print(grouped.head(20))
        
        # 保存统计结果到 Excel
        output_excel = 'hospital_department_bed_usage_summary.xlsx'
        grouped.to_excel(output_excel, index=False)
        print(f"\n统计结果已保存至: {output_excel}")
        
        # 准备可视化数据
        # 使用透视表将数据转换为适合热力图的格式
        pivot_table = grouped.pivot(index='hospital_name', columns='department_name', values='occupancy_rate')
        
        # 绘制热力图
        plt.figure(figsize=(16, 10))  # 增加宽度以适应更多科室
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=.5, cbar_kws={'label': '使用率 (%)'})
        
        plt.title('各医院及科室病床使用率 (%)', fontsize=16)
        plt.xlabel('科室名称', fontsize=12)
        plt.ylabel('医院名称', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图片
        output_file = 'hospital_department_bed_usage.png'
        plt.savefig(output_file, dpi=300)
        print(f"\n图表已保存至: {output_file}")
        
        # 也可以尝试绘制分组条形图作为另一种视图
        plt.figure(figsize=(14, 8))
        sns.barplot(x='hospital_name', y='occupancy_rate', hue='department_name', data=grouped)
        plt.title('各医院及科室病床使用率对比', fontsize=16)
        plt.xlabel('医院名称', fontsize=12)
        plt.ylabel('使用率 (%)', fontsize=12)
        plt.legend(title='科室', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        output_bar_file = 'hospital_department_bed_usage_bar.png'
        plt.savefig(output_bar_file, dpi=300)
        print(f"条形图已保存至: {output_bar_file}")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    file_path = 'hospital_bed_usage_data.xlsx'
    analyze_bed_usage(file_path)
