"""
================================================================================
文件名: 门票数据智能透视与可视化助手.py
功能描述: 
    本程序是一个高级智能数据分析助手，专为门票业务设计。
    它不仅支持通过自然语言对话查询 MySQL 数据库，还能根据查询结果的数据结构，
    【智能选择】最合适的图表类型（普通柱状图或堆积柱状图）进行自动可视化。

核心能力:
    1. Text-to-SQL: 理解复杂业务意图，生成包含 CASE WHEN 等逻辑的 SQL。
    2. 智能绘图引擎:
       - 自动识别多维度数据 (如：时间 + 票种)，自动生成【堆积柱状图】。
       - 自动识别单维度多指标数据，生成【分组/堆叠柱状图】。
    3. 中文与特殊字符适配: 完美解决 Matplotlib 中文乱码及 SQL 特殊符号 (%) 转义问题。
    4. 混合输出: 同时返回 Markdown 数据明细表和可视化图表。

核心工作流程:
    1. [初始化] 加载 System Prompt，配置中文字体，建立数据库连接池。
    2. [用户提问] 用户输入自然语言 (例：“对比各月一日票和二日票销量”)。
    3. [意图识别] LLM 生成 SQL (通常包含 GROUP BY 多个字段)。
    4. [工具执行 (exc_sql)]:
       a. 使用 SQLAlchemy text() 安全执行 SQL，获取 DataFrame。
       b. 调用 generate_chart_png 进行智能绘图：
          - 检测非数值列数量。
          - 若存在多个非数值列 (如：月份、票种)，执行 pivot_table 数据透视。
          - 绘制堆积柱状图 (Stacked Bar)，展示总量与构成。
          - 对标签中的 %, {} 等特殊字符进行转义，防止 Matplotlib 报错。
       c. 保存图片至本地 'image_show' 目录。
       d. 组装返回内容："[数据表格] \n\n ![图表](路径)"。
    5. [结果渲染] WebUI 解析 Markdown，直接展示带中文标签的统计图表。

依赖环境:
    - Python 3.8+
    - qwen_agent, dashscope, pandas, sqlalchemy, pymysql
    - matplotlib, numpy, tabulate (用于 markdown 转换)
================================================================================
"""

import os
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

# Qwen Agent 相关导入
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool
from dotenv import load_dotenv

load_dotenv()

# ================= 配置区域 =================

# 定义资源文件根目录
ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

# --- DashScope 配置 ---
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
dashscope.timeout = 30

# 数据库配置
DB_USER = os.getenv('DB_USER', '')
DB_PASS = os.getenv('DB_PASSWORD', '')
DB_HOST = os.getenv('DB_HOST', '')
DB_PORT = os.getenv('DB_PORT', '')
DB_NAME = os.getenv('DB_NAME', '')


# --- Matplotlib 中文与符号配置 ---
# 设置字体列表，优先尝试黑体、微软雅黑，确保中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
# 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# --- 图片保存路径配置 ---
IMG_SAVE_DIR = Path(os.path.dirname(__file__)) / 'image_show'
IMG_SAVE_DIR.mkdir(exist_ok=True)  # 确保目录存在

# ================= 核心逻辑定义 =================

# ====== 1. 系统提示词 (System Prompt) ======
system_prompt = """
我是门票助手，专门负责分析门票订单数据并生成可视化报表。
基于以下 MySQL 表结构 (tkt_orders) 进行查询：
[order_time, account_id, gov_id, gender, age, province, SKU, product_serial_no, 
 eco_main_order_id, sales_channel, status, order_value, quantity]

业务规则:
1. 一日票 SKU: LIKE 'Universal Studios Beijing One-Day%'
2. 二日票 SKU: LIKE 'USB%' (需排除 One-Day)
3. 统计逻辑示例: SUM(CASE WHEN ... THEN quantity ELSE 0 END)

重要指令:
当 exc_sql 工具返回 Markdown 表格和图片链接时，你【必须】原样输出所有内容。
不要省略图片标记，不要仅总结文字。确保用户能直接看到带有中文标签的统计图表。
"""

# ====== 2. 工具定义 ======
functions_desc = [
    {
        "name": "exc_sql",
        "description": "执行 SQL 查询，自动分析数据结构并生成合适的柱状图（含堆积图）",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_input": {
                    "type": "string",
                    "description": "需要执行的标准 SQL 查询语句",
                },
                "database": {
                    "type": "string",
                    "description": "目标数据库名称",
                    "default": "ubr"
                }
            },
            "required": ["sql_input"],
        },
    },
]

# ====== 3. 通用可视化函数 (核心增强) ======
def generate_chart_png(df: pd.DataFrame, save_path: str) -> None:
    """
    智能绘图函数：根据 DataFrame 的结构自动选择绘制普通柱状图或堆积柱状图。
    
    逻辑判断:
    - 如果存在 >1 个非数值列 (例如：['月份', '票种']) -> 透视数据 -> 绘制堆积柱状图。
    - 如果只有 1 个非数值列 (例如：['月份']) -> 直接绘制多系列柱状图 (堆叠模式)。
    
    安全措施:
    - 对所有文本标签进行转义 (% -> %%, { -> {{)，防止 Matplotlib 格式化报错。
    """
    if df.empty:
        return

    columns = df.columns.tolist()
    # 分离数值列和非数值列 (Object 类型)
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    # 确保第一列作为 X 轴 (通常是时间或分类)
    x_col = columns[0]
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- 场景 A: 多维度数据 -> 堆积柱状图 (Stacked Bar) ---
    # 例如：X=月份, Column=票种, Value=销量
    if len(object_cols) > 1 and x_col in object_cols:
        # 移除 X 轴列，剩下的非数值列作为堆叠维度
        stack_cols = [c for c in object_cols if c != x_col]
        
        # 数据透视：Index=X 轴, Columns=堆叠维度, Values=数值列
        # fill_value=0 防止空值导致绘图断裂
        pivot_df = df.pivot_table(index=x_col, columns=stack_cols, values=num_cols, fill_value=0)
        
        bottoms = None
        # 遍历透视后的列 (可能是 MultiIndex，需展平处理)
        # 这里简化处理，假设 pivot 后 columns 是直接的堆叠类别
        for col in pivot_df.columns:
            label_str = str(col)
            # 【关键】转义特殊字符，防止 Matplotlib 抛出 ValueError
            safe_label = label_str.replace('%', '%%').replace('{', '{{').replace('}', '}}')
            
            data_series = pivot_df[col]
            ax.bar(pivot_df.index.astype(str), data_series, bottom=bottoms, label=safe_label)
            
            # 更新底部高度，实现堆叠效果
            if bottoms is None:
                bottoms = data_series.values
            else:
                bottoms += data_series.values

    # --- 场景 B: 单维度多指标 -> 堆叠柱状图 ---
    # 例如：X=月份, Y=[销量，金额] (虽然通常不堆叠不同单位，但此处逻辑保持堆叠以展示总量)
    else:
        x_labels = df[x_col].astype(str)
        x_pos = np.arange(len(df))
        bottom = np.zeros(len(df))
        
        for col in num_cols:
            label_str = str(col)
            safe_label = label_str.replace('%', '%%').replace('{', '{{').replace('}', '}}')
            
            ax.bar(x_pos, df[col], bottom=bottom, label=safe_label)
            bottom += df[col].values
        
        # 设置 X 轴刻度
        safe_xtick_labels = [str(val).replace('%', '%%').replace('{', '{{').replace('}', '}}') for val in x_labels]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(safe_xtick_labels, rotation=45, ha='right')

    # --- 通用图表美化 ---
    ax.legend(loc='best', frameon=True)
    ax.set_title("门票销售统计可视化", fontsize=14, fontweight='bold')
    
    # 安全处理轴标签
    xlabel_str = str(x_col)
    safe_xlabel = xlabel_str.replace('%', '%%').replace('{', '{{').replace('}', '}}')
    ax.set_xlabel(safe_xlabel)
    ax.set_ylabel("数量/金额")
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=150)
    plt.close(fig)  # 释放内存

# ====== 4. 工具实现类 (ExcSQLTool) ======
@register_tool('exc_sql')
class ExcSQLTool(BaseTool):
    """
    SQL 执行与智能可视化工具。
    集成数据查询、自动透视分析、图表生成于一体。
    """
    description = '执行 SQL 查询，并根据数据维度自动绘制堆积柱状图或普通柱状图'
    parameters = [{
        'name': 'sql_input',
        'type': 'string',
        'description': '生成的 SQL 语句',
        'required': True
    }, {
        'name': 'database',
        'type': 'string',
        'description': '数据库名称',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        try:
            args = json.loads(params)
            sql_input = args['sql_input']
            
            print(f"[SQL] 执行查询: {sql_input}")

            # 构建数据库连接字符串
            # 格式: mysql+mysqlconnector://user:pass@host:port/db?charset=utf8mb4
            conn_str = (
                f'mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
                f'?charset=utf8mb4'
            )
            engine = create_engine(
                conn_str,
                connect_args={'connect_timeout': 10},
                pool_size=10,
                max_overflow=20
            )
            
            # 【关键】使用 text() 包装 SQL，防止 % 符号被误解析为参数占位符
            df = pd.read_sql(text(sql_input), engine)
            
            if df.empty:
                return "查询结果为空，未生成图表。"

            # 生成表格预览
            md_table = df.head(10).to_markdown(index=False)
            
            # 生成唯一文件名并绘图
            timestamp = int(time.time() * 1000)
            filename = f'chart_{timestamp}.png'
            save_path = IMG_SAVE_DIR / filename
            
            # 调用智能绘图函数
            generate_chart_png(df, str(save_path))
            
            # 构建相对路径 (WebUI 通常以脚本运行目录为根)
            img_rel_path = f'image_show/{filename}'
            img_md = f'![销售统计图表]({img_rel_path})'
            
            return f"{md_table}\n\n{img_md}"
            
        except Exception as e:
            error_msg = f"执行出错: {str(e)}"
            print(f"[Error] {error_msg}")
            return error_msg

# ================= 服务初始化与交互模式 =================

def init_agent_service():
    """初始化 Agent 服务"""
    llm_cfg = {
        'model': 'qwen-max',
        'timeout': 30,
        'retry_count': 3,
    }
    
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='门票透视助手',
            description='支持自动堆积图生成的门票数据分析专家',
            system_message=system_prompt,
            function_list=['exc_sql'],
        )
        print("✅ 助手初始化成功！已启用智能堆积图引擎。")
        return bot
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        raise

def app_tui():
    """终端交互模式"""
    try:
        bot = init_agent_service()
        messages = []
        print("\n--- 命令行模式 (输入 'quit' 退出) ---")
        print("提示：图片将保存在 ./image_show 目录")
        
        while True:
            query = input('\n👤 提问: ').strip()
            if query.lower() in ['quit', 'exit']: break
            if not query: continue
            
            messages.append({'role': 'user', 'content': query})
            print("🤖 分析中...")
            
            response_content = []
            for response in bot.run(messages):
                response_content = response
            
            if isinstance(response_content, list):
                for msg in response_content:
                    if msg.get('role') == 'assistant':
                        print(f"\n💡 回答:\n{msg.get('content', '')}")
                        messages.append(msg)
                        break
    except Exception as e:
        print(f"错误: {e}")

def app_gui():
    """图形界面模式"""
    try:
        print("🚀 启动 Web 界面...")
        bot = init_agent_service()
        
        chatbot_config = {
            'prompt.suggestions': [
                '统计 4-6 月各周的一日票和二日票销量，用堆积图展示。',
                '分析不同省份在暑期的入园人数分布。',
                '对比各销售渠道在国庆期间的订单金额。'
            ]
        }
        
        print(f"✅ 服务就绪。图表保存目录: {IMG_SAVE_DIR.resolve()}")
        WebUI(bot, chatbot_config=chatbot_config).run()
        
    except Exception as e:
        print(f"❌ Web 启动失败: {e}")

if __name__ == '__main__':
    if not os.getenv('DASHSCOPE_API_KEY'):
        print("⚠️ 警告: 缺少 DASHSCOPE_API_KEY 环境变量")
    app_gui()
