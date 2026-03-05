"""
================================================================================
文件名: 门票数据智能分析与可视化助手.py
功能描述: 
    本程序是一个增强版的智能数据分析助手。
    它不仅支持通过自然语言对话查询 MySQL 数据库中的门票订单数据，
    还具备【自动数据可视化】功能。当查询结果适合展示时，系统会自动生成
    柱状图并嵌入到对话回复中，实现“ queried data + visual chart ”的一站式输出。

核心能力:
    1. Text-to-SQL: 理解用户意图，自动生成复杂的 SQL 查询语句。
    2. 自动绘图: 智能识别数据维度，自动绘制柱状图 (Bar Chart)。
    3. 混合输出: 同时返回 Markdown 数据表格和图片链接，并在 Web 界面渲染。

核心工作流程:
    1. [初始化] 加载 System Prompt，定义表结构及“必须原样输出图片”的指令。
    2. [用户提问] 用户在 Web UI 输入自然语言问题 (如：“各省份销量排名”)。
    3. [意图识别] LLM 分析意图，生成对应的 SQL 语句。
    4. [工具执行 (exc_sql)]:
       a. 连接数据库执行 SQL，获取 Pandas DataFrame。
       b. 将数据前 10 行转换为 Markdown 表格。
       c. [自动绘图] 
          - 自动推断 X 轴 (分类列) 和 Y 轴 (数值列)。
          - 使用 Matplotlib 绘制柱状图，处理多系列数据宽度。
          - 保存图片至本地 'image_show' 目录。
       d. 组装返回字符串："[Markdown 表格] \n\n ![图片](相对路径)"。
    5. [结果渲染] LLM 接收工具返回内容，按指令原样输出，Web 前端解析并显示图表。

依赖环境:
    - Python 3.8+
    - qwen_agent, dashscope, pandas, sqlalchemy, mysql-connector-python
    - matplotlib, pillow (用于图片处理)
================================================================================
"""

import os
import json
import time
import io
from typing import Optional
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# 第三方库导入
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool
from dotenv import load_dotenv

load_dotenv()

# ================= 配置区域 =================

# 定义资源文件根目录
ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

# 配置 DashScope API
# 优先从环境变量获取，确保密钥安全
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
dashscope.timeout = 30  # 设置 API 超时时间

# 数据库配置
DB_USER = os.getenv('DB_USER', '')
DB_PASS = os.getenv('DB_PASSWORD', '')
DB_HOST = os.getenv('DB_HOST', '')
DB_PORT = os.getenv('DB_PORT', '')
DB_NAME = os.getenv('DB_NAME', '')

# 图片保存目录配置
IMG_SAVE_DIR = Path(os.path.dirname(__file__)) / 'image_show'
# 确保目录存在
IMG_SAVE_DIR.mkdir(exist_ok=True)

# ================= 核心逻辑定义 =================

# ====== 1. 系统提示词 (System Prompt) ======
system_prompt = """
我是门票助手，专门负责分析门票订单数据并自动生成可视化图表。
我将基于以下 MySQL 表结构编写 SQL 进行查询：

-- 门票订单表 (tkt_orders)
CREATE TABLE tkt_orders (
    order_time DATETIME,             -- 订单日期
    account_id INT,                  -- 预定用户 ID
    gov_id VARCHAR(18),              -- 商品使用人 ID (身份证号)
    gender VARCHAR(10),              -- 使用人性别
    age INT,                         -- 年龄
    province VARCHAR(30),            -- 使用人省份
    SKU VARCHAR(100),                -- 商品 SKU 名
    product_serial_no VARCHAR(30),   -- 商品 ID
    eco_main_order_id VARCHAR(20),   -- 订单 ID
    sales_channel VARCHAR(20),       -- 销售渠道
    status VARCHAR(30),              -- 商品状态
    order_value DECIMAL(10,2),       -- 订单金额
    quantity INT                     -- 商品数量
);

业务规则说明:
1. 一日门票 SKU 特征: 包含 'Universal Studios Beijing One-Day%'
2. 二日门票 SKU 特征: 包含 'USB%' (注意区分 One-Day 和 USB 前缀)
3. 统计示例:
   - 一日票销量: SUM(CASE WHEN SKU LIKE 'Universal Studios Beijing One-Day%' THEN quantity ELSE 0 END)
   - 二日票销量: SUM(CASE WHEN SKU LIKE 'USB%' AND SKU NOT LIKE 'One-Day%' THEN quantity ELSE 0 END)

重要指令:
每当 exc_sql 工具返回 markdown 表格和图片时，你【必须】原样输出工具返回的全部内容。
不要省略图片标记，不要只总结表格。确保用户能直接看到生成的柱状图和数据明细。
"""

# ====== 2. 工具定义 (Function Description for LLM) ======
functions_desc = [
    {
        "name": "exc_sql",
        "description": "执行 SQL 查询，自动返回数据表格和可视化的柱状图",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_input": {
                    "type": "string",
                    "description": "需要执行的标准 SQL 查询语句 (SELECT only)",
                },
                "database": {
                    "type": "string",
                    "description": "目标数据库名称，默认为 ubr",
                    "default": "ubr"
                }
            },
            "required": ["sql_input"],
        },
    },
]

# ====== 3. 工具实现类 (ExcSQLTool) ======
@register_tool('exc_sql')
class ExcSQLTool(BaseTool):
    """
    SQL 执行与自动可视化工具。
    功能：执行 SQL -> 获取数据 -> 生成 Markdown 表格 -> 自动推断维度绘制柱状图 -> 返回组合结果。
    """
    description = '对于生成的 SQL，进行 SQL 查询，并自动可视化生成柱状图'
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
        """
        执行 SQL 并生成可视化图表
        :param params: JSON 格式的参数字符串
        :return: 包含 Markdown 表格和图片链接的字符串
        """
        try:
            args = json.loads(params)
            sql_input = args['sql_input']
            
            # 1. 建立数据库连接
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
            
            # 2. 执行查询
            df = pd.read_sql(sql_input, engine)
            
            if df.empty:
                return "查询结果为空，未生成图表。"

            # 3. 生成数据表格 (限制前 10 行以防过长)
            md_table = df.head(10).to_markdown(index=False)
            
            # 4. 自动推断绘图维度
            # X 轴候选：优先选择非数值型列 (分类数据)，如果没有则选第一列
            x_candidates = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if not x_candidates:
                x_candidates = [df.columns[0]]
            x_col = x_candidates[0]
            
            # Y 轴候选：所有数值型列 (度量数据)
            y_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if not y_cols:
                return f"{md_table}\n\n (无合适数值列，跳过绘图)"

            # 5. 绘制柱状图
            plt.figure(figsize=(10, 6))
            num_y = len(y_cols)
            bar_width = 0.8 / num_y if num_y > 1 else 0.6  # 动态调整柱宽
            x_labels = df[x_col].astype(str)
            x_pos = range(len(df))
            
            # 处理多系列数据 (分组柱状图)
            for idx, y_col in enumerate(y_cols):
                offset = (idx - num_y / 2 + 0.5) * bar_width if num_y > 1 else 0
                plt.bar([p + offset for p in x_pos], df[y_col], width=bar_width, label=y_col)
            
            # 设置图表样式
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(', '.join(y_cols), fontsize=12)
            plt.title(f"{' & '.join(y_cols)} by {x_col}", fontsize=14, fontweight='bold')
            
            # 旋转 X 轴标签以防重叠
            plt.xticks(x_pos, x_labels, rotation=45, ha='right')
            plt.legend(loc='best')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # 6. 保存图片
            timestamp = int(time.time() * 1000)
            filename = f'chart_{timestamp}.png'
            save_path = IMG_SAVE_DIR / filename
            
            plt.savefig(save_path, dpi=150)  # 提高分辨率
            plt.close()  # 释放内存
            
            # 构建相对路径供 Markdown 引用 (假设 WebUI 以脚本目录为根)
            img_rel_path = f'image_show/{filename}'
            img_md = f'![数据可视化图表]({img_rel_path})'
            
            # 7. 返回组合结果
            return f"{md_table}\n\n{img_md}"
            
        except Exception as e:
            error_msg = f"SQL 执行或可视化出错: {str(e)}"
            print(f"[Error] {error_msg}")
            return error_msg

# ================= 服务初始化与交互模式 =================

def init_agent_service():
    """
    初始化门票助手 Agent 服务
    """
    llm_cfg = {
        'model': 'qwen-turbo-2025-04-28',  # 指定模型版本
        'timeout': 30,
        'retry_count': 3,
    }
    
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='门票可视化助手',
            description='门票查询、订单分析及自动图表生成',
            system_message=system_prompt,
            function_list=['exc_sql'],
        )
        print("✅ 助手初始化成功！已启用自动绘图功能。")
        return bot
    except Exception as e:
        print(f"❌ 助手初始化失败: {str(e)}")
        raise

def app_tui():
    """终端交互模式 (TUI)"""
    try:
        bot = init_agent_service()
        messages = []
        
        print("\n--- 进入命令行交互模式 (输入 'quit' 退出) ---")
        print("注：命令行模式下图片链接将显示为文本路径，请在文件夹中查看 image_show 目录。")
        
        while True:
            try:
                query = input('\n👤 用户提问: ').strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if not query:
                    continue
                
                file_url = input('📎 附件 URL (直接回车跳过): ').strip()
                
                if not file_url:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file_url}]})

                print("🤖 正在思考、查询并绘制图表...")
                
                response_content = []
                for response in bot.run(messages):
                    response_content = response 
                
                if isinstance(response_content, list):
                    for msg in response_content:
                        if msg.get('role') == 'assistant':
                            content = msg.get('content', '')
                            print(f"\n💡 助手回答:\n{content}")
                            # 提示用户查看图片
                            if '![数据可视化图表]' in content:
                                print(f"\n📊 [提示] 图表已保存至当前目录下的 'image_show' 文件夹中。")
                            messages.append(msg)
                            break
                else:
                    print(f"\n💡 助手回答:\n{response_content}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️ 出错: {str(e)}")
                
    except Exception as e:
        print(f"❌ 启动终端模式失败: {str(e)}")

def app_gui():
    """图形界面模式 (Web UI)"""
    try:
        print("🚀 正在启动 Web 图形界面...")
        bot = init_agent_service()
        
        chatbot_config = {
            'prompt.suggestions': [
                '2023年4-6月一日票和二日票的周销量对比，请画图。',
                '2023年7月不同省份的入园人数统计，生成柱状图。',
                '查看2023年国庆期间各销售渠道的订单金额排名并可视化。',
                '分析不同年龄段用户的购票数量分布。'
            ]
        }
        
        print("✅ Web 界面准备就绪。")
        print(f"📂 图表将自动保存至: {IMG_SAVE_DIR.resolve()}")
        print("🌐 请在浏览器中打开显示的地址开始对话。")
        
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
        
    except Exception as e:
        print(f"❌ 启动 Web 界面失败: {str(e)}")
        print("💡 请检查 API Key、数据库连接及文件写入权限。")

# ================= 程序入口 =================

if __name__ == '__main__':
    if not os.getenv('DASHSCOPE_API_KEY'):
        print("⚠️ 警告: 未检测到 DASHSCOPE_API_KEY 环境变量。")
    
    # 默认启动图形界面
    app_gui()
