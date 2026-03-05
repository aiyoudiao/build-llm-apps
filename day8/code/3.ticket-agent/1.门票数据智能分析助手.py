"""
================================================================================
文件名: 门票数据智能分析助手.py
功能描述: 
    本程序构建了一个基于大语言模型 (LLM) 的智能数据分析助手。
    它允许用户通过自然语言对话，查询和分析 MySQL 数据库中的"门票订单表"数据。
    核心能力包括：意图识别、自动 SQL 生成、安全执行查询、结果可视化展示。

适用场景:
    - 业务人员查询每日/每月销量
    - 分析不同省份/渠道的用户画像
    - 统计特定 SKU (如一日票/二日票) 的销售表现

核心工作流程:
    1. [初始化] 加载系统提示词 (System Prompt)，定义数据表结构和业务规则。
    2. [注册工具] 注册 `exc_sql` 工具，用于安全地连接数据库并执行 SQL。
    3. [构建 Agent] 实例化 Qwen-Agent，绑定 LLM 模型与自定义工具。
    4. [用户交互] 
       - 用户输入自然语言问题 (Web UI 或 命令行)。
       - Agent 理解意图，生成对应的 SQL 语句。
       - 调用 `exc_sql` 工具执行 SQL，获取 Pandas DataFrame 结果。
       - Agent 根据数据结果生成自然语言回答。
    5. [展示结果] 在界面中返回统计图表描述或数据表格。

依赖环境:
    - Python 3.8+
    - qwen_agent, dashscope, pandas, sqlalchemy, mysql-connector-python
================================================================================
"""

import os
import json
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine

# 第三方库导入
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool
from dotenv import load_dotenv

load_dotenv()
# ================= 配置区域 =================

# 定义资源文件根目录 (预留扩展用)
ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

# 配置 DashScope API
# 优先从环境变量获取，若无则使用默认空值 (需在运行前 export DASHSCOPE_API_KEY)
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
dashscope.timeout = 30  # 设置 API 超时时间

# 数据库配置
DB_USER = os.getenv('DB_USER', '')
DB_PASS = os.getenv('DB_PASSWORD', '')
DB_HOST = os.getenv('DB_HOST', '')
DB_PORT = os.getenv('DB_PORT', '')
DB_NAME = os.getenv('DB_NAME', '')

# ================= 核心逻辑定义 =================

# ====== 1. 系统提示词 (System Prompt) ======
# 定义助手的角色、数据库 schema 以及特定的业务统计逻辑
system_prompt = """
我是门票助手，专门负责分析门票订单数据。
我将基于以下 MySQL 8.0 表结构编写 SQL 进行查询：

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

请根据用户问题，生成准确的 SQL 语句，并调用 exc_sql 工具获取数据后回答。
"""

# ====== 2. 工具定义 (Function Description for LLM) ======
functions_desc = [
    {
        "name": "exc_sql",
        "description": "执行生成的 SQL 语句以查询门票订单数据库",
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
    SQL 执行工具：负责建立数据库连接，执行 LLM 生成的 SQL，并返回格式化结果。
    包含安全限制：仅返回前 10 行数据以防输出过长。
    """
    description = '对于生成的 SQL，进行 SQL 查询并返回结果摘要'
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
        执行 SQL 查询的核心方法
        :param params: JSON 格式的参数字符串
        :return: Markdown 格式的查询结果或错误信息
        """
        try:
            args = json.loads(params)
            sql_input = args['sql_input']
            database = args.get('database', DB_NAME)
            
            # 构建数据库连接字符串
            # 格式: mysql+mysqlconnector://user:pass@host:port/db?charset=utf8mb4
            conn_str = (
                f'mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{database}'
                f'?charset=utf8mb4'
            )
            
            engine = create_engine(
                conn_str,
                connect_args={'connect_timeout': 10}, 
                pool_size=10, 
                max_overflow=20
            )
            
            # 执行查询
            df = pd.read_sql(sql_input, engine)
            
            # 安全截断：只返回前 10 行，避免大数据量导致上下文溢出
            result_df = df.head(10)
            
            # 转换为 Markdown 表格返回
            return result_df.to_markdown(index=False)
            
        except Exception as e:
            # 捕获所有异常并返回友好提示
            error_msg = f"SQL 执行出错: {str(e)}"
            print(f"[Error] {error_msg}") # 同时在控制台打印以便调试
            return error_msg

# ================= 服务初始化与交互模式 =================

def init_agent_service():
    """
    初始化门票助手 Agent 服务
    配置 LLM 模型参数、系统提示词及可用工具列表
    """
    llm_cfg = {
        'model': 'qwen-turbo-latest',  # 使用通义千问 Turbo 模型
        'timeout': 30,
        'retry_count': 3,              # 失败重试次数
    }
    
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='门票助手',
            description='门票查询与订单数据分析专家',
            system_message=system_prompt,
            function_list=['exc_sql'],  # 绑定已注册的 exc_sql 工具
        )
        print("✅ 助手初始化成功！ ready to serve.")
        return bot
    except Exception as e:
        print(f"❌ 助手初始化失败: {str(e)}")
        raise

def app_tui():
    """
    终端交互模式 (TUI)
    提供命令行界面，支持连续对话和简单的文件引用模拟
    """
    try:
        bot = init_agent_service()
        messages = []  # 维护对话历史
        
        print("\n--- 进入命令行交互模式 (输入 'quit' 退出) ---")
        
        while True:
            try:
                query = input('\n👤 用户提问: ').strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("再见！")
                    break
                    
                if not query:
                    print('⚠️ 问题不能为空，请重新输入。')
                    continue
                
                # 模拟文件输入 (当前版本主要处理文本)
                file_url = input('📎 附件 URL (直接回车跳过): ').strip()
                
                # 构建消息体
                if not file_url:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file_url}]})

                print("🤖 正在思考并查询数据...")
                
                # 流式处理响应
                response_content = []
                for response in bot.run(messages):
                    # 这里简化处理，实际可能包含多个 chunk
                    response_content = response 
                
                # 打印最终回复
                if isinstance(response_content, list):
                    for msg in response_content:
                        if msg.get('role') == 'assistant':
                            print(f"\n💡 助手回答:\n{msg.get('content', '')}")
                            messages.append(msg)
                            break
                else:
                    print(f"\n💡 助手回答:\n{response_content}")
                    
            except KeyboardInterrupt:
                print("\n中断退出。")
                break
            except Exception as e:
                print(f"⚠️ 处理请求时出错: {str(e)}")
                
    except Exception as e:
        print(f"❌ 启动终端模式失败: {str(e)}")

def app_gui():
    """
    图形界面模式 (Web UI)
    启动基于浏览器的聊天界面，提供预设问题建议
    """
    try:
        print("🚀 正在启动 Web 图形界面...")
        bot = init_agent_service()
        
        # 配置聊天界面的预设建议问题 (Prompt Suggestions)
        chatbot_config = {
            'prompt.suggestions': [
                '2023年4、5、6月一日门票和二日门票的销量分别是多少？请按周统计。',
                '2023年7月不同省份的入园人数统计排名。',
                '查看2023年10月1日-7日各销售渠道的订单金额排名。',
                '分析一下购买一日票的用户年龄分布。'
            ]
        }
        
        print("✅ Web 界面准备就绪。")
        print("🌐 请在浏览器中打开显示的地址开始对话。")
        
        # 启动 Web 服务
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
        
    except Exception as e:
        print(f"❌ 启动 Web 界面失败: {str(e)}")
        print("💡 请检查网络连接、DashScope API Key 配置以及数据库连通性。")

# ================= 程序入口 =================

if __name__ == '__main__':
    # 默认启动图形界面模式，如需命令行模式可修改为 app_tui()
    # 提示：确保已设置 DASHSCOPE_API_KEY 环境变量
    if not os.getenv('DASHSCOPE_API_KEY'):
        print("⚠️ 警告: 未检测到 DASHSCOPE_API_KEY 环境变量，API 调用可能会失败。")
    
    app_gui()
