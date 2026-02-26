"""
===============================================================================
基于 LangChain 的保险数据库 SQL 智能体

依赖库：langchain, langchain-openai, langchain-community, sqlalchemy, pymysql, python-dotenv

【功能概述】
使用 LangChain 框架构建一个 SQL Agent（智能体）。
该智能体能够理解自然语言问题，自主探索数据库结构（Schema），
生成正确的 SQL 查询语句，执行查询，并将结果以自然语言形式返回。
相比直接生成 SQL，Agent 模式具备自我纠错和多步推理能力。

【核心组件】
1. SQLDatabase: 负责连接 MySQL 数据库并提供元数据查询能力。
2. ChatModel: 使用阿里云通义千问 (Qwen) 作为推理大脑。
3. SQLDatabaseToolkit: 提供一组工具（查表结构、执行SQL等）给 Agent 使用。
4. AgentExecutor: 控制推理循环，决定何时调用何种工具。

【安全警告】
1. 数据库账号必须仅具备 READ-ONLY (只读) 权限。
2. 敏感配置（密码、API Key）必须通过环境变量管理。

配置 .env 文件

```
# 数据库配置
DB_USER=xxxx
DB_PASSWORD=xxxx
DB_HOST=xxxx
DB_PORT=3306
DB_NAME=life_insurance

# 大模型配置
DASHSCOPE_API_KEY=xxxx
# 可选：指定模型
LLM_MODEL=xxxx
```
===============================================================================
"""

import os
from typing import Optional
from dotenv import load_dotenv

# 尝试加载新版 LangChain 结构，若失败则提示用户安装
try:
    # 1. 大模型接口 (来自 langchain_openai)
    from langchain_openai import ChatOpenAI
    # 2. 数据库工具 (来自 langchain_community)
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    # 3. Agent 构建工具 (来自 langchain_community.agent_toolkits.sql.base)
    from langchain_community.agent_toolkits.sql.base import create_sql_agent
    # 4. 核心执行器 (来自 langchain.agents)
    from langchain.agents import AgentExecutor
except ImportError as e:
    print(f"导入错误：{e}")
    print("请确保已安装最新依赖：pip install langchain langchain-openai langchain-community sqlalchemy pymysql python-dotenv")
    raise

# 加载 .env 文件中的环境变量
load_dotenv()

# =================配置区域=================

# 数据库连接配置 (从环境变量读取，若无则使用默认值仅作演示，生产环境请务必配置)
DB_USER = os.environ.get('DB_USER', '')
DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
DB_HOST = os.environ.get('DB_HOST', '')
DB_PORT = os.environ.get('DB_PORT', '3306')
DB_NAME = os.environ.get('DB_NAME', '')

# 大模型配置
# 阿里云 DashScope 兼容模式地址
DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 推荐使用阿里云原生模型，如 qwen-plus, qwen-turbo, qwen-max
LLM_MODEL_NAME = os.environ.get('LLM_MODEL', 'qwen-plus')
API_KEY = os.environ.get('DASHSCOPE_API_KEY')
if not API_KEY:
    raise ValueError("未在环境变量中找到 DASHSCOPE_API_KEY，请先行配置。")

# =================初始化函数=================
def init_database():
    """
    初始化数据库连接对象。
    使用 pymysql 驱动连接 MySQL。
    """
    if not DB_PASSWORD:
        raise ValueError("数据库密码未配置，请检查环境变量 DB_PASSWORD。")
    
    # 构建连接 URI
    # 格式: mysql+pymysql://user:password@host:port/database
    db_uri = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    try:
        # 创建 SQLDatabase 实例
        # sample_rows_in_table_info: 在建表信息中展示几行样本数据，有助于模型理解数据格式
        db = SQLDatabase.from_uri(
            db_uri, 
            sample_rows_in_table_info=3,
            include_tables=None,  # 包含所有表，也可指定表名列表
            ignore_tables=None    # 忽略某些表
        )
        print("数据库连接初始化成功。")
        print(f"可用表列表：{db.get_usable_table_names()}")
        return db
    except Exception as e:
        print(f"数据库连接失败：{e}")
        raise

def init_llm():
    """
    初始化大语言模型客户端。
    配置为使用阿里云 DashScope 兼容接口。
    """
    if not API_KEY:
        raise ValueError("API Key 未配置，请检查环境变量 DASHSCOPE_API_KEY。")
    
    try:
        llm = ChatOpenAI(
            model=LLM_MODEL_NAME,
            temperature=0.01,  # 低温度值，使输出更确定，适合代码生成
            openai_api_base=DASHSCOPE_API_BASE,
            openai_api_key=API_KEY,
            # 增加超时设置，防止长查询中断
            request_timeout=60
        )
        print(f"大模型初始化成功：{LLM_MODEL_NAME}")
        return llm
    except Exception as e:
        print(f"大模型初始化失败：{e}")
        raise

def create_sql_agent_executor(db, llm):
    """
    创建并配置 SQL Agent 执行器。
    """
    # 创建工具包
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    # 创建 Agent
    # agent_type 指定为 'zero-shot-react-description'，这是处理数据库查询的经典模式
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="zero-shot-react-description",
        verbose=True,  # 开启详细日志，可以看到模型的思考过程
        max_iterations=10,  # 限制最大思考步数，防止死循环
        max_execution_time=60,  # 限制最大执行时间
        handle_parsing_errors=True  # 自动处理 SQL 解析错误，让模型尝试重试
    )
    
    return agent_executor

# =================主执行流程=================

def main():
    print("正在启动 SQL 智能体任务...")
    
    # 1. 初始化组件
    try:
        db = init_database()
        llm = init_llm()
        agent = create_sql_agent_executor(db, llm)
    except Exception as e:
        print(f"启动失败：{e}")
        return

    # 2. 定义测试问题列表
    questions = [
        "获取所有客户的姓名和联系电话。",
        "查询所有未支付保费的保单号和客户姓名。",
        "找出所有理赔金额大于 10000 元的理赔记录，并列出相关客户的姓名和联系电话。",
        "统计每个险种的保单数量，并按数量降序排列。",
        "找出所有已婚客户的客户ID和配偶姓名。",
        "查找代理人的姓名和执照到期日期，按照执照到期日期升序排序。",
        "获取所有保险产品的产品名称和保费，按照保费降序排序。",
        "查询销售区域在上海的员工，列出他们的姓名和职位。",
        "找出所有年龄在30岁以下的客户，并列出其客户ID、姓名和出生日期。",
        "查找所有已审核但尚未支付的理赔记录，包括理赔号、审核人和审核日期。",
        "获取每个产品类型下的平均保费，以及该产品类型下的产品数量。"
    ]

    # 3. 循环执行问题
    print("-" * 30)
    for i, question in enumerate(questions, 1):
        print(f"\n[问题 {i}/{len(questions)}]: {question}")
        print("-" * 20)
        
        try:
            # 运行 Agent
            response = agent.invoke({"input": question})
            
            # 提取最终输出
            # 在新版 LangChain 中，invoke 返回一个字典，输出通常在 'output' 键中
            final_answer = response.get('output', '未获取到回答')
            
            print(f"\n[回答]: {final_answer}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"\n[执行错误]: {error_msg}")
            # 如果是超时或上下文错误，可以选择中断或继续
            
        print("-" * 30)

if __name__ == '__main__':
    main()
