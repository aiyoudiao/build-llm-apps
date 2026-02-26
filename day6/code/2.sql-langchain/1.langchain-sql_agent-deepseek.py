"""
===============================================================================
基于 LangChain 的英雄/业务数据库 SQL 智能体

依赖库：langchain, langchain-openai, langchain-community, sqlalchemy, pymysql, python-dotenv

【功能概述】
构建一个 SQL 智能体，能够自主探索数据库结构（Schema），
理解自然语言指令，生成并执行 SQL 查询，最终返回准确答案。
特别针对“英雄”和“订单”相关业务场景进行了测试。

【核心特性】
1. 自主探索：模型可自动查询表结构，无需人工提供 Schema 文档。
2. 自我纠错：若生成的 SQL 报错，模型能根据错误信息自动修正重试。
3. 防幻觉：面对不存在的表（如 HeroDetails），模型会如实报告而非编造。

【安全警告】
1. 数据库账号必须仅具备 READ-ONLY (只读) 权限。
2. 敏感配置必须通过环境变量管理。

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
from dotenv import load_dotenv

# =================导入区域 (适配 LangChain 0.2+) =================

# 1. 大模型接口
from langchain_openai import ChatOpenAI

# 2. 数据库工具 (来自 community 包)
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# 3. Agent 构建工具 
from langchain_community.agent_toolkits.sql.base import create_sql_agent

# 加载 .env 文件中的环境变量
load_dotenv()

# =================配置区域=================

# 数据库连接配置 (优先从环境变量读取)
# 数据库连接配置 (从环境变量读取，若无则使用默认值仅作演示，生产环境请务必配置)
DB_USER = os.environ.get('DB_USER', '')
DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
DB_HOST = os.environ.get('DB_HOST', '')
DB_PORT = os.environ.get('DB_PORT', '3306')
DB_NAME = os.environ.get('DB_NAME_2', 'action')

# 大模型配置
DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 默认使用 deepseek-v3
LLM_MODEL_NAME = os.environ.get('LLM_MODEL', 'deepseek-v3') 
API_KEY = os.environ.get('DASHSCOPE_API_KEY')

# =================初始化函数=================

def init_database():
    """
    初始化数据库连接对象。
    配置 sample_rows_in_table_info=3 以便模型看到数据样例，提高准确率。
    """
    if not DB_PASSWORD:
        raise ValueError("数据库密码未配置，请检查环境变量 DB_PASSWORD。")
    
    # 构建连接 URI
    # 注意：DB_HOST 可能已经包含端口，需处理避免重复
    if ':' not in DB_HOST.split('@')[-1]:
        connection_str = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    else:
        # 如果 host 已经包含 :port，直接拼接
        connection_str = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    
    try:
        db = SQLDatabase.from_uri(
            connection_str, 
            sample_rows_in_table_info=3,
            include_tables=None,  # 允许访问所有表
            ignore_tables=None
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
    """
    if not API_KEY:
        raise ValueError("API Key 未配置，请检查环境变量 DASHSCOPE_API_KEY。")
    
    try:
        llm = ChatOpenAI(
            model=LLM_MODEL_NAME,
            temperature=0.01,  # 低温度，确保 SQL 生成的确定性
            openai_api_base=DASHSCOPE_API_BASE,
            openai_api_key=API_KEY,
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
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="zero-shot-react-description",
        verbose=True,          # 打印思考过程
        max_iterations=10,     # 最大重试次数
        max_execution_time=60, # 最大执行时间 (秒)
        handle_parsing_errors=True # 开启自动错误修正
    )
    
    return agent_executor

# =================主执行流程=================

def main():
    print("正在启动 SQL 智能体任务 (英雄/业务数据库)...")
    
    # 1. 初始化组件
    try:
        db = init_database()
        llm = init_llm()
        agent = create_sql_agent_executor(db, llm)
    except Exception as e:
        print(f"启动失败：{e}")
        return

    # 2. 定义测试任务列表
    # 包含了 Schema 探索、存在性检查、具体查询等多种场景
    tasks = [
        "描述与订单相关的表及其关系",
        "描述 HeroDetails 表",       # 测试对不存在表的反应
        "描述 Hero 表",              # 测试对存在表的反应
        "找出英雄攻击力最高的前 5 个英雄"
    ]

    # 3. 循环执行任务
    print("-" * 30)
    for i, task in enumerate(tasks, 1):
        print(f"\n[任务 {i}/{len(tasks)}]: {task}")
        print("-" * 20)
        
        try:
            # 运行 Agent
            response = agent.invoke({"input": task})
            final_answer = response.get('output', '未获取到回答')
            
            print(f"\n[最终回答]:\n{final_answer}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"\n[执行异常]: {error_msg}")
            # 如果是 DeepSeek 模型不可用，这里会抛出连接或模型错误
            
        print("-" * 30)

if __name__ == '__main__':
    main()
