
"""
自然语言数据库查询助手 (Text-to-SQL) - 持久化增强版

基于 Vanna AI 框架，实现通过自然语言直接查询 MySQL 数据库的功能。
本版本特别优化了向量数据的持久化存储，并引入了强化训练机制，
确保 AI 模型在重启后仍能保留“记忆”，并能精准理解特定业务语义。
s
核心原理:
1. 向量存储 (Vector Store): 
   - 使用 ChromaDB (SQLite 后端) 持久化存储数据库表结构 (DDL)、业务文档及示例问答。
   - 支持自定义存储路径 (./vanna_data)，确保数据在程序重启后不丢失。
2. 大语言模型 (LLM): 
   - 通过 OpenAI 兼容接口 (如阿里云 DashScope) 调用大模型。
   - 负责理解自然语言问题，结合检索到的上下文生成可执行的 SQL。
3. 检索增强 (RAG): 
   - 在生成 SQL 前，根据用户问题从向量库中精准检索最相关的表结构和业务语境。
   - 解决通用大模型不了解特定数据库 Schema 的痛点。
4. 强化训练 (Fine-tuning via RAG): 
   - 通过注入自然语言文档描述 (documentation) 和典型问答对 (Golden Query)。
   - 显式引导模型在模糊语义下正确关联特定表结构，显著提高检索准确率。

执行流程步骤:
1. 环境准备与配置加载:
   - 加载 .env 文件中的环境变量 (API 密钥、数据库凭证等)。
   - 定义并创建向量数据库的持久化存储目录 (默认为 ./vanna_data)。

2. 框架初始化 (自定义类 MyVanna):
   - 继承 ChromaDB_VectorStore 和 OpenAI_Chat，解决配置参数冲突。
   - 显式实例化 chromadb.PersistentClient 并绑定固定路径，确保 chroma.sqlite3 存储在指定位置。
   - 初始化 OpenAI 客户端，配置自定义 Base URL 和模型名称。

3. 数据库连接与 Schema 索引构建 (模型训练):
   - 建立与目标 MySQL 数据库的连接，并通知 Vanna 实例以便后续执行 SQL。
   - 自动扫描数据库中所有表的元数据 (information_schema)。
   - 提取每个表的 DDL (建表语句)，调用 vn.train(ddl=...) 进行向量化训练。
   - *注：ChromaDB 会自动去重，重复运行不会导致数据冗余。*

4. 强化训练 (关键优化步骤):
   - 针对特定业务表 (如 heros)，额外训练自然语言文档描述，补充字段语义。
   - 注入典型的问答对 (Golden Query)，明确告诉模型：“当问到 X 时，请使用 Y 表并写出 Z SQL”。
   - 此步骤有效解决了 RAG 检索失效或关联错误的问题。

5. 自然语言查询推理:
   - 接收用户的自然语言问题。
   - 系统在向量库中检索与问题最相关的表结构、文档及示例。
   - 大模型结合检索到的上下文生成可执行的 SQL 语句。
   - 自动在数据库中执行 SQL 并返回结果 (Pandas DataFrame 格式)。

6. 异常处理与资源清理:
   - 捕获数据库连接错误及通用异常，确保程序稳健运行。
   - 在 finally 块中强制关闭 MySQL 游标和连接，修复了游标对象误用 is_connected() 的问题。

依赖环境:
- Python 3.11+
- 核心库: vanna, chromadb, openai, mysql-connector-python, python-dotenv
- 环境变量配置 (.env 文件):
  - DASHSCOPE_API_KEY: 阿里云 API 密钥 (或 OPENAI_API_KEY)
  - DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME: 数据库连接信息
  - LLM_MODEL: (可选) 指定模型名称，默认 qwen-plus

安全与运维注意:
- 敏感信息保护: 严禁在代码中硬编码数据库密码或 API 密钥，必须使用环境变量。
- 数据持久化: chroma.sqlite3 文件存储在 './vanna_data' 目录中。请勿手动删除该文件，否则 AI 会“失忆”。
- Git 管理: 务必将 'vanna_data/' 目录和 '.env' 文件加入 .gitignore，避免泄露敏感数据或提交二进制数据库文件。
- 首次运行: 首次运行时会自动创建向量库并执行全量训练，耗时稍长；后续启动将直接加载已有索引，速度极快。
- 表结构变更: 如果数据库表结构发生变化，建议删除 'vanna_data/chroma.sqlite3' 后重新运行，以更新索引。
"""

import os
import chromadb
import mysql.connector
from mysql.connector import Error
from openai import OpenAI
from vanna.openai import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from dotenv import load_dotenv

load_dotenv()

# 定义一个专门的目录来存储向量数据
persist_directory = "./vanna_data"  # 或者绝对路径 '/var/data/vanna'
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# 定义自定义 Vanna 类，整合向量存储与大语言模型能力
class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    """
    自定义 Vanna 类
    继承自 ChromaDB_VectorStore（负责向量存储）和 OpenAI_Chat（负责大模型对话）
    目的是解决两个基类初始化时配置参数冲突的问题，特别是 client 参数的传递。
    """
    def __init__(self, config=None):
        # 若未提供配置字典，则初始化为空字典
        if config is None:
            config = {}
        
        # 1. 获取自定义路径，如果没有则用默认值
        persist_path = config.get('path', './vanna_data')
        # 确保目录存在
        if not os.path.exists(persist_path):
            os.makedirs(persist_path)
            print(f"创建向量数据存储目录: {persist_path}")

        
        # 2. 显式创建一个带路径的 ChromaDB 客户端
        # 这才是控制 chroma.sqlite3 生成位置的关键
        chroma_client = chromadb.PersistentClient(path=persist_path)
        
        # 3. 准备 OpenAI 的配置
        openai_config = config.copy() # 分离配置
        self.client = openai_config.pop('client', None)
        
        # 4. 【变化点】不再传递 config 字典给 ChromaDB 基类
        # 而是通过 config={'client': chroma_client} 传入我们刚创建的客户端
        # 注意：这里传给 Vanna 的是 chroma_client，而不是 config 字典里的 path
        ChromaDB_VectorStore.__init__(self, config={'client': chroma_client})
        
        # 5. 初始化 OpenAI
        OpenAI_Chat.__init__(self, config=openai_config)


def main():
    """
    主函数：执行数据库连接、Schema 训练及自然语言查询演示
    """
    
    # --- 1. 安全加载配置信息 ---
    # 从环境变量中读取敏感信息，避免代码中硬编码密钥和密码
    # 在实际运行前，请在终端设置 export DASHSCOPE_API_KEY='...' 等环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY")
    api_base_url = os.getenv("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    db_host = os.getenv("DB_HOST", "")
    db_user = os.getenv("DB_USER", "")
    db_password = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_NAME", "")
    db_port = int(os.getenv("DB_PORT", ""))
    model_name = os.getenv("LLM_MODEL", "qwen-plus")

    # 检查必要的环境变量是否已设置
    if not api_key or not db_password:
        print("错误：请设置必要的环境变量 (DASHSCOPE_API_KEY, DB_PASSWORD)")
        return

    # --- 2. 初始化 OpenAI 客户端 ---
    # 创建自定义的 OpenAI 客户端实例，指定 API 密钥、基础 URL 等参数
    client = OpenAI(
        api_key=api_key,
        base_url=api_base_url
    )

    # --- 3. 初始化 Vanna 实例 ---
    # 实例化自定义的 MyVanna 类，传入模型名称和自定义客户端
    vn = MyVanna(config={
        'model': model_name, 
        'client': client,
        # 'n_results': 20  # 默认可能是 10，增加到 20 或更多，确保覆盖所有表
        'path': persist_directory  # <--- 关键：指定 ChromaDB 的持久化路径
    })

    # --- 4. 连接 MySQL 数据库 ---
    connection = None
    cursor = None
    
    try:
        # 建立与 MySQL 数据库的物理连接
        connection = mysql.connector.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        )
        
        # 验证连接状态
        if connection.is_connected():
            print("成功连接到 MySQL 数据库")
            
            # 同时通知 Vanna 实例连接到此数据库，以便后续执行生成的 SQL
            vn.connect_to_mysql(
                host=db_host,
                dbname=db_name,
                user=db_user,
                password=db_password,
                port=db_port
            )
            
            # 创建游标对象，用于执行 SQL 语句
            cursor = connection.cursor()
            
            # --- 5. 获取数据库元数据并训练模型 ---
            # 查询 information_schema 以获取当前数据库中所有表的名称
            cursor.execute("""
                SELECT TABLE_NAME 
                FROM information_schema.TABLES 
                WHERE TABLE_SCHEMA = %s
            """, (db_name,))
            
            # 获取查询结果，tables 是一个包含表名的元组列表
            tables = cursor.fetchall()
            
            if not tables:
                print("未找到任何表，跳过训练步骤。")
                return

            print(f"发现 {len(tables)} 个表，开始进行 Schema 训练...")
            
            # 遍历每一个表，提取其 DDL 语句并训练 Vanna
            for (table_name,) in tables:
                try:
                    # 获取表的创建语句 (DDL)，包含字段定义、索引等信息
                    cursor.execute(f"SHOW CREATE TABLE `{table_name}`")
                    result = cursor.fetchone()
                    
                    # SHOW CREATE TABLE 返回两列：表名和创建语句，我们取第二列
                    create_table_stmt = result[1]
                    
                    print(f"正在训练表 [{table_name}] 的 schema...")
                    
                    # 将 DDL 语句传递给 Vanna 进行训练
                    # 这会将表结构向量化存储，使模型理解该表的含义
                    vn.train(ddl=create_table_stmt)
                    
                except Exception as e:
                    # 捕获单个表训练失败的异常，避免中断整个流程
                    print(f"警告：训练表 [{table_name}] 时失败 - {str(e)}")
                    continue
            
            print("所有表的 Schema 训练完成。")

            # --- [新增步骤] 强化训练 ---
            # 解决检索不到 heros 表的问题
            # 5.1 训练一个关于该表的自然语言描述
            vn.train(documentation="heros 表包含了所有英雄的详细属性，包括 name (名字), attack_max (最大攻击力), hp_max (最大生命值), defense_max (最大防御力) 等。")

            # 5.2 或者直接训练一个类似的问答对 (Golden Query)
            # 这会明确告诉模型：当问到类似问题时，应该用这个 SQL
            vn.train(
                question="哪个英雄的攻击力最高？",
                sql="SELECT name, attack_max FROM heros ORDER BY attack_max DESC LIMIT 1;"
            )

            print("强化训练完成。\n")
            print("已添加额外训练数据以增强检索。")
            
            # --- 6. 执行自然语言查询演示 ---
            question = "找出英雄攻击力最高的前 5 个英雄"
            print(f"\n正在处理问题：{question}")
            
            # 使用 Vanna 的 ask 方法
            # 该方法内部流程：检索相关 schema -> 生成 SQL -> 执行 SQL -> 返回结果 (通常是 DataFrame)
            # 注意：确保数据库中有对应的表和数据，否则可能报错或返回空结果
            result = vn.ask(question)
            
            # 打印查询结果
            print("\n查询结果:")
            print(result)
            
    except Error as err:
        # 捕获数据库连接或执行过程中的错误
        print(f"数据库发生错误：{err}")
        
    except Exception as e:
        # 捕获其他非数据库相关的通用异常
        print(f"发生未知错误：{e}")
        
    finally:
        # --- 7. 资源清理 (已修复) ---
        # 修复点 1: 游标 (cursor) 没有 is_connected() 方法，直接检查是否为 None 后关闭
        if cursor is not None:
            try:
                cursor.close()
                print("数据库游标已关闭")
            except Exception as e:
                print(f"关闭游标时出错: {e}")
            
        # 连接对象 (connection) 才有 is_connected() 方法
        if connection is not None and connection.is_connected():
            connection.close()
            print("MySQL 连接已安全关闭")


if __name__ == "__main__":
    # 仅当脚本直接运行时才执行 main 函数
    main()
