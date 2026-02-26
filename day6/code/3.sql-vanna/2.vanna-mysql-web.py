"""
基于 Vanna AI 的自然语言数据库查询 Web 助手

功能概述:
启动一个基于 Flask 的 Web 服务，允许用户通过浏览器界面使用自然语言
直接查询 MySQL 数据库。它利用检索增强生成 (RAG) 技术，将用户问题自动转换
为准确的 SQL 语句并执行，同时提供结果可视化和 SQL 修正功能。

核心原理:
1. 向量存储 (Vector Store): 
   - 使用 ChromaDB (SQLite 后端) 持久化存储数据库表结构 (DDL)、业务文档及示例问答的向量索引。
   - 支持自定义存储路径，确保数据在重启后不丢失。
2. 大语言模型 (LLM): 
   - 通过 OpenAI 兼容接口 (如阿里云 DashScope) 调用大模型 (如 qwen-turbo)。
   - 负责理解自然语言问题，结合检索到的上下文生成可执行的 SQL。
3. 检索增强 (RAG): 
   - 在生成 SQL 前，根据用户问题从向量库中精准检索最相关的表结构和业务语境。
   - 解决通用大模型不了解特定数据库 Schema 的痛点。
4. Web 交互界面: 
   - 集成 VannaFlaskApp，提供聊天窗口、SQL 编辑器、结果表格及图表生成工具。

执行流程步骤:
1. 环境准备与配置加载:
   - 加载 .env 文件中的环境变量 (API 密钥、数据库凭证等)。
   - 定义向量数据库的持久化存储路径 (./vanna_data)，若不存在则自动创建。

2. 框架初始化 (自定义类 MyVanna):
   - 继承 ChromaDB_VectorStore 和 OpenAI_Chat，解决配置冲突。
   - 显式实例化 chromadb.PersistentClient，绑定固定存储路径，确保数据持久化。
   - 初始化 OpenAI 客户端，配置自定义 Base URL 和模型名称。

3. 数据库连接与 Schema 索引构建 (模型训练):
   - 建立与目标 MySQL 数据库的连接，并通知 Vanna 实例以便后续执行 SQL。
   - 自动扫描数据库中所有表的元数据 (information_schema)。
   - 提取每个表的 DDL (建表语句)，调用 vn.train(ddl=...) 进行向量化训练。
   - *注：若向量库中已存在相同 ID 的记录，ChromaDB 会自动去重，避免重复训练。*

4. 可选强化训练 (Optimization):
   - 针对复杂业务场景，可额外注入自然语言文档描述 (documentation)。
   - 注入典型问答对 (Golden Query)，明确特定问题对应的 SQL 写法，提高检索准确率。

5. 启动 Web 服务:
   - 初始化 VannaFlaskApp，传入训练好的 vn 实例。
   - 启动 Flask 服务器 (默认监听 0.0.0.0:8080)，允许局域网访问。
   - 用户可通过浏览器进行交互式查询、SQL 修正及结果可视化。

6. 异常处理与资源清理:
   - 捕获数据库连接错误及通用异常，确保程序稳健运行。
   - 在 finally 块中强制关闭 MySQL 游标和连接，防止资源泄露。

依赖环境:
- Python 3.11+
- 核心库: vanna, chromadb, openai, mysql-connector-python, python-dotenv, flask
- 环境变量配置 (.env 文件):
  - DASHSCOPE_API_KEY: 阿里云 API 密钥
  - DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME: 数据库连接信息
  - LLM_MODEL: (可选) 指定模型名称，默认代码中硬编码为 qwen-turbo-latest

安全与运维注意:
- 敏感信息保护: 严禁在代码中硬编码数据库密码或 API 密钥，必须使用环境变量。
- 数据持久化: chroma.sqlite3 文件存储在配置的 persist_directory 中，请勿手动删除，否则 AI 会“失忆”。
- Git 管理: 务必将 chroma.sqlite3 和 .env 文件加入 .gitignore，避免泄露敏感数据或提交二进制数据库文件。
- 生产部署: 建议将 debug=True 设置为 False，并使用 Gunicorn 等 WSGI 服务器替代 Flask 内置服务器。
- 网络访问: host='0.0.0.0' 允许外部访问，请确保防火墙规则安全，仅受信任网络可访问 8080 端口。
"""

# 导入必要的库
# Vanna 核心组件：OpenAI 聊天接口、ChromaDB 向量存储、Flask Web 应用封装
from vanna.openai import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.flask import VannaFlaskApp
from dotenv import load_dotenv

load_dotenv()

# 数据库连接与操作
import mysql.connector

# 系统工具：时间处理、OpenAI 客户端、环境变量读取
import time
from openai import OpenAI
import os
import chromadb

# 定义一个专门的目录来存储向量数据
persist_directory = "./vanna_data"  # 或者绝对路径 '/var/data/vanna'
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


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
    主函数：负责加载配置、初始化模型、训练 Schema 并启动 Web 服务
    """
    
    # --- 1. 安全加载配置信息 ---
    # 从环境变量读取 API 密钥。实际部署时，请在终端 export DASHSCOPE_API_KEY 或使用 .env 文件
    api_key = os.getenv("DASHSCOPE_API_KEY")
    
    # 检查 API 密钥是否存在，若不存在则终止程序，避免后续报错
    if not api_key:
        print("错误：未找到环境变量 DASHSCOPE_API_KEY，请设置后重试。")
        return

    # --- 2. 初始化 OpenAI 客户端 ---
    # 创建兼容 OpenAI 接口的客户端，指向阿里云 DashScope 服务
    client = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云兼容接口地址
        api_key=api_key                                               # 使用环境变量中的密钥
    )

    # --- 3. 初始化 Vanna 实例 ---
    # 实例化自定义的 MyVanna 类
    vn = MyVanna(config={
        'model': 'qwen-turbo-latest',  # 指定使用的模型名称，此处选用通义千问 Turbo
        'client': client,               # 传入自定义的客户端实例
        # 可选：增加 'n_results': 20 以提高检索相关表结构的数量
        'path': persist_directory  # <--- 关键：指定 ChromaDB 的持久化路径
    })

    # --- 4. 定义数据库连接参数 ---
    # 为了代码安全，建议将以下敏感信息也放入环境变量，此处为演示逻辑暂保留变量形式
    # 生产环境请务必使用 os.getenv() 读取
    db_host = os.getenv("DB_HOST", "")
    db_name = os.getenv("DB_NAME", "")
    db_user = os.getenv("DB_USER", "")
    db_password = os.getenv("DB_PASSWORD", "")  # 强烈建议从环境变量读取，不要硬编码
    db_port = int(os.getenv("DB_PORT", "3306"))

    # 检查数据库密码是否已配置（若是默认值或为空，建议警告）
    if db_password == "student321":
        print("警告：检测到使用默认示例密码，生产环境请修改并配置环境变量 DB_PASSWORD。")

    # --- 5. 连接数据库并通知 Vanna ---
    # 让 Vanna 实例知道如何连接数据库，以便后续执行生成的 SQL
    vn.connect_to_mysql(
        host=db_host,
        dbname=db_name,
        user=db_user,
        password=db_password,
        port=db_port
    )

    # 建立原生的 MySQL 连接，用于手动扫描表结构进行训练
    connection = None
    cursor = None
    
    try:
        connection = mysql.connector.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        )
        
        if connection.is_connected():
            print("成功连接到 MySQL 数据库")
            
            # 创建游标对象
            cursor = connection.cursor()
            
            # --- 6. 获取数据库元数据 ---
            # 查询 information_schema 获取当前数据库中所有表的名称
            cursor.execute("""
                SELECT TABLE_NAME 
                FROM information_schema.TABLES 
                WHERE TABLE_SCHEMA = %s
            """, (db_name,))
            
            tables = cursor.fetchall()
            
            if not tables:
                print("未找到任何表，跳过训练步骤。")
            else:
                print(f"发现 {len(tables)} 个表，开始进行 Schema 训练...")
                
                # --- 7. 遍历并训练每个表的 Schema ---
                for (table_name,) in tables:
                    try:
                        # 获取表的创建语句 (DDL)，包含字段类型、注释等详细信息
                        # 使用反引号包裹表名以防表名包含特殊字符
                        cursor.execute(f"SHOW CREATE TABLE `{table_name}`")
                        result = cursor.fetchone()
                        
                        # SHOW CREATE TABLE 返回 (表名, 建表语句)，取第二列
                        create_table_stmt = result[1]
                        
                        print(f"正在训练表 [{table_name}] 的 schema...")
                        
                        # 将 DDL 语句传递给 Vanna 进行向量化训练
                        # 这是让 AI 理解数据库结构的关键步骤
                        vn.train(ddl=create_table_stmt)
                        
                    except Exception as e:
                        # 捕获单个表训练失败的异常，打印警告并继续处理下一个表
                        print(f"警告：训练表 [{table_name}] 时失败 - {str(e)}")
                        continue
                
                print("所有表的 Schema 训练完成。")

                # --- 8. 可选：强化训练 (针对特定业务场景优化) ---
                # 如果自动检索效果不佳，可在此处添加文档描述或示例问答对
                # 例如：vn.train(documentation="表 users 存储用户信息...")
                # 例如：vn.train(question="查销量", sql="SELECT ...")
                
            # --- 9. 启动 Vanna Web 界面 ---
            print("\n正在启动 Vanna Web 界面...")
            print("请在浏览器中访问：http://localhost:8080 (或服务器 IP:8080)")
            
            # 初始化 Flask 应用，传入训练好的 vn 实例
            # debug=True 开启调试模式，便于开发时查看日志
            app = VannaFlaskApp(vn, debug=True)
            
            # 启动 Web 服务器
            # host='0.0.0.0' 表示监听所有网络接口，允许局域网访问
            # port=8080 指定服务端口
            app.run(host='0.0.0.0', port=8080)
            
    except mysql.connector.Error as err:
        # 捕获数据库连接或执行过程中的特定错误
        print(f"数据库发生错误：{err}")
        
    except Exception as e:
        # 捕获其他未知异常
        print(f"发生未知错误：{e}")
        
    finally:
        # --- 10. 资源清理 ---
        # 无论程序是否正常结束，都尝试关闭数据库资源
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
            
        if connection is not None and connection.is_connected():
            try:
                connection.close()
                print("MySQL 连接已安全关闭")
            except Exception:
                pass


if __name__ == "__main__":
    # 仅当脚本直接运行时才执行 main 函数
    main()
