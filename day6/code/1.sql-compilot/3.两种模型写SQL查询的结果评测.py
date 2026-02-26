"""
===============================================================================
大模型生成 SQL 自动化执行评测工具

依赖库：pandas, sqlalchemy, mysql-connector-python, openpyxl, python-dotenv

【功能概述】
可用于评估大模型生成的 SQL 语句的可执行性（Execution Accuracy）。
它读取包含生成 SQL 的 Excel 文件，自动连接到 MySQL 数据库逐条执行，
记录执行状态（成功/失败），并将成功的查询结果转换为 Markdown 格式保存。

【安全警告】
1. 数据库账号必须具备且仅具备 READ-ONLY (只读) 权限，严禁使用高权限账号。
2. 数据库密码严禁硬编码在代码中，必须通过环境变量配置。
3. 脚本仅执行每条 SQL 的第一条语句，以防止多语句混合执行的风险。

【核心流程】
1. 配置加载：从环境变量读取数据库连接信息。
2. 数据读取：加载待评测的 Excel 文件。
3. 批量执行：
   - 遍历每一行 SQL。
   - 预处理：清洗空白字符，截取第一条语句。
   - 安全检查：简单拦截非 SELECT 语句（可选）。
   - 执行查询：在数据库中运行 SQL。
   - 结果格式化：成功则转为 Markdown 表格，失败则记录错误堆栈。
4. 结果持久化：更新 Excel 文件，增加“执行状态”和“执行结果”列。

# 以下是.env文件的环境变量配置 

xxxxxxxx 替换为正确的数据库信息，建表语句在 ./data/sql 目录下，默认为 mysql

```
DB_HOST=xxxxxxxx
DB_USER=xxxxxxxx
DB_PASSWORD=xxxxxxx
DB_PORT=3306
DB_NAME=xxxxxxxx

INPUT_FILE_PATH=./data/sql_result_qwen_turbo.xlsx
OUTPUT_FILE_PATH=./data/sql_result_qwen_turbo_evaluated.xlsx
```

===============================================================================
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import traceback
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# =================配置区域=================

# 数据库连接配置
# 强烈建议从环境变量获取，避免硬编码敏感信息
DB_HOST = os.environ.get('DB_HOST', '')
DB_USER = os.environ.get('DB_USER', '')
DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
DB_PORT = int(os.environ.get('DB_PORT', 3306))
DB_NAME = os.environ.get('DB_NAME', '')

# 输入输出文件路径
INPUT_FILE_PATH = os.environ.get('INPUT_FILE_PATH', 'sql_result_qwen_turbo.xlsx')
OUTPUT_FILE_PATH = os.environ.get('OUTPUT_FILE_PATH', 'sql_result_qwen_turbo_evaluated.xlsx')


# 限制 Markdown 结果的最大长度，防止 Excel 单元格过大或文件膨胀
MAX_MARKDOWN_LENGTH = 32000

# =================数据库连接函数=================

def get_db_engine():
    """
    创建数据库连接引擎。
    使用 mysql+mysqlconnector 驱动。
    """
    # 构建连接字符串
    connection_str = (
        f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
        f'?charset=utf8mb4'
    )
    
    try:
        # 创建引擎，设置连接池参数
        engine = create_engine(
            connection_str, 
            pool_size=5, 
            max_overflow=10, 
            pool_recycle=3600,
            echo=False  # 生产环境建议关闭 SQL 日志打印
        )
        print("数据库引擎创建成功。")
        return engine
    except Exception as e:
        print(f"数据库引擎创建失败：{e}")
        raise

def execute_sql_safe(session, sql_query):
    """
    安全地执行 SQL 查询并返回格式化结果。
    
    参数:
        session: SQLAlchemy Session 对象
        sql_query: 待执行的 SQL 字符串
    
    返回:
        tuple: (状态标志, 结果内容)
               状态标志: 'Success' 或 'Failed'
               结果内容: 成功时为 Markdown 表格字符串，失败时为错误信息
    """
    # 简单的安全检查：仅允许 SELECT 语句 (可选，视需求开启)
    # 如果模型可能生成复杂的 CTE 或子查询，简单的 startswith 可能不够，这里仅作基础防护
    cleaned_sql = sql_query.strip().upper()
    if not cleaned_sql.startswith('SELECT') and not cleaned_sql.startswith('WITH'):
        return 'Failed', '安全拦截：仅允许执行 SELECT 或 WITH (CTE) 开头的查询语句。'

    try:
        # 执行 SQL 查询
        # 使用 text() 包裹原始 SQL 以防止 SQLAlchemy 的自动转义干扰，但需注意注入风险
        result_proxy = session.execute(text(sql_query))
        
        # 获取列名
        columns = result_proxy.keys()
        
        # 获取所有数据行
        rows = result_proxy.fetchall()
        
        # 处理空结果集
        if not rows:
            return 'Success', '查询执行成功，但结果为空。'
            
        # 构建 Markdown 表格
        # 表头
        md_table = '| ' + ' | '.join(str(col) for col in columns) + ' |\n'
        # 分隔符
        md_table += '| ' + ' | '.join(['---'] * len(columns)) + ' |\n'
        
        # 数据行
        # 限制显示行数，防止结果集过大导致内存溢出或 Excel 卡顿 (例如限制前 100 行)
        display_rows = rows[:100]
        
        for row in display_rows:
            # 将单元格内容转换为字符串，处理 None 值
            cells = [str(cell) if cell is not None else 'NULL' for cell in row]
            # 清理单元格内容中的换行符和竖线，防止破坏 Markdown 格式
            clean_cells = [c.replace('\n', ' ').replace('|', '/') for c in cells]
            md_table += '| ' + ' | '.join(clean_cells) + ' |\n'
            
        if len(rows) > 100:
            md_table += f"\n... 还有 {len(rows) - 100} 行数据未显示 ..."
            
        # 截断过长的 Markdown 字符串
        if len(md_table) > MAX_MARKDOWN_LENGTH:
            md_table = md_table[:MAX_MARKDOWN_LENGTH] + "\n... (结果过长已截断)"
            
        return 'Success', md_table
        
    except Exception as e:
        # 捕获所有 SQL 执行异常
        error_msg = str(e)
        # 获取详细堆栈信息用于调试，但存入 Excel 时只保留主要错误信息
        # traceback_msg = traceback.format_exc() 
        return 'Failed', f'执行错误: {error_msg}'

# =================主执行流程=================

def main():
    print("正在启动 SQL 自动化执行评测任务...")
    
    # 1. 检查必要的环境变量
    if not DB_PASSWORD or DB_PASSWORD == 'student321':
        print("警告：检测到使用默认密码或未配置 DB_PASSWORD 环境变量。")
        print("在生产环境中，请务必通过 .env 文件或系统环境变量配置真实的数据库密码。")
        # 在实际生产场景中，这里可以选择直接退出
        # return 

    # 2. 初始化数据库连接
    engine = None
    try:
        engine = get_db_engine()
        # 创建 Session 工厂
        SessionFactory = sessionmaker(bind=engine)
    except Exception as e:
        print(f"无法连接数据库，任务终止：{e}")
        return

    # 3. 读取待评测文件
    try:
        if not os.path.exists(INPUT_FILE_PATH):
            print(f"文件未找到：{INPUT_FILE_PATH}")
            return
            
        df = pd.read_excel(INPUT_FILE_PATH)
        print(f"成功加载文件，共 {len(df)} 条记录。")
        
        # 检查必要的列是否存在
        if 'SQL' not in df.columns:
            print("错误：Excel 文件中缺少 'SQL' 列。")
            return
            
    except Exception as e:
        print(f"读取 Excel 文件失败：{e}")
        return

    # 4. 准备结果列
    # 使用列表暂存结果，避免在循环中频繁操作 DataFrame 导致性能下降
    status_list = []
    result_list = []
    
    # 创建一个 Session 用于整个循环 (利用事务隔离，或者每次循环新建，这里选择复用以提高效率)
    # 注意：如果某次执行导致 Session 污染，可能需要每次循环新建。对于只读查询，复用通常是安全的。
    session = SessionFactory()
    
    print("-" * 30)
    for index, row in df.iterrows():
        raw_sql = row.get('SQL')
        
        # 处理空值或无效 SQL
        if pd.isna(raw_sql) or str(raw_sql).strip() == '' or str(raw_sql).lower() == 'nan':
            status_list.append('Failed')
            result_list.append('错误：未找到生成的 SQL 语句')
            print(f"[{index+1}/{len(df)}] 跳过：SQL 为空")
            continue
            
        sql_query = str(raw_sql).strip()
        
        # 预处理：如果有多个语句（以分号分隔），只取第一个执行
        # 这是为了防止模型生成 "SELECT ...; DROP TABLE ...;" 这种恶意或多重语句
        if ';' in sql_query:
            sql_query = sql_query.split(';')[0].strip()
            
        print(f"[{index+1}/{len(df)}] 正在执行：{sql_query[:50]}...")
        
        try:
            # 执行 SQL
            status, content = execute_sql_safe(session, sql_query)
            
            status_list.append(status)
            result_list.append(content)
            
            if status == 'Success':
                print(f"   -> 执行成功")
            else:
                print(f"   -> 执行失败：{content[:50]}...")
                
        except Exception as e:
            # 捕获未预料的异常，确保循环不中断
            error_detail = traceback.format_exc()
            status_list.append('Failed')
            result_list.append(f'系统异常：{str(e)}')
            print(f"   -> 系统异常：{e}")
            
        # 可选：每执行一定次数提交或刷新，保持连接活跃
        if (index + 1) % 50 == 0:
            session.close()
            session = SessionFactory() # 重建 session 以防连接超时
            
    # 关闭最终 session
    session.close()
    engine.dispose() # 关闭连接池

    # 5. 将结果写入 DataFrame
    df['执行状态'] = status_list
    df['执行结果(Markdown)'] = result_list
    
    # 计算成功率
    success_count = sum(1 for s in status_list if s == 'Success')
    total_count = len(df)
    accuracy = (success_count / total_count * 100) if total_count > 0 else 0
    
    print("-" * 30)
    # !NOTE 这里的准确率只是sql执行是否正常率，并不是 sql 真正的正确率，这个暂时没有一个客观性的量化标准去衡量
    print(f"评测完成！总计：{total_count}, 成功：{success_count}, 准确率：{accuracy:.2f}%")
    
    # 6. 保存结果
    try:
        df.to_excel(OUTPUT_FILE_PATH, index=False)
        print(f"详细结果已保存至：{OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"保存 Excel 文件失败：{e}")

if __name__ == '__main__':
    main()
