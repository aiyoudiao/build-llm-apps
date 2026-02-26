"""
===============================================================================
保险业务自然语言转SQL批量生成工具 (Qwen-Coder版)

依赖库：dashscope, pandas, openpyxl, python-dotenv

【功能概述】
使用阿里云通义千问代码专用模型（Qwen-coder-plus），基于标准的 SQL 建表语句 (DDL)，
将保险业务领域的自然语言问题自动转换为高质量的可执行 SQL 查询语句。
相比通用模型，Qwen-Coder 在理解数据库 Schema 和生成复杂 SQL 逻辑方面表现更佳。

【核心流程】
1. 初始化：加载 API Key，读取 DDL 建表语句和问题列表。
2. 构建提示词：将 DDL 作为上下文，结合用户问题，构造适合代码模型的指令。
3. 调用模型：批量请求 Qwen-coder-plus 接口，包含超时重试和错误处理。
4. 解析提取：通过正则表达式精准提取 ```sql 代码块。
5. 结果持久化：将问题、生成的 SQL、耗时及状态保存为 Excel 文件，并打印统计报告。

【注意事项】
- 确保已安装依赖：pip install dashscope pandas openpyxl python-dotenv
- 需在环境变量或 .env 文件中配置 DASHSCOPE_API_KEY。
- 输入文件 create_sql.txt 应包含标准的 CREATE TABLE 语句。
===============================================================================
"""

import json
import os
import re
import time
import pandas as pd
import dashscope
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# =================配置区域=================

# 定义输入输出文件路径
SAVE_FILE_PATH = './data/sql_result_qwen_coder.xlsx'
QA_LIST_FILE_PATH = './data/qa_list-2.txt'
DDL_FILE_PATH = './data/create_sql.txt'

# 定义使用的通义千问代码模型版本
# qwen-coder-plus 在代码生成任务上通常优于 turbo 版本
MODEL_NAME = 'qwen-coder-plus'

# =================初始化设置=================

def init_api_key():
    """
    从环境变量中获取并设置 DashScope API Key。
    支持从系统环境变量或 .env 文件读取。
    """
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        raise ValueError("未在环境变量中找到 DASHSCOPE_API_KEY，请检查 .env 文件或系统配置。")
    dashscope.api_key = api_key
    print("系统提示：DashScope API Key 加载成功。")

# =================核心功能函数=================

def call_llm_api(messages):
    """
    调用通义千问大模型接口获取响应。
    包含重试逻辑、状态码检查和通用异常处理，确保批量任务的稳定性。
    
    参数:
        messages: 符合 DashScope 格式的对话消息列表
    
    返回:
        response: 模型原始响应对象
    """
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # 调用 API
            response = dashscope.Generation.call(
                model=MODEL_NAME,
                messages=messages,
                result_format='message',  # 指定返回格式为 message 对象
                timeout=60  # 设置超时时间为 60 秒，防止长 SQL 生成中断
            )
            
            # 检查业务状态码
            if response.status_code == 200:
                return response
            else:
                error_msg = f"API 请求失败，状态码：{response.status_code}, 信息：{response.message}"
                print(error_msg)
                
                # 针对服务器错误 (5xx) 或限流 (429) 进行重试
                if response.status_code >= 500 or response.status_code == 429:
                    retry_count += 1
                    wait_time = 2 * retry_count  # 指数退避
                    print(f"正在重试 ({retry_count}/{max_retries})，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                else:
                    # 客户端错误 (如 401, 400) 重试通常无效，直接抛出
                    raise RuntimeError(error_msg)
                
        except Exception as e:
            # 捕获网络超时、连接错误等通用异常
            error_str = str(e)
            print(f"发生网络或 SDK 错误：{error_str}，正在重试 ({retry_count + 1}/{max_retries})...")
            retry_count += 1
            
            if retry_count < max_retries:
                time.sleep(2)
            else:
                raise RuntimeError(f"多次重试后仍失败。最后错误信息：{error_str}")
            
    raise RuntimeError("达到最大重试次数，请求失败。")

def extract_sql_code(response_content):
    """
    从模型返回的文本中提取纯 SQL 代码。
    优先匹配 ```sql 代码块，其次匹配任意 ``` 代码块。
    
    参数:
        response_content: 模型返回的完整文本内容
    
    返回:
        sql_code: 提取出的 SQL 字符串
    """
    # 正则表达式 1：精准匹配 ```sql ... ``` (忽略大小写)
    pattern_sql = r'```sql\s*(.*?)\s*```'
    match = re.search(pattern_sql, response_content, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    # 正则表达式 2：兜底策略，匹配任意 ``` ... ```
    pattern_generic = r'```\s*(.*?)\s*```'
    match = re.search(pattern_generic, response_content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # 如果未找到代码块，返回原始内容（可能模型直接输出了 SQL）
    return response_content.strip()

def generate_single_sql(query_text, ddl_schema):
    """
    构建针对代码模型优化的提示词，并调用模型生成 SQL。
    
    参数:
        query_text: 用户的自然语言问题
        ddl_schema: 数据库建表语句 (DDL)
    
    返回:
        tuple: (生成的 SQL 字符串, 原始响应内容, 耗时秒数)
    """
    # 系统提示词：设定专家角色，强调基于 DDL 生成标准 SQL
    system_prompt = (
        "你是一位精通数据库架构和 SQL 编写的专家助手。"
        "你的任务是根据提供的数据库建表语句 (DDL)，将用户的自然语言问题转换为准确、高效的 SQL 查询语句。"
        "请严格遵守以下规则：\n"
        "1. 只输出 SQL 代码，不要包含任何解释、注释或额外的文字。\n"
        "2. 必须将 SQL 代码包裹在 ```sql 和 ``` 标记中。\n"
        "3. 确保引用的表名和字段名与提供的 DDL 完全一致。\n"
        "4. 如果问题涉及多表关联，请正确使用 JOIN 语法。"
    )
    
    # 用户提示词：采用结构化格式，清晰区分输入 (DDL) 和任务 (Question)
    # 这种格式特别适合 Qwen-Coder 系列模型
    user_prompt = (
        f"-- Database Schema (DDL):\n{ddl_schema}\n\n"
        f"### Question:\n{query_text}\n\n"
        f"### Response:\n"
        f"Here is the SQL query to answer the question:\n"
    )
    
    # 组装消息列表
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # 记录开始时间
    start_time = time.time()
    
    # 调用 API
    response = call_llm_api(messages)
    
    # 计算耗时
    elapsed_time = round(time.time() - start_time, 2)
    
    # 安全地获取响应内容
    try:
        content = response.output.choices[0].message.content
    except (AttributeError, IndexError, KeyError) as e:
        raise RuntimeError(f"解析模型响应结构失败：{e}. 原始响应：{str(response)}")
    
    # 提取纯净的 SQL 代码
    sql_code = extract_sql_code(content)
    
    return sql_code, content, elapsed_time

# =================主执行流程=================

def main():
    """
    主程序入口：协调文件读取、批量处理和结果保存。
    """
    print("正在启动 SQL 自动生成任务 (Qwen-Coder 版)...")
    
    # 1. 初始化 API Key
    try:
        init_api_key()
    except ValueError as e:
        print(f"启动失败：{e}")
        return

    # 2. 读取基础数据文件
    ddl_schema = ""
    qa_list = []
    
    try:
        # 读取 DDL 建表语句
        with open(DDL_FILE_PATH, 'r', encoding='utf-8') as f:
            ddl_schema = f.read()
        print(f"成功加载建表语句文件：{DDL_FILE_PATH}")
        
        # 读取问题列表
        with open(QA_LIST_FILE_PATH, 'r', encoding='utf-8') as f:
            raw_qa_content = f.read()
        
        # 使用 ===== 作为分隔符切分问题，并去除空白项
        qa_list = [q.strip() for q in raw_qa_content.split('=====') if q.strip()]
        print(f"成功加载问题列表，共 {len(qa_list)} 个问题。")
        
    except FileNotFoundError as e:
        print(f"文件未找到错误：{e}，请检查文件路径是否正确。")
        return
    except Exception as e:
        print(f"读取文件时发生未知错误：{e}")
        return

    # 3. 准备结果存储容器
    results_data = {
        'QA': [],       # 原始问题
        'SQL': [],      # 生成的 SQL
        'Time_Cost': [],# 耗时
        'Status': []    # 执行状态
    }

    # 4. 循环处理每个问题
    print("-" * 30)
    for index, query in enumerate(qa_list, 1):
        # 清理问题中的多余换行符
        clean_query = query.replace('\n', ' ')
        print(f"[{index}/{len(qa_list)}] 正在处理：{clean_query[:50]}...")
        
        try:
            # 调用生成函数
            sql, raw_response, cost_time = generate_single_sql(clean_query, ddl_schema)
            
            # 记录成功结果
            results_data['QA'].append(clean_query)
            results_data['SQL'].append(sql)
            results_data['Time_Cost'].append(cost_time)
            results_data['Status'].append('Success')
            
            print(f"   -> 生成成功 (耗时：{cost_time}s)")
            print(f"   -> 模型结果：\n{raw_response}")
            
        except Exception as e:
            # 记录失败结果，确保单个失败不影响后续处理
            error_msg = str(e)
            print(f"   -> 生成失败：{error_msg}")
            
            results_data['QA'].append(clean_query)
            results_data['SQL'].append(f"-- Error: {error_msg}")
            results_data['Time_Cost'].append(0.0)
            results_data['Status'].append('Failed')
            
        # 简单的延时，避免触发 API 频率限制
        time.sleep(0.5)

    # 5. 将结果保存为 DataFrame 并导出 Excel
    print("-" * 30)
    print("正在保存结果到 Excel...")
    
    df_result = pd.DataFrame(results_data)
    
    try:
        # 导出时不包含索引列
        df_result.to_excel(SAVE_FILE_PATH, index=False)
        print(f"任务完成！结果已保存至：{SAVE_FILE_PATH}")
        
        # 打印简要统计
        total = len(df_result)
        success = len(df_result[df_result['Status'] == 'Success'])
        if success > 0:
            avg_time = df_result[df_result['Status'] == 'Success']['Time_Cost'].mean()
            print(f"统计信息：总计 {total} 条，成功 {success} 条，平均耗时 {avg_time:.2f} 秒。")
        else:
            print(f"统计信息：总计 {total} 条，成功 0 条。")
        
    except Exception as e:
        print(f"保存 Excel 文件失败：{e}")

    # 6. 结果展示
        
    print("\n=== 详细结果预览 ===")
    
    # 使用 zip 并行遍历，代码更优雅，避免索引越界风险
    # 如果数据量很大（比如超过50条），建议只打印前 5 条和最后 5 条，或者注释掉此循环以免刷屏
    limit_print = 10  # 限制只打印前10条，防止控制台刷屏，如需全部请改为 len(results_data['QA'])
    
    for i, (qa, sql, cost, status) in enumerate(zip(
        results_data['QA'],
        results_data['SQL'],
        results_data['Time_Cost'],
        results_data['Status']
    )):
        if i >= limit_print:
            print(f"... 还有 {len(results_data['QA']) - limit_print} 条记录未显示，请查看 Excel 文件 ...")
            break
            
        print(f"Record {i+1}: [{status}] (耗时: {cost}s)")
        print(f"  QA : {qa}")
        # 如果状态是成功，只显示 SQL；如果是失败，显示错误信息
        if status == 'Success':
            # 简单截断过长的 SQL 以便预览
            sql_preview = sql if len(sql) < 200 else sql[:200] + "..."
            print(f"  SQL: {sql_preview}")
        else:
            print(f"  Err: {sql}") # 此时 SQL 列存的是错误信息
        print("-" * 30)

if __name__ == '__main__':
    # 执行主程序
    main()
