"""
===============================================================================
保险业务自然语言转SQL批量生成工具

依赖库：dashscope, pandas, openpyxl (用于Excel输出)

【功能概述】
本脚本旨在利用阿里云通义千问大模型（Qwen-turbo），将保险业务领域的自然语言问题
自动转换为可执行的 SQL 查询语句。它通过读取本地的数据库表结构描述文件和问题列表，
批量调用大模型接口，生成对应的 SQL 代码，并将结果（原始问题、生成的SQL、耗时、状态）
保存为 Excel 报告。

【适用场景】
1. 辅助数据分析师：让不懂 SQL 的业务人员快速获取数据查询语句。
2. 模型效果评估：批量测试大模型在特定领域（保险）Schema 下的 Text-to-SQL 准确率。
3. 自动化报表前置：作为自动化数据 pipeline 的一环，动态生成查询逻辑。

【执行流程】
1. 初始化配置：
   - 从环境变量读取 DashScope API Key。
   - 定义输入（表结构文件、问题列表文件）和输出（Excel 结果文件）路径。

2. 数据加载：
   - 读取数据库表结构描述（Schema Description），作为模型的上下文知识。
   - 读取自然语言问题列表，支持自定义分隔符（=====）切分多个问题。

3. 批量处理循环：
   - 遍历每一个自然语言问题。
   - 构建提示词（Prompt）：结合“系统指令（角色设定+格式约束）” + “表结构上下文” + “用户问题”。
   - 调用大模型接口：发送请求并获取响应，包含重试机制以应对网络波动或限流。
   - 解析响应：使用正则表达式从模型返回的文本中提取纯净的 SQL 代码（去除多余解释）。
   - 记录结果：保存原始问题、生成的 SQL、耗时以及执行状态（成功/失败）。
   - 异常保护：单个问题生成失败不会中断整个流程，会记录错误信息并继续处理下一个。

4. 结果输出：
   - 将所有处理结果整理为 Pandas DataFrame。
   - 导出为 Excel 文件，方便后续人工审核或自动化执行。
   - 打印任务统计信息（总数量、成功率、平均耗时）。

【注意事项】
- 请确保已安装依赖库：pip install dashscope pandas openpyxl
- 请在操作系统环境变量中配置 DASHSCOPE_API_KEY。
- 输入文件路径需与实际文件位置一致。
===============================================================================
"""

import json
import os
import re
import time
import pandas as pd
import dashscope
from dotenv import load_dotenv
load_dotenv()

# =================配置区域=================

# 定义输入输出文件路径
# 存储生成结果的 Excel 文件路径
SAVE_FILE_PATH = './input/sql_result_qwen_turbo.xlsx'
# 数据库表结构描述文件路径（包含表名、字段名及含义）
TABLE_DESC_FILE_PATH = './data/数据表字段说明-精简1.txt'
# 自然语言问题列表文件路径（每个问题之间用 ===== 分隔）
QA_LIST_FILE_PATH = './data/qa_list-2.txt'

# 定义使用的通义千问模型版本
MODEL_NAME = 'qwen-turbo-latest'

# =================初始化设置=================

def init_api_key():
    """
    从环境变量中获取并设置 DashScope API Key。
    """
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        raise ValueError("未在环境变量中找到 DASHSCOPE_API_KEY，请先行配置。")
    dashscope.api_key = api_key
    print("系统提示：DashScope API Key 加载成功。")

# =================核心功能函数=================

def call_llm_api(messages):
    """
    调用通义千问大模型接口获取响应。
    包含重试逻辑和通用异常处理。
    """
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # 调用 API
            response = dashscope.Generation.call(
                model=MODEL_NAME,
                messages=messages,
                result_format='message',
                timeout=60
            )
            
            # 关键修改：通过检查 status_code 判断业务是否成功
            # 200 表示成功，其他代码表示各种错误（如限流、参数错误等）
            if response.status_code == 200:
                return response
            else:
                error_msg = f"API 请求失败，状态码：{response.status_code}, 信息：{response.message}"
                print(error_msg)
                
                # 如果是 5xx 服务器错误或 429 限流，尝试重试
                if response.status_code >= 500 or response.status_code == 429:
                    retry_count += 1
                    print(f"正在重试 ({retry_count}/{max_retries})...")
                    time.sleep(2)
                else:
                    # 其他错误（如 401 授权失败，400 参数错误）通常重试无效，直接抛出
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
    """
    # 优先匹配 ```sql ... ```
    pattern_sql = r'```sql\s*(.*?)\s*```'
    match = re.search(pattern_sql, response_content, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    # 兜底匹配任意 ``` ... ```
    pattern_generic = r'```\s*(.*?)\s*```'
    match = re.search(pattern_generic, response_content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    return response_content.strip()

def generate_single_sql(query_text, table_schema):
    """
    构建提示词并调用模型生成单个问题的 SQL。
    """
    system_prompt = (
        "你是一位精通保险业务数据的 SQL 专家。"
        "请根据提供的数据库表结构描述，将用户的自然语言问题转换为标准的 SQL 查询语句。"
        "要求：\n"
        "1. 仅输出 SQL 代码，不要包含任何解释性文字。\n"
        "2. 必须将 SQL 代码包裹在 ```sql 和 ``` 标记中。\n"
        "3. 如果涉及多表查询，请确保 JOIN 条件正确。\n"
        "4. 如果有多个查询需求，尝试合并为一个高效的 SQL 语句。"
    )
    
    user_prompt = (
        f"以下是数据库的表结构描述：\n{table_schema}\n"
        f"=====分割线=====\n"
        f"请将以下自然语言问题转换为 SQL：\n{query_text}"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    start_time = time.time()
    
    # 调用封装好的 API 函数
    response = call_llm_api(messages)
    
    elapsed_time = round(time.time() - start_time, 2)
    
    # 安全地获取内容，防止结构变化导致报错
    try:
        content = response.output.choices[0].message.content
    except (AttributeError, IndexError, KeyError) as e:
        raise RuntimeError(f"解析模型响应结构失败：{e}. 原始响应：{str(response)}")
    
    sql_code = extract_sql_code(content)
    
    return sql_code, content, elapsed_time

# =================主执行流程=================

def main():
    print("正在启动 SQL 自动生成任务...")
    
    # 1. 初始化 API Key
    try:
        init_api_key()
    except ValueError as e:
        print(f"启动失败：{e}")
        return

    # 2. 读取基础数据文件
    try:
        with open(TABLE_DESC_FILE_PATH, 'r', encoding='utf-8') as f:
            table_description = f.read()
        print(f"成功加载表结构描述文件：{TABLE_DESC_FILE_PATH}")
        
        with open(QA_LIST_FILE_PATH, 'r', encoding='utf-8') as f:
            raw_qa_content = f.read()
        
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
        'QA': [],       
        'SQL': [],      
        'Time_Cost': [],
        'Status': []    
    }

    # 4. 循环处理每个问题
    print("-" * 30)
    for index, query in enumerate(qa_list, 1):
        print(f"[{index}/{len(qa_list)}] 正在处理：{query[:50]}...")
        
        try:
            sql, raw_response, cost_time = generate_single_sql(query, table_description)
            
            results_data['QA'].append(query)
            results_data['SQL'].append(sql)
            results_data['Time_Cost'].append(cost_time)
            results_data['Status'].append('Success')
            
            print(f"   -> 生成成功 (耗时：{cost_time}s)")
            print(f"   -> 模型结果：\n{raw_response}")
            
            
        except Exception as e:
            error_msg = str(e)
            print(f"   -> 生成失败：{error_msg}")
            
            results_data['QA'].append(query)
            results_data['SQL'].append(f"-- Error: {error_msg}")
            results_data['Time_Cost'].append(0.0)
            results_data['Status'].append('Failed')
            
        time.sleep(0.5)

    # 5. 保存结果
    print("-" * 30)
    print("正在保存结果到 Excel...")
    
    df_result = pd.DataFrame(results_data)
    
    try:
        df_result.to_excel(SAVE_FILE_PATH, index=False)
        print(f"任务完成！结果已保存至：{SAVE_FILE_PATH}")
        
        total = len(df_result)
        success = len(df_result[df_result['Status'] == 'Success'])
        if success > 0:
            avg_time = df_result[df_result['Status'] == 'Success']['Time_Cost'].mean()
            print(f"统计信息：总计 {total} 条，成功 {success} 条，平均耗时 {avg_time:.2f} 秒。")
        else:
            print(f"统计信息：总计 {total} 条，成功 0 条。")
        
    except Exception as e:
        print(f"保存 Excel 文件失败：{e}")
    
    # =================结果展示=================
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
    main()
