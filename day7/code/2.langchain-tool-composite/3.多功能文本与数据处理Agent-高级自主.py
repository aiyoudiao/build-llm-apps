"""
================================================================================
LangChain 新版多功能数据处理 Agent (基于 LCEL 架构)
================================================================================

【核心功能】
一个智能数据处理助手，能够自主调用本地工具完成复杂的文本分析、
数据格式转换及文本编辑任务。它结合了大模型的语义理解能力与 Python 代码的
精确执行能力。

主要能力：
1. 智能文本分析：统计字数/行数，并基于加权关键词算法判断情感倾向。
2. 稳健的数据转换：支持 JSON <-> CSV 互转，使用标准 csv 库处理复杂格式（如含逗号的字段）。
3. 高级文本编辑：支持查找定位、批量替换及行统计。
4. 自主决策：Agent 根据用户意图，自动选择最合适的工具并组合执行。

【技术优化】
1. 架构升级：完全遵循 LangChain 0.2+ LCEL 标准，使用 create_react_agent。
2. 工具增强：
   - 引入 @tool 装饰器，自动解析函数签名作为工具描述。
   - 使用 csv 模块重写转换逻辑，解决原生 split 无法处理带逗号字段的 Bug。
   - 优化情感分析算法，区分正负向权重。
3. 鲁棒性：开启 handle_parsing_errors，自动修复模型输出格式偏差。
4. 代码规范：完善类型提示 (Type Hints) 和文档字符串。

【执行流程步骤】
--------------------------------------------------------------------------------
Step 1: 定义增强型工具 (Tools)
   - 使用 @tool 装饰器封装三个核心功能类的方法。
   - 工具 1 (analyze_text): 统计 + 加权情感分析。
   - 工具 2 (convert_data): 基于 csv/JSON 标准库的稳健转换。
   - 工具 3 (process_text): 查找、替换、统计行数。

Step 2: 构建 ReAct 提示词 (Prompt Engineering)
   - 定义中文优化的 ReAct 模板，强制模型按 "思考->行动->观察" 循环。
   - 明确工具调用格式，减少幻觉。
        - 强制模型在 Action 阶段输出 ```json {...}``` 代码块。
        - 格式：{"action": "tool_name", "action_input": {"param1": "val1", ...}}

Step 3: 组装 Agent 核心 (LCEL)
   - 初始化 ChatTongyi (qwen-turbo) 模型。
   - 注入 ReActJsonSingleInputOutputParser 解析器。
   - 调用 create_react_agent(llm, tools, prompt) 构建代理。
   - 实例化 AgentExecutor，开启详细日志和自动容错。

Step 4: 任务执行循环
   - 接收用户自然语言指令 (如 "把这个 CSV 转成 JSON")。
   - Agent 自动解析意图 -> 调用对应工具 -> 获取结果。
   - 若需多步操作 (如先分析再转换)，Agent 会自动规划序列。
   - 输出最终处理结果。

【业务流程图解】
  用户指令 --> Agent 意图识别
      ├── (若需分析) --> [文本分析工具] --> 统计 + 情感打分
      ├── (若需转换) --> [数据转换工具] --> csv/JSON 标准库解析与序列化
      └── (若需编辑) --> [文本处理工具] --> 正则/字符串操作
      ↓
  收集工具返回结果 --> LLM 综合总结 --> 输出最终答案

================================================================================
"""

import os
import json
import csv
import io
from typing import List, Dict, Any, Optional

# LangChain 核心组件
from langchain.agents import create_react_agent, AgentExecutor, Tool
# 【关键导入】专用 JSON 解析器，确保多参数工具能正确接收字典而非字符串
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
# from langchain_community.llms import Tongyi
from langchain_community.chat_models import ChatTongyi  # 导入通义千问 ChatTongyi 模型
import dashscope
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. 环境配置
# -----------------------------------------------------------------------------
# 加载 .env 文件
load_dotenv()

# 从环境变量获取 API Key (推荐方式)
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

# -----------------------------------------------------------------------------
# 2. 定义增强型工具 (Tools)
# -----------------------------------------------------------------------------
# 使用 @tool 装饰器，LangChain 会自动读取函数签名和 docstring 生成 Schema
# 当配合 ReActJsonSingleInputOutputParser 使用时，返回的 JSON 会被自动解包为函数参数
@tool
def analyze_text(text: str) -> str:
    """
    分析文本内容，提取字数、字符数、行数及情感倾向。
    输入：完整的文本字符串。
    输出：格式化的分析报告文本。
    """
    if not text:
        return "输入文本为空"
    
    char_count = len(text)
    line_count = len(text.splitlines())
    word_count = len(text.split()) 
    
    # 加权情感分析
    sentiment_scores = {
        "好": 1, "优秀": 2, "喜欢": 2, "快乐": 2, "成功": 2, "美好": 2, "棒": 2, "推荐": 2,
        "差": -1, "糟糕": -2, "讨厌": -2, "悲伤": -2, "失败": -2, "痛苦": -2, "烂": -2
    }
    
    score = 0
    found_words = []
    
    for word, weight in sentiment_scores.items():
        if word in text:
            score += weight
            found_words.append(f"{word}")
    
    if score > 0:
        sentiment = "积极"
    elif score < 0:
        sentiment = "消极"
    else:
        sentiment = "中性"
    
    # 【优化】返回纯文本报告，不要返回 JSON 字符串，减少解析歧义
    report = (
        f"【文本分析报告】\n"
        f"字符数：{char_count}\n"
        f"行数：{line_count}\n"
        f"粗略词数：{word_count}\n"
        f"情感倾向：{sentiment} (得分：{score})\n"
        f"命中关键词：{', '.join(found_words) if found_words else '无'}"
    )
    return report

@tool
def convert_data(input_data: str, input_format: str, output_format: str) -> str:
    """
    在不同数据格式之间进行稳健转换 (支持 JSON <-> CSV)。
    输入参数：input_data(数据), input_format(json/csv), output_format(json/csv)。
    """
    try:
        in_fmt = input_format.lower().strip()
        out_fmt = output_format.lower().strip()
        
        if in_fmt == out_fmt:
            return "输入输出格式相同，无需转换。"

        if in_fmt == "json" and out_fmt == "csv":
            data = json.loads(input_data)
            if not isinstance(data, list):
                data = [data]
            if not data:
                return "空数据"
            
            output = io.StringIO()
            headers = list(set(k for item in data for k in item.keys()))
            writer = csv.DictWriter(output, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
            return output.getvalue()

        elif in_fmt == "csv" and out_fmt == "json":
            output = io.StringIO(input_data)
            reader = csv.DictReader(output)
            result = [row for row in reader]
            return json.dumps(result, ensure_ascii=False, indent=2)
        
        else:
            return f"暂不支持的转换组合：{in_fmt} -> {out_fmt}。仅支持 JSON <-> CSV。"
            
    except json.JSONDecodeError as e:
        return f"JSON 解析失败：{str(e)}"
    except Exception as e:
        return f"转换过程出错：{str(e)}"

@tool
def process_text(operation: str, content: str, search_text: Optional[str] = None, replace_text: Optional[str] = None) -> str:
    """
    执行具体的文本处理操作：统计行数、查找文本、替换文本。
    参数：operation(count_lines/find_text/replace_text), content, search_text, replace_text。
    """
    if not content:
        return "内容为空"
    
    if operation == "count_lines":
        return f"文本共有 {len(content.splitlines())} 行。"
    
    elif operation == "find_text":
        if not search_text:
            return "错误：查找操作需要提供 search_text 参数。"
        
        lines = content.splitlines()
        matches = []
        for i, line in enumerate(lines):
            if search_text in line:
                matches.append(f"第 {i+1} 行：{line}")
        
        if matches:
            return f"找到 {len(matches)} 处匹配:\n" + "\n".join(matches)
        else:
            return f"未在文本中找到 '{search_text}'。"
    
    elif operation == "replace_text":
        if not search_text:
            return "错误：替换操作需要提供 search_text 参数。"
        if replace_text is None:
            return "错误：替换操作需要提供 replace_text 参数。"
        
        count = content.count(search_text)
        if count == 0:
            return f"未找到 '{search_text}'，未进行任何替换。"
        
        new_content = content.replace(search_text, replace_text)
        return f"成功替换 {count} 处。\n新内容预览:\n{new_content[:200]}..." # 截断过长输出
    
    else:
        return f"不支持的操作类型：{operation}。"

tools = [analyze_text, convert_data, process_text]

# -----------------------------------------------------------------------------
# 3. 定义 ReAct 提示词模板 (关键修复：使用英文标签)
# -----------------------------------------------------------------------------

# 注意：Thought, Action, Action Input, Observation, Final Answer 必须是英文
# 这是 LangChain 解析器的硬性要求。
# 模型可以用中文写具体内容，但标签必须保留英文。
# 此模板专门配合 ReActJsonSingleInputOutputParser 使用。
# 它不再要求传统的 "Action: tool\nAction Input: json" 两行格式，
# 而是强制模型直接输出一个包含 action 和 action_input 的 JSON 代码块。
# 这确保了多参数工具能接收到正确的字典结构，而不是被当作字符串塞进第一个参数。
REACT_PROMPT_TEMPLATE = """You are an intelligent data processing assistant. You have access to the following tools:

{tools}

Use the following format strictly:

Question: the input question you must answer
Thought: you should always think about what to do, write your thinking process in Chinese if needed.
Action:
```json
{{"action": "<one of [{tool_names}]>", "action_input": {{...}}}}
```
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, respond in Chinese.

Begin!

Question: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

# -----------------------------------------------------------------------------
# 4. 构建 Agent 与执行器
# -----------------------------------------------------------------------------

def create_agent_executor():
    llm = ChatTongyi(model_name="qwen-turbo", dashscope_api_key=api_key)
    
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        output_parser=ReActJsonSingleInputOutputParser(),
    )
    
    # 关键配置：
    # 1. handle_parsing_errors=True: 允许自动重试
    # 2. max_iterations=15: 防止死循环的最大次数保护
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15 
    )
    
    return executor

# -----------------------------------------------------------------------------
# 5. 主程序入口
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== 多功能数据处理 Agent (修复版) 已启动 ===")
    print("提示：中间思考过程将使用英文标签 (Thought/Action)，但最终回答是中文。\n")
    
    agent_executor = create_agent_executor()
    
    tasks = [

        # 任务 1: 复合任务 (分析 + 统计)
        "分析这段文本的情感倾向并统计行数：'这个产品非常好用，我很喜欢它的设计，体验棒！\n价格也很合理，强烈推荐。\n但是物流稍微有点慢，让人有点失望。'",
        
        # 任务 2: 数据转换 (CSV -> JSON，包含逗号测试)
        "将以下 CSV 数据转换为 JSON 格式：'name,role,comment\n\"张三，经理\",开发,\"代码质量高，态度好\"\n李四,测试,\"发现了很多 Bug，工作细致\"'",
        
        # 任务 3: 文本编辑 (查找与替换)
        "帮我把下面这段话里的'bug'全部替换为'特性'，并告诉我替换了几处：'这个版本修复了 3 个严重 bug，但引入了 2 个新 bug。总体还是有 bug。'"
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n[任务 {i}] : {task}")
        print("-" * 30)
        try:
            response = agent_executor.invoke({"input": task})
            result = response.get("output", "无输出")
            print(f"\n[✅ 最终结果]:\n{result}")
        except Exception as e:
            print(f"\n[❌ 执行错误]: {e}")
        
        print("\n" + "="*60 + "\n")
