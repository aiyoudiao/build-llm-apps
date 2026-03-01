"""
================================================================================
LangChain 新版多功能数据处理 Agent (保留类结构 + 现代 API)
================================================================================

【核心功能】
一个智能数据处理助手，能够自主调用本地工具完成复杂的文本分析、
数据格式转换及文本编辑任务。其独特之处在于完整保留了 V1 版本的“面向对象”
工具类结构，同时通过桥接模式无缝接入了 LangChain 最新的 ReAct Agent 架构。

主要能力：
1. 智能文本分析：统计字数/行数，并基于基础关键词匹配算法判断情感倾向。
2. 稳健的数据转换：支持 JSON <-> CSV 互转，保留原有的字符串处理逻辑。
3. 高级文本编辑：支持查找定位、批量替换及行统计。
4. 自主决策：Agent 根据用户意图，自动选择最合适的工具并组合执行。

【技术优化】
1. 架构升级：完全遵循 LangChain 0.2+ LCEL 标准，使用 create_react_agent。
2. 兼容性设计：
   - 保留 V1 的 class Tool 定义 (TextAnalysisTool, DataConversionTool 等)。
   - 引入 @tool 装饰器作为“适配器”，将旧式类方法转换为新标准工具。
   - 解决旧版 Tool 无法正确处理多参数的问题。
3. 解析增强：
   - 引入 ReActJsonSingleInputOutputParser，强制模型输出结构化 JSON。
   - 确保多参数工具 (如数据转换) 能准确接收字典而非字符串。
4. 鲁棒性：开启 handle_parsing_errors，自动修复模型输出格式偏差。

【执行流程步骤】
--------------------------------------------------------------------------------
Step 1: 实例化经典工具类 (Legacy Classes)
   - 初始化 TextAnalysisTool, DataConversionTool, TextProcessingTool 实例。
   - 保留原有的 run() 方法逻辑和内部算法。

Step 2: 构建适配层 (Adapter Layer)
   - 定义辅助函数 (如 use_text_analysis)，内部调用类实例的 run 方法。
   - 使用 @tool 装饰器包装辅助函数，自动生成 Schema 和描述。
   - 显式声明参数类型，确保 LangChain 能正确解析多参数。

Step 3: 构建 ReAct 提示词 (Prompt Engineering)
   - 定义中文优化的 ReAct 模板，强制模型按 "思考->行动->观察" 循环。
   - 明确工具调用格式，减少幻觉。
        - 强制模型在 Action 阶段输出 ```json {...}``` 代码块。
        - 格式：{"action": "tool_name", "action_input": {"param1": "val1", ...}}

Step 4: 组装 Agent 核心 (LCEL)
   - 初始化 ChatTongyi (qwen-turbo) 模型 (替代旧版 LLM)。
   - 注入 ReActJsonSingleInputOutputParser 解析器。
   - 调用 create_react_agent(llm, tools, prompt) 构建代理。
   - 实例化 AgentExecutor，开启详细日志和自动容错。

Step 5: 任务执行循环
   - 接收用户自然语言指令。
   - Agent 自动解析意图 -> 调用适配后的工具 -> 执行类方法 -> 获取结果。
   - 若需多步操作，Agent 会自动规划序列。
   - 输出最终处理结果。

【业务流程图解】
  用户指令 --> Agent 意图识别
      ├── (若需分析) --> [@tool 适配层] --> [TextAnalysisTool.run()] --> 统计 + 情感打分
      ├── (若需转换) --> [@tool 适配层] --> [DataConversionTool.run()] --> JSON/CSV 字符串处理
      └── (若需编辑) --> [@tool 适配层] --> [TextProcessingTool.run()] --> 字符串操作
      ↓
  收集工具返回结果 --> LLM 综合总结 --> 输出最终答案

================================================================================
"""

from langchain.agents import Tool, AgentExecutor, create_react_agent
# 【关键升级】引入专用 JSON 解析器，解决多参数传递问题
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain_core.prompts import PromptTemplate
# 【关键升级】使用 Chat 模型替代老式 LLM
from langchain_community.chat_models import ChatTongyi
import re
import json
from typing import List, Union, Dict, Any, Optional
import os
import dashscope
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. 环境配置
# -----------------------------------------------------------------------------
load_dotenv()
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

# -----------------------------------------------------------------------------
# 2. 保留 V1 特色的自定义工具类 (核心逻辑未变)
# -----------------------------------------------------------------------------

class TextAnalysisTool:
    """V1 风格：手动定义的文本分析类"""
    def __init__(self):
        self.name = "文本分析"
        self.description = "分析文本内容，提取字数、字符数和情感倾向"
    
    def run(self, text: str) -> str:
        # 【保留 V1 逻辑】简单的 split 计数和关键词匹配
        if "\\n" in text:
            text = text.replace("\\n", "\n")
        word_count = len(text.split())
        char_count = len(text)
        line_count = len(text.splitlines())
        
        positive_words = ["好", "优秀", "喜欢", "快乐", "成功", "美好"]
        negative_words = ["差", "糟糕", "讨厌", "悲伤", "失败", "痛苦"]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        sentiment = "积极" if positive_count > negative_count else "消极" if negative_count > positive_count else "中性"
        
        return f"文本分析结果:\n- 字数: {word_count}\n- 字符数: {char_count}\n- 行数: {line_count}\n- 情感倾向: {sentiment}"

class DataConversionTool:
    """V1 风格：手动定义的数据转换类"""
    def __init__(self):
        self.name = "数据转换"
        self.description = "在不同数据格式之间转换，如 JSON、CSV 等"
    
    def run(self, input_data: str, input_format: str, output_format: str) -> str:
        # 【保留 V1 逻辑】基础的 JSON/CSV 字符串处理
        try:
            if input_format.lower() == "json" and output_format.lower() == "csv":
                data = json.loads(input_data)
                if isinstance(data, list):
                    if not data: return "空数据"
                    headers = set()
                    for item in data: headers.update(item.keys())
                    headers = list(headers)
                    csv = ",".join(headers) + "\n"
                    for item in data:
                        row = [str(item.get(header, "")) for header in headers]
                        csv += ",".join(row) + "\n"
                    return csv
                else: return "输入数据必须是 JSON 数组"
            
            elif input_format.lower() == "csv" and output_format.lower() == "json":
                if "\\n" in input_data:
                    input_data = input_data.replace("\\n", "\n")
                input_data = input_data.replace("，", ",")
                lines = input_data.strip().split("\n")
                if len(lines) < 2: return "CSV 数据至少需要标题行和数据行"
                headers = lines[0].split(",")
                result = []
                for line in lines[1:]:
                    values = line.split(",")
                    if len(values) != len(headers): continue
                    item = {header: values[i] for i, header in enumerate(headers)}
                    result.append(item)
                return json.dumps(result, ensure_ascii=False, indent=2)
            else:
                return f"不支持的转换：{input_format} -> {output_format}"
        except Exception as e:
            return f"转换失败：{str(e)}"

class TextProcessingTool:
    """V1 风格：手动定义的文本处理类"""
    def __init__(self):
        self.name = "文本处理"
        self.description = "处理文本内容，如查找、替换、统计等"
    
    def run(self, operation: str, content: str, search_text: Optional[str] = None, new_text: Optional[str] = None) -> str:
        # 【保留 V1 逻辑】基于 if/else 的操作分发
        # 注意：为了适配新 API 的多参数，这里稍微调整了参数签名以支持 kwargs 解包
        # 原 V1 使用 **kwargs，这里显式化以便 LangChain 更好地识别 Schema
        
        op_map = {
            "count_lines": "count_lines",
            "find_text": "find_text",
            "replace_text": "replace_text",
            "统计": "count_lines",
            "查找": "find_text",
            "替换": "replace_text",
        }
        operation = op_map.get(operation, operation)

        if "\\n" in content:
            content = content.replace("\\n", "\n")

        if operation == "count_lines":
            return f"文本共有 {len(content.splitlines())} 行"
        
        elif operation == "find_text":
            if not search_text: return "请提供要查找的文本"
            lines = content.splitlines()
            matches = [f"第 {i+1} 行：{line}" for i, line in enumerate(lines) if search_text in line]
            if matches:
                return f"找到 {len(matches)} 处匹配:\n" + "\n".join(matches)
            else:
                return f"未找到文本 '{search_text}'"
        
        elif operation == "replace_text":
            # 兼容 V1 的参数名 old_text/new_text 或新版的 search_text/replace_text
            # 这里为了演示兼容性，优先使用传入的 search_text，如果没有则尝试兼容旧逻辑
            target = search_text if search_text else new_text # 简单兼容逻辑
            replacement = new_text if new_text else "" 
            
            # 如果用户传的是 old_text/new_text (通过 kwargs 动态传入的情况)
            # 由于 @tool 装饰器通常处理显式参数，这里我们假设调用方会匹配参数名
            # 在实际 V1 中是 **kwargs，这里为了配合新 API 的自动解包，我们假设参数名已对齐
            
            if not target: return "请提供要替换的文本"
            
            # 修正：为了严格保留 V1 逻辑，我们假设外部调用会传入正确的参数名
            # 如果通过 JSON 解包，参数名必须匹配函数签名。
            # 这里我们做一个小适配，允许 search_text 充当 old_text
            old_val = search_text 
            new_val = new_text
            
            if not old_val: return "请提供要替换的文本 (search_text)"
            if new_val is None: return "请提供替换后的文本 (new_text)"

            new_content = content.replace(old_val, new_val)
            count = content.count(old_val)
            return f"替换完成，共替换 {count} 处。\n新内容:\n{new_content}"
        
        else:
            return f"不支持的操作：{operation}"

# -----------------------------------------------------------------------------
# 3. 工具注册 (关键变化：用 @tool 包裹 V1 的类方法)
# -----------------------------------------------------------------------------
from langchain_core.tools import tool

# 实例化 V1 的类
v1_text_analysis = TextAnalysisTool()
v1_data_conversion = DataConversionTool()
v1_text_processing = TextProcessingTool()

# 【升级点】使用 @tool 装饰器将 V1 的类方法转换为 LangChain 标准工具
# 这样既保留了 V1 的类封装特色，又获得了 V3 的自动 Schema 推导和多参数支持
@tool("文本分析")
def use_text_analysis(text: str) -> str:
    """分析文本内容，提取字数、字符数和情感倾向。参数：text (完整文本)"""
    return v1_text_analysis.run(text)

@tool("数据转换")
def use_data_conversion(input_data: str, input_format: str, output_format: str) -> str:
    """在不同数据格式之间转换，如 JSON、CSV 等。参数：input_data, input_format, output_format"""
    return v1_data_conversion.run(input_data, input_format, output_format)

@tool("文本处理")
def use_text_processing(operation: str, content: str, search_text: Optional[str] = None, new_text: Optional[str] = None) -> str:
    """处理文本内容，如查找、替换、统计等。参数：operation, content, search_text (查找目标), new_text (替换内容)"""
    # 适配 V1 的 replace_text 逻辑需要的参数名映射
    if operation == "replace_text":
        # V1 类内部逻辑可能需要 old_text/new_text，但这里我们直接调用修改后的 run 或者适配参数
        # 为了保持 V1 类的纯粹性，我们在 wrapper 层做参数映射
        return v1_text_processing.run(operation, content, search_text=search_text, new_text=new_text)
    return v1_text_processing.run(operation, content, search_text=search_text, new_text=new_text)

tools = [use_text_analysis, use_data_conversion, use_text_processing]

# -----------------------------------------------------------------------------
# 4. 提示词模板 (升级为标准 JSON ReAct 格式)
# -----------------------------------------------------------------------------
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
# 5. 构建 Agent 执行器 (升级配置)
# -----------------------------------------------------------------------------
def create_tool_chain():
    # 【升级】使用 ChatTongyi
    llm = ChatTongyi(model_name="qwen-turbo", dashscope_api_key=api_key)
    
    # 【升级】创建 Agent 并指定 JSON 解析器
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        output_parser=ReActJsonSingleInputOutputParser(),
    )
    
    # 【升级】增加错误处理和迭代限制
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,  # 开启自动重试
        max_iterations=15            # 防止死循环
    )
    
    return agent_executor

def process_task(task_description):
    try:
        agent_executor = create_tool_chain()
        response = agent_executor.invoke({"input": task_description})
        return response["output"]
    except Exception as e:
        return f"处理任务时出错：{str(e)}"

# -----------------------------------------------------------------------------
# 6. 示例用法 (与原版保持一致)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== V1 升级版 Agent (保留类结构 + 新 API) ===\n")
    
    # 任务 1: 复合任务 (分析 + 统计)
    task1 = "分析这段文本的情感倾向并统计行数：'这个产品非常好用，我很喜欢它的设计，体验棒！\n价格也很合理，强烈推荐。\n但是物流稍微有点慢，让人有点失望。'"
    print(f"[任务 1] {task1}")
    print("结果:", process_task(task1))
    
    print("\n" + "="*60 + "\n")
    
    # # 任务 2: 数据转换 (CSV -> JSON，包含逗号测试)
    task2 = "将以下 CSV 数据转换为 JSON 格式：'name,role,comment\n\"张三，经理\",开发,\"代码质量高，态度好\"\n李四,测试,\"发现了很多 Bug，工作细致\"'"
    print(f"[任务 2] {task2}")
    print("结果:", process_task(task2))
    
    # 任务 3: 文本编辑 (查找与替换)
    task3 = "帮我把下面这段话里的'bug'全部替换为'特性'，并告诉我替换了几处：'这个版本修复了 3 个严重 bug，但引入了 2 个新 bug。总体还是有 bug。'"
    print(f"[任务 3] {task3}")
    print("结果:", process_task(task3))
