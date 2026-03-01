"""
================================================================================
LangChain 新版多功能数据处理链 (LCEL 手动编排 + 确定性工作流)
================================================================================

【核心功能】
一个高效、确定的数据处理流水线，专注于执行预设的文本分析、
数据格式转换及文本编辑任务。与 Agent 不同，本版本不依赖大模型进行决策，
而是由 Python 代码精确控制执行路径，展示了 LangChain Expression Language (LCEL)
在构建“非生成式”后端服务中的强大能力。

主要能力：
1. 智能文本分析：统计字数/行数，并基于基础关键词匹配算法判断情感倾向。
2. 稳健的数据转换：支持 JSON <-> CSV 互转，保留原有的字符串处理逻辑。
3. 高级文本编辑：支持查找定位、批量替换及行统计。
4. 混合编排能力：支持在确定性流程中灵活嵌入 LLM 步骤 (如数据总结)。

【技术优化】
1. 架构升级：完全遵循 LangChain 0.2+ LCEL 标准，使用 RunnableLambda 和 RunnableSequence。
2. 路由模式：
   - 保留 V2 的 class Tool 定义。
   - 构建手动路由字典 (Router Pattern)，由代码决定调用哪个工具。
   - 彻底消除 Agent 的“幻觉”风险和重试延迟。
3. 类型安全：
   - 显式定义输入校验逻辑，替代隐式的 lambda 表达式。
   - 提供清晰的错误提示，便于调试。
4. 高级流水线：
   - 演示如何构建 `Prepare -> Process -> LLM_Summary` 的串行链条。
   - 展示 LCEL 如何将“确定性代码”与“生成式 AI”无缝融合。

【执行流程步骤】
--------------------------------------------------------------------------------
Step 1: 实例化经典工具类 (Legacy Classes)
   - 初始化 TextAnalysisTool, DataConversionTool, TextProcessingTool 实例。
   - 保留原有的 run() 方法逻辑和内部算法。

Step 2: 构建 LCEL 节点 (Runnable Nodes)
   - 定义具名函数 (如 run_text_analysis_node) 包装类方法。
   - 添加参数存在性校验，确保输入数据完整。
   - 使用 RunnableLambda 将函数转换为可执行单元。

Step 3: 构建手动路由器 (Manual Router)
   - 创建工具字典：{"文本分析": runnable_a, "数据转换": runnable_b, ...}。
   - 定义 dispatch 函数：根据传入的 task_type 字符串，直接索引并 invoke 对应节点。
   - 流程完全由程序员控制，无 LLM 介入决策。

Step 4: (可选) 构建高级混合流水线 (Advanced Pipeline)
   - 使用 RunnableSequence 串联多个步骤。
   - 示例流程：原始 CSV -> 数据转换工具 -> LLM 总结摘要。
   - 展示 LCEL 在复杂 ETL 场景下的灵活性。

Step 5: 任务执行循环
   - 接收结构化指令 (task_type + params)。
   - 路由器直接分发到对应工具 -> 执行类方法 -> 获取结果。
   - (若启用流水线) 结果自动流入下一步骤或 LLM。
   - 输出最终处理结果。

【业务流程图解】
  用户/系统指令 (含 task_type) --> 手动路由器 (Python Code)
      ├── (Case: 文本分析) --> [RunnableLambda] --> [TextAnalysisTool.run()] --> 统计 + 情感打分
      ├── (Case: 数据转换) --> [RunnableLambda] --> [DataConversionTool.run()] --> JSON/CSV 字符串处理
      └── (Case: 编辑/流水线) --> [RunnableSequence] --> [工具] --> [LLM 总结] --> 最终输出
      ↓
  直接返回结果 (无 Agent 思考循环，低延迟，高确定)

================================================================================
"""

import json
import os
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv

# LangChain Core & Community
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
# 【升级】使用 Chat 模型，虽然 LCEL 主流程不依赖它，但展示混合能力
from langchain_community.chat_models import ChatTongyi
import dashscope

# -----------------------------------------------------------------------------
# 1. 环境配置
# -----------------------------------------------------------------------------
load_dotenv()
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

# -----------------------------------------------------------------------------
# 2. 保留 V2 特色的自定义工具类 (核心逻辑未变)
# -----------------------------------------------------------------------------

class TextAnalysisTool:
    """V2 风格：手动定义的文本分析类"""
    def __init__(self):
        self.name = "文本分析"
        self.description = "分析文本内容，提取字数、字符数和情感倾向"
    
    def run(self, text: str) -> str:
        word_count = len(text.split())
        char_count = len(text)
        positive_words = ["好", "优秀", "喜欢", "快乐", "成功", "美好"]
        negative_words = ["差", "糟糕", "讨厌", "悲伤", "失败", "痛苦"]
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        sentiment = "积极" if positive_count > negative_count else "消极" if negative_count > positive_count else "中性"
        return f"文本分析结果:\n- 字数: {word_count}\n- 字符数: {char_count}\n- 情感倾向: {sentiment}"

class DataConversionTool:
    """V2 风格：手动定义的数据转换类"""
    def __init__(self):
        self.name = "数据转换"
        self.description = "在不同数据格式之间转换，如 JSON、CSV 等"
    
    def run(self, input_data: str, input_format: str, output_format: str) -> str:
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
    """V2 风格：手动定义的文本处理类"""
    def __init__(self):
        self.name = "文本处理"
        self.description = "处理文本内容，如查找、替换、统计等"
    
    def run(self, operation: str, content: str, **kwargs) -> str:
        if operation == "count_lines":
            return f"文本共有 {len(content.splitlines())} 行"
        
        elif operation == "find_text":
            search_text = kwargs.get("search_text", "")
            if not search_text: return "请提供要查找的文本"
            lines = content.splitlines()
            matches = [f"第 {i+1} 行：{line}" for i, line in enumerate(lines) if search_text in line]
            if matches:
                return f"找到 {len(matches)} 处匹配:\n" + "\n".join(matches)
            else:
                return f"未找到文本 '{search_text}'"
        
        elif operation == "replace_text":
            old_text = kwargs.get("old_text", "")
            new_text = kwargs.get("new_text", "")
            if not old_text: return "请提供要替换的文本"
            new_content = content.replace(old_text, new_text)
            count = content.count(old_text)
            return f"替换完成，共替换 {count} 处。\n新内容:\n{new_content}"
        
        else:
            return f"不支持的操作：{operation}"

# -----------------------------------------------------------------------------
# 3. 构建 LCEL 工具路由 (保留 V2 的手动调度特色)
# -----------------------------------------------------------------------------

# 实例化 V2 的类
v2_text_analysis = TextAnalysisTool()
v2_data_conversion = DataConversionTool()
v2_text_processing = TextProcessingTool()

# 【升级点】定义更健壮的 RunnableLambda
# 使用显式的函数定义代替简单的 lambda，便于调试和添加类型提示
def run_text_analysis_node(inputs: Dict[str, Any]) -> str:
    if "text" not in inputs:
        raise ValueError("Missing 'text' parameter for Text Analysis")
    return v2_text_analysis.run(inputs["text"])

def run_data_conversion_node(inputs: Dict[str, Any]) -> str:
    required = ["input_data", "input_format", "output_format"]
    if not all(k in inputs for k in required):
        raise ValueError(f"Missing parameters for Data Conversion. Required: {required}")
    return v2_data_conversion.run(
        inputs["input_data"], 
        inputs["input_format"], 
        inputs["output_format"]
    )

def run_text_processing_node(inputs: Dict[str, Any]) -> str:
    if "operation" not in inputs or "content" not in inputs:
        raise ValueError("Missing 'operation' or 'content' for Text Processing")
    # 提取额外的 kwargs
    extra_params = {k: v for k, v in inputs.items() if k not in ["operation", "content"]}
    return v2_text_processing.run(inputs["operation"], inputs["content"], **extra_params)

# 构建工具路由字典 (Router Pattern)
# V2 的核心特色：程序员明确控制哪个请求去哪个工具
tool_router = {
    "文本分析": RunnableLambda(run_text_analysis_node),
    "数据转换": RunnableLambda(run_data_conversion_node),
    "文本处理": RunnableLambda(run_text_processing_node),
}

def manual_dispatch_chain(task_type: str, params: Dict[str, Any]) -> str:
    """
    V2 风格的手动调度器
    这是 LCEL 的“硬编码”用法，适合业务逻辑固定的场景
    """
    if task_type not in tool_router:
        return f"错误：未知的工具类型 '{task_type}'。可用工具：{list(tool_router.keys())}"
    
    # 使用 .invoke() 执行对应的 Runnable
    return tool_router[task_type].invoke(params)

# -----------------------------------------------------------------------------
# 4. 【新增】高级 LCEL 组合示例 (展示 LCEL 的真正威力)
# -----------------------------------------------------------------------------

# 场景：构建一个自动化的“数据处理流水线”
# 流程：接收 CSV -> 转为 JSON -> (可选) 让 LLM 总结数据 -> 输出
# 这展示了 LCEL 如何将“确定性代码”与“AI 能力”无缝结合

def build_advanced_pipeline():
    # 1. 定义一个预处理步骤 (纯代码)
    prepare_data = RunnableLambda(lambda x: {
        "input_data": x["csv_raw"],
        "input_format": "csv",
        "output_format": "json"
    })
    
    # 2. 调用转换工具 (确定性代码)
    convert_step = RunnableLambda(run_data_conversion_node)
    
    # 3. (可选) 让 LLM 对转换后的 JSON 进行简单总结
    # 注意：这里我们才真正用到 ChatTongyi，体现 LCEL 的混合编排能力
    llm = ChatTongyi(model_name="qwen-turbo", dashscope_api_key=api_key)
    summary_prompt = RunnableLambda(lambda json_data: f"请简要总结以下 JSON 数据中的关键信息（不超过 50 字）：\n{json_data}")
    llm_step = summary_prompt | llm | StrOutputParser()
    
    # 4. 构建完整序列
    # 序列：准备数据 -> 转换 -> 并行输出 (原始 JSON + LLM 总结)
    # 这里为了演示简单，我们做成串行：转换 -> 总结
    pipeline = prepare_data | convert_step | summary_prompt | llm | StrOutputParser()
    
    return pipeline

# -----------------------------------------------------------------------------
# 5. 示例用法
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== V2 升级版 LCEL 链 (手动调度 + 高级编排) ===\n")
    
    # --- 基础测试：保留 V2 原有的手动调度风格 ---
    print("[基础测试 1] 文本分析")
    result1 = manual_dispatch_chain("文本分析", {"text": "这个产品非常好用，我很喜欢它的设计，使用体验非常棒！"})
    print(result1)
    print("-" * 30)

    print("[基础测试 2] 数据格式转换 (CSV -> JSON)")
    csv_data = "name,age,comment\n张三，25,这个产品很好\n李四，30,服务态度差\n王五，28,性价比高"
    result2 = manual_dispatch_chain("数据转换", {
        "input_data": csv_data, 
        "input_format": "csv", 
        "output_format": "json"
    })
    print(result2)
    print("-" * 30)

    print("[基础测试 3] 文本处理 (查找)")
    text = "第一行内容\n第二行包含关键词\n第三行内容"
    result3 = manual_dispatch_chain("文本处理", {
        "operation": "find_text", 
        "content": text, 
        "search_text": "关键词"
    })
    print(result3)
    print("-" * 30)

    # --- 进阶测试：展示 LCEL 的链式组合能力 (V2 的潜力) ---
    print("[进阶测试 4] LCEL 自动化流水线 (CSV -> JSON -> LLM 总结)")
    print("正在执行：原始 CSV -> 转换工具 -> LLM 总结...")
    
    try:
        pipeline = build_advanced_pipeline()
        # 输入原始 CSV
        raw_csv = "product,score,review\n手机，9,非常好\n电脑，7,还可以\n耳机，4,噪音大"
        
        #  invoke 整个链条
        final_summary = pipeline.invoke({"csv_raw": raw_csv})
        
        print(f"\n✅ 流水线最终输出 (LLM 总结):\n{final_summary}")
    except Exception as e:
        print(f"\n❌ 流水线执行出错 (可能是网络或 API Key 问题): {e}")
    
    print("\n" + "="*60)
    print("对比总结:")
    print("- V2 的核心是 '手动路由' (manual_dispatch_chain)，由代码决定流程。")
    print("- V2 的优势是 '可控性'，适合固定业务逻辑。")
    print("- 通过 LCEL，V2 也可以轻松嵌入 AI 步骤 (如进阶测试 4)，实现混合编排。")
    print("="*60)
