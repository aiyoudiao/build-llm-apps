"""
**LangChain ReAct Agent 特斯拉智能助手(LCEL 架构)**

【核心功能】
一个具备“自主决策”与“工具调用”能力的智能问答助手，专门用于
回答关于“特斯拉”公司的定制化问题。它解决了大模型常见的“幻觉”问题，确保
回答基于真实的内部数据。

主要能力：
1. 意图识别与路由：自动分析用户问题，判断是需要查具体数据还是查宏观文档。
2. 动态工具调用：
   - 针对车型查询 (如 "Model 3 价格")：调用本地字典数据库，返回精确数值。
   - 针对公司咨询 (如 "自动驾驶技术")：调用内部文档检索工具，结合 LLM 总结回答。
3. ReAct 多步推理：支持 "思考 -> 行动 -> 观察 -> 再思考" 的循环，处理复杂问题。
4. 沉浸式交互：内置打字机流式输出效果，提升用户体验。

【技术特性】
1. 架构升级：使用 `create_react_agent` 替代旧版 `LLMSingleActionAgent`，符合
   LangChain 0.2+ 标准。
2. 鲁棒解析：利用内置 ReAct 解析器，彻底移除了脆弱的手写正则表达式匹配逻辑。
3. 自定义控制：保留自定义 Prompt 模板，精确控制 AI 的思考格式和行为风格。
4. 容错机制：开启 `handle_parsing_errors`，当模型输出格式微偏时自动重试，防止崩溃。

【执行流程步骤】
--------------------------------------------------------------------------------
Step 1: 初始化与工具注册
   - 加载通义千问 (DashScope) API Key。
   - 实例化 `TeslaDataSource`，封装“产品查询”和“公司信息检索”两个业务方法。
   - 将方法注册为 LangChain `Tool` 对象，并编写详细的 `description` (这是 AI
     选择工具的唯一依据)。

Step 2: 构建 ReAct 提示词 (Prompt Engineering)
   - 定义 `AGENT_TMPL` 模板，强制 AI 遵循标准格式：
     Question -> Thought -> Action -> Observation -> Final Answer。
   - 预留 `{agent_scratchpad}` 供框架自动填充历史思考记录。

Step 3: 组装 Agent 核心 (LCEL)
   - 实例化 `Tongyi` 模型。
   - 调用 `create_react_agent(llm, tools, prompt)` 构建代理逻辑。
   - 创建 `AgentExecutor` 执行器，开启详细日志 (`verbose`) 和自动容错。

Step 4: 运行交互循环 (Runtime Loop)
   - 监听用户输入。
   - 调用 `agent_executor.invoke({"input": ...})` 启动推理。
   - [内部自动流转]:
     a. LLM 生成思考与行动指令。
     b. Executor 解析指令，执行对应的 Python 函数 (查库/查文档)。
     c. 获取结果作为 "Observation" 反馈给 LLM。
     d. LLM 综合所有信息，生成 "Final Answer"。

Step 5: 格式化输出
   - 提取最终答案字符串。
   - 通过 `output_response` 函数实现逐字打印 (打字机效果) 和自动换行。
   - 等待下一次用户输入。

【业务流程图解】
  用户提问 --> Agent 分析意图 
      ├── (若问车型) --> 调用 [查询产品名称] 工具 --> 返回精确价格/描述
      └── (若问公司) --> 调用 [公司相关信息] 工具 --> 检索文档并 LLM 总结
      ↓
  收集所有观察结果 --> LLM 生成最终答案 --> 打字机效果输出 --> 等待下一问

"""

import os
import re
import time
import textwrap
from typing import List, Optional

# LangChain 核心组件
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatTongyi  # 导入通义千问 ChatTongyi 模型
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. 全局配置与工具函数
# -----------------------------------------------------------------------------
# 加载 .env 文件
load_dotenv()

# 注意：实际项目中请将 API Key 放入环境变量，不要硬编码在代码中
# 获取并设置 DashScope (通义千问) API Key
DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY')

def output_response(response: str) -> None:
    """
    以打字机效果逐字打印响应，增加交互体验。
    每行限制 60 字符，每个字符间隔 0.1 秒。
    """
    if not response:
        return
    
    # 文本换行处理
    lines = textwrap.wrap(response, width=60)
    
    for line in lines:
        words = line.split()
        for i, word in enumerate(words):
            for char in word:
                print(char, end="", flush=True)
                time.sleep(0.05)  # 稍微加快一点速度，0.05s 更流畅
            # 单词间空格
            if i < len(words) - 1:
                print(" ", end="", flush=True)
        print()  # 换行
    
    print("-" * 60)

# -----------------------------------------------------------------------------
# 2. 数据源与工具定义 (Tools)
# -----------------------------------------------------------------------------

class TeslaDataSource:
    """
    模拟特斯拉公司的数据源，提供具体的业务逻辑。
    这些方法将被封装为 Agent 可调用的 Tools。
    """
    
    def __init__(self):
        # 模拟产品数据库
        self.product_info = {
            "Model 3": "具有简洁、动感的外观设计，流线型车身和现代化前脸。定价 23.19-33.19 万",
            "Model Y": "在外观上与 Model 3 相似，但采用了更高的车身和更大的后备箱空间。定价 26.39-36.39 万",
            "Model X": "拥有独特的翅子门设计和更加大胆的外观风格。定价 89.89-105.89 万",
            "Model S": "特斯拉的旗舰轿车，拥有极致的性能和续航里程。定价 68.49-82.89 万"
        }
        
        # 模拟公司介绍文档
        self.company_context = """
        特斯拉最知名的产品是电动汽车，其中包括 Model S、Model 3、Model X 和 Model Y 等多款车型。
        特斯拉以其技术创新、高性能和领先的自动驾驶技术而闻名。公司不断推动自动驾驶技术的研发，
        并在车辆中引入了各种驾驶辅助功能，如自动紧急制动、自适应巡航控制和车道保持辅助等。
        特斯拉还涉足太阳能产品和能源存储解决方案。
        """

    def find_product_description(self, product_name: str) -> str:
        """
        工具 1：查询具体车型的描述和价格。
        Args:
            product_name: 车型名称 (如 "Model 3", "Model Y")
        Returns:
            车型的具体描述字符串
        """
        print(f"[Tool Call] 正在查询产品: {product_name}...")
        # 简单的模糊匹配优化 (可选)
        product_name = product_name.strip()
        result = self.product_info.get(product_name, "没有找到这个产品，请尝试 Model 3, Model Y, Model X 或 Model S")
        return result

    def find_company_info(self, query: str) -> str:
        """
        工具 2：查询公司宏观信息。
        内部使用 LLM 对固定上下文进行总结，以回答具体问题。
        Args:
            query: 用户关于公司的具体问题
        Returns:
            基于上下文的回答
        """
        print(f"[Tool Call] 正在检索公司信息并分析: {query}...")
        
        # 初始化一个临时的 LLM 实例用于内部 RAG (复用全局 Key)
        # 注意：这里为了避免循环依赖，我们直接调用 dashscope 或者复用外部传入的 llm
        # 为简化演示，这里假设我们可以直接调用外部 llm，或者在此处实例化一个轻量级调用
        # 由于类初始化时没有 llm，我们在主流程中注入，或者此处简单模拟
        # 为了保持代码独立性，这里我们简单返回上下文 + 问题，让主 Agent 去处理？
        # 不，按照原逻辑，这个工具内部应该调用 LLM 生成答案。
        # 我们需要在主程序中把 llm 传进来，或者在这里实例化。
        # 修正：在 __main__ 中实例化 llm 并传给此类，或者此类内部实例化。
        # 这里采用内部实例化以保持工具类的独立性，但实际生产建议依赖注入。
        temp_llm = ChatTongyi(model_name="qwen-turbo", dashscope_api_key=DASHSCOPE_API_KEY)
        
        context_qa_template = """
        根据以下提供的信息，回答用户的问题。如果信息中不包含答案，请直接说不知道。
        信息：{context}
        问题：{query}
        回答：
        """
        prompt = context_qa_template.format(context=self.company_context, query=query)
        return temp_llm.invoke(prompt)

# 初始化工具数据源
tesla_source = TeslaDataSource()

# 定义 LangChain Tools
# name: 工具的唯一标识 (Agent 会根据这个名字决定调用哪个)
# func: 实际执行的函数
# description: 工具的详细描述，Agent 依靠这个描述来判断何时使用该工具 (非常重要)
tools = [
    Tool(
        name="查询产品名称",
        func=tesla_source.find_product_description,
        description="当用户询问具体车型 (如 Model 3, Model Y) 的价格、外观或配置时使用。输入必须是具体的车型名称。"
    ),
    Tool(
        name="公司相关信息",
        func=tesla_source.find_company_info,
        description="当用户询问特斯拉公司的整体情况、历史、自动驾驶技术概况或非具体车型的一般性问题时使用。输入是用户的具体问题。"
    )
]

# -----------------------------------------------------------------------------
# 3. 定义自定义 ReAct Prompt 模板
# -----------------------------------------------------------------------------

# 新版 create_react_agent 允许传入自定义 prompt
# 必须包含 {input}, {agent_scratchpad}, {tool_names}, {tools} 这些变量
AGENT_TMPL = """你是一个智能助手，负责回答关于特斯拉的问题。你可以使用以下工具：

{tools}

请严格按照以下格式进行思考和行动，不要跳过任何步骤：

---
Question: 需要回答的输入问题
Thought: 思考当前应该做什么，是否需要使用工具？
Action: 从 [{tool_names}] 中选择一个工具
Action Input: 工具的输入参数
Observation: 工具返回的结果
... (Thought/Action/Action Input/Observation 可以重复多次)
Thought: 我现在已经收集到足够的信息，可以回答最终问题了
Final Answer: 对原始输入问题的最终完整回答
---

开始！

Question: {input}
{agent_scratchpad}
"""

react_prompt = PromptTemplate.from_template(AGENT_TMPL)

# -----------------------------------------------------------------------------
# 4. 构建 Agent 和 执行器 (Executor)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 初始化 LLM
    llm = ChatTongyi(model_name="qwen-turbo", dashscope_api_key=DASHSCOPE_API_KEY)
    
    # 检查工具列表
    tool_names = [t.name for t in tools]
    print(f"已加载工具：{tool_names}")

    # 【核心变化】使用 create_react_agent 构建 Agent
    # 它会自动处理：
    # 1. 将 prompt, llm, tools 绑定
    # 2. 内置强大的输出解析器 (不再需要手写 CustomOutputParser 和正则)
    # 3. 处理中间步骤的状态管理
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt
    )

    # 创建 AgentExecutor
    # handle_parsing_errors=True: 当 LLM 输出格式略有偏差时，自动提示重试，提高稳定性
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # 打印详细的执行过程 (Thought, Action, Observation)
        handle_parsing_errors=True
    )

    print("\n=== 特斯拉智能助手已启动 (按 Ctrl+C 退出) ===")
    
    # 主交互循环
    while True:
        try:
            user_input = input("\n请输入您的问题：").strip()
            if not user_input:
                continue
            
            # 执行 Agent
            # 新版 invoke 接收字典，返回字典 {'output': '...'}
            result = agent_executor.invoke({"input": user_input})
            
            # 提取并打印结果
            final_answer = result.get("output", "未生成回答")
            print("\n[助手回答]:")
            output_response(final_answer)
            
        except KeyboardInterrupt:
            print("\n\n程序已终止。")
            break
        except Exception as e:
            print(f"\n发生错误：{e}")
            # 在生产环境中，这里应该记录日志而不是直接打印堆栈
