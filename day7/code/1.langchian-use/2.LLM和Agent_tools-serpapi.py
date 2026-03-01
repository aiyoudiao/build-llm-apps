import os
from langchain.agents import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Tongyi  # 导入通义千问 Tongyi 模型
import dashscope
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. 环境变量与 API Key 配置
# -----------------------------------------------------------------------------
# 加载 .env 文件中的环境变量 (如 DASHSCOPE_API_KEY, SERPAPI_API_KEY)
load_dotenv()

# 获取通义千问 (DashScope) 的 API Key
api_key = os.environ.get('DASHSCOPE_API_KEY')
# 设置 dashscope 全局 API Key，确保底层 SDK 能正确认证
dashscope.api_key = api_key

# -----------------------------------------------------------------------------
# 2. 初始化大语言模型 (LLM)
# -----------------------------------------------------------------------------
# 实例化通义千问模型
# model_name="qwen-turbo": 指定使用 Qwen-Turbo 模型 (速度快、成本低)
# dashscope_api_key: 显式传入 key，优先于全局设置
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=api_key)

# -----------------------------------------------------------------------------
# 3. 加载外部工具 (Tools)
# -----------------------------------------------------------------------------
# 加载 serpapi 工具，用于实时搜索互联网信息
# 注意：确保已安装 `pip install google-search-results` 且环境变量中配置了 SERPAPI_API_KEY
tools = load_tools(["serpapi"])

# -----------------------------------------------------------------------------
# 4. 定义 ReAct 模式的 Prompt 模板
# -----------------------------------------------------------------------------
"""
ReAct (Reasoning + Acting) 框架核心逻辑：
模型需要遵循特定的思考 - 行动 - 观察循环。
此模板强制模型按照固定格式输出，以便 AgentExecutor 能够正则匹配并解析出 Action。

变量说明：
- {tools}: 工具的名称和描述列表
- {tool_names}: 工具名称的逗号分隔字符串 (用于限制 Action 的选择范围)
- {input}: 用户的具体问题
- {agent_scratchpad}: 历史记录，包含之前的 Thought/Action/Observation 循环，由框架自动填充
"""
template = '''尽你所能回答以下问题。你可以使用以下工具：

{tools}

请使用以下格式：

Question: 你必须回答的输入问题
Thought: 你应该时刻思考该做什么
Action: 采取的行动，必须是 [{tool_names}] 之一
Action Input: 行动的输入
Observation: 行动的结果
... (Thought/Action/Action Input/Observation 这个过程可以重复 N 次)
Thought: 我现在知道最终答案了
Final Answer: 对原始输入问题的最终答案

开始！

Question: {input}
Thought:{agent_scratchpad}'''

# 打印模板以便调试，确认变量占位符是否正确
print(f"template \n{template}")

# 将字符串模板转换为 LangChain 的 PromptTemplate 对象
prompt = PromptTemplate.from_template(template)

# -----------------------------------------------------------------------------
# 5. 构建 Agent 和 执行器 (Executor)
# -----------------------------------------------------------------------------
# 创建 ReAct 代理
# 这是 LangChain 0.2+ 推荐的新方式：将 LLM、Tools 和 Prompt 绑定成一个 Runnable
agent = create_react_agent(llm, tools, prompt)

# 创建 AgentExecutor
# 这是实际运行循环的控制器：
# - agent: 上面创建的代理逻辑
# - tools: 可供调用的工具列表
# - verbose=True: 在控制台打印详细的执行步骤 (Thought, Action, Observation)，便于调试
# - handle_parsing_errors=True: 当模型输出的格式不符合 Prompt 要求时，自动提示模型重试，而不是直接报错崩溃
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# -----------------------------------------------------------------------------
# 6. 执行任务
# -----------------------------------------------------------------------------
# 调用 Agent 执行任务
# 新版 LangChain 使用 .invoke() 方法，输入通常为字典 {"input": "..."}
# 流程：用户提问 -> LLM 思考 -> 调用 SerpAPI -> 获取结果 -> LLM 总结 -> 输出最终答案
response = agent_executor.invoke({"input": "今天是几月几号？历史上的今天有哪些名人出生"})

# 打印最终结果 (invoke 返回的是一个字典，包含 'output' 键)
print("\n--- 最终回答 ---")
print(response.get("output"))

# 调试：确认 SerpAPI 的 Key 是否成功加载
print(f"\nSERPAPI_API_KEY 是否存在：{'Yes' if os.getenv('SERPAPI_API_KEY') else 'No'}")
