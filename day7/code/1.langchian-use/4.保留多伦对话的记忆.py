"""
LangChain 新版对话链示例 (基于 LCEL 声明式链式表达语法 prompt | llm | parser)
------------------------------------
本脚本演示如何使用 LangChain 0.2+ 的新式 API 构建带有记忆功能的对话链。
替代了旧版的 ConversationChain 和 initialize_agent 模式。

核心组件：
1. RunnableWithMessageHistory: 包装器，自动处理消息历史的读取和写入。
2. ChatMessageHistory: 内存中的历史记录存储实现。
3. PromptTemplate: 定义对话的系统指令和输入格式。
"""

import os
import logging
# Parent run ... not found for run ... Treating as a root run 警告是 langchain-core 库中的一个已知日志问题，通常出现在使用 RunnableWithMessageHistory 时。这个警告本身不影响代码的功能和逻辑，但会干扰输出。
# Suppress the specific warning from langchain_core
logging.getLogger("langchain_core.tracers.base").setLevel(logging.ERROR)

from typing import Dict, List, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatTongyi  # 导入通义千问 ChatTongyi 模型
from langchain_core.messages import HumanMessage, AIMessage
import dashscope
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. 环境变量与 API Key 配置
# -----------------------------------------------------------------------------
# 加载 .env 文件
load_dotenv()

# 获取并设置 DashScope (通义千问) API Key
api_key = os.environ.get('DASHSCOPE_API_KEY')
if not api_key:
    raise ValueError("未找到 DASHSCOPE_API_KEY 环境变量，请检查 .env 文件或系统环境变量")
    
dashscope.api_key = api_key

# -----------------------------------------------------------------------------
# 2. 初始化大语言模型 (LLM)
# -----------------------------------------------------------------------------
# 实例化通义千问模型
# 注意：ChatTongyi 是为 ChatModel 设计的，而不是 LLM。
# 它会自动将 HumanMessage/AIMessage 转换为 DashScope 格式。
llm = ChatTongyi(model_name="qwen-turbo", dashscope_api_key=api_key)

# -----------------------------------------------------------------------------
# 3. 定义对话 Prompt 模板
# -----------------------------------------------------------------------------
"""
新版对话链通常使用 ChatPromptTemplate。
- system: 设定 AI 的人设和行为准则。
- MessagesPlaceholder: 一个特殊占位符，会自动被替换为历史消息列表 (Human/AI 对)。
- human: 当前用户的输入。
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好、乐于助人的 AI 助手。请用简洁自然的中文回答。"),
    MessagesPlaceholder(variable_name="history"), # 这里会自动填入历史对话
    ("human", "{input}")
])

# -----------------------------------------------------------------------------
# 4. 构建带有记忆的链 (Chain)
# -----------------------------------------------------------------------------

# 4.1 创建基础链 (不含历史)
# 将 prompt 和 llm 串联起来
basic_chain = prompt | llm

# 4.2 管理会话历史 (Memory Store)
# 在实际应用中，这里可以是 Redis, SQL 等持久化存储。
# 这里使用简单的内存字典：{ session_id: ChatMessageHistory 对象 }
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    根据 session_id 获取或创建历史记录对象。
    这是 RunnableWithMessageHistory 要求的回调函数。
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 4.3 包装为带历史的链
# RunnableWithMessageHistory 会自动：
# 1. 在调用前，从 store 获取历史消息注入到 prompt 的 'history' 变量中。
# 2. 在调用后，将新的 (用户问，AI 答) 对追加到 store 中。
conversation_chain = RunnableWithMessageHistory(
    basic_chain,
    get_session_history,
    input_messages_key="input",       # 用户输入的变量名，对应 prompt 中的 {input}
    history_messages_key="history",   # 历史消息的变量名，对应 prompt 中的 {history}
)

# -----------------------------------------------------------------------------
# 5. 执行多轮对话
# -----------------------------------------------------------------------------

# 定义一个固定的 session_id，用于标识当前对话会话
# 如果改变这个 ID，AI 将不会记得之前的对话内容（开启新会话）
session_id = "user_123_conversation"

print("--- 开始对话 ---")

# 第一轮对话
# 注意：需要传入 config 参数指定 session_id，以便链知道操作哪一份历史记录
response_1 = conversation_chain.invoke(
    {"input": "Hi there!"},
    config={"configurable": {"session_id": session_id}}
)
print(f"User: Hi there!\nAI: {response_1.content}\n")

# 第二轮对话
# AI 应该能记住上一轮的 "Hi" 语境
response_2 = conversation_chain.invoke(
    {"input": "I'm doing well! Just having a conversation with an AI."},
    config={"configurable": {"session_id": session_id}}
)
print(f"User: I'm doing well! Just having a conversation with an AI.\nAI: {response_2.content}\n")

# 第三轮对话 (测试记忆)
# 询问之前提到的内容，验证记忆是否生效
response_3 = conversation_chain.invoke(
    {"input": "刚才我说我在做什么？"},
    config={"configurable": {"session_id": session_id}}
)
print(f"User: 刚才我说我在做什么？\nAI: {response_3.content}\n")

# -----------------------------------------------------------------------------
# 6. (可选) 查看内部存储的历史记录
# -----------------------------------------------------------------------------
# 直接访问 store 可以看到原始的消息对象列表
history_obj = store.get(session_id)
if history_obj:
    print("--- 当前会话历史快照 ---")
    for msg in history_obj.messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"[{role}]: {msg.content}")
