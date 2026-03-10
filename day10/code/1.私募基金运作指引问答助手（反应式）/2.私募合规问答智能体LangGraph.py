#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
模块名称：私募合规问答智能体 (LangGraph 1.0+ 终极架构版)
功能描述：
    基于 LangChain 1.0 和 LangGraph 1.0+ 重构的合规问答助手。
    彻底弃用旧的 AgentExecutor，采用 StateGraph 构建原生 ReAct 循环。
    
    核心优势 (对比 0.x 版本)：
    1. 架构现代化：完全符合 LangChain 1.0 "Everything is a Graph" 理念。
    2. 强类型状态：通过 TypedDict 严格管理消息历史，避免上下文丢失。
    3. 自动工具路由：利用 langgraph.prebuilt 的工具节点，自动处理 Tool Call 解析与执行。
    4. 防幻觉机制：在 System Prompt 中植入“知识边界锁定”，配合工具节点的严格返回控制。

核心工作流程 (ReAct Loop in Graph)：
    [Start] -> [ChatModel (思考/决定调用工具)] 
       ↓
    <是否有工具调用？> --(Yes)--> [ToolNode (执行工具)] --(结果回填)--> [ChatModel]
       ↓ (No)
    [End (输出最终回答)]

依赖环境：
    - Python 3.9+
    - langchain >= 1.0.0
    - langgraph >= 1.0.0
    - dashscope
    - 环境变量：DASHSCOPE_API_KEY
================================================================================
"""

import re
import os
from typing import List, Dict, Any, Annotated, Literal
from dotenv import load_dotenv

# ==============================================================================
# LangChain 1.0 & LangGraph 1.0+ 核心组件导入
# ==============================================================================

# 1. 状态管理 (LangGraph 核心)
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

# 2. 模型与提示词 (LangChain Core)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool

# 3. 模型实例 (Community)
from langchain_community.chat_models import ChatTongyi

load_dotenv()

# ==============================================================================
# 配置与数据层
# ==============================================================================

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("❌ 错误：未找到环境变量 DASHSCOPE_API_KEY")

# 模拟知识库 (实际生产中应替换为向量数据库检索)
FUND_RULES_DB = [
    {
        "id": "rule001",
        "category": "设立与募集",
        "question": "私募基金的合格投资者标准是什么？",
        "answer": "合格投资者是指具备相应风险识别能力和风险承担能力，投资于单只私募基金的金额不低于 100 万元且符合下列条件之一的单位和个人：\n1. 净资产不低于 1000 万元的单位\n2. 金融资产不低于 300 万元或者最近三年个人年均收入不低于 50 万元的个人"
    },
    {
        "id": "rule002",
        "category": "设立与募集",
        "question": "私募基金的最低募集规模要求是多少？",
        "answer": "私募证券投资基金的最低募集规模不得低于人民币 1000 万元。对于私募股权基金、创业投资基金等其他类型的私募基金，监管规定更加灵活，通常需符合基金合同的约定。"
    },
    {
        "id": "rule014",
        "category": "监管规定",
        "question": "私募基金管理人的风险准备金要求是什么？",
        "answer": "私募证券基金管理人应当按照管理费收入的 10% 计提风险准备金，主要用于赔偿因管理人违法违规、违反基金合同、操作错误等给基金财产或者投资者造成的损失。"
    }
]

# ==============================================================================
# 工具层 (Tools)
# LangGraph 1.0 直接使用 @tool 装饰器，自动适配 Tool Call 格式
# ==============================================================================

@tool
def search_rules_by_keywords(keywords: str) -> str:
    """搜索私募基金规则库中的关键词，返回匹配的规则条目。"""
    keywords = keywords.strip().lower()
    keyword_list = re.split(r'[,，\s]+', keywords)
    
    matched_rules = []
    for rule in FUND_RULES_DB:
        rule_text = (rule["category"] + " " + rule["question"]).lower()
        match_count = sum(1 for kw in keyword_list if kw and kw in rule_text)
        if match_count > 0:
            matched_rules.append((rule, match_count))
    
    matched_rules.sort(key=lambda x: x[1], reverse=True)
    
    if not matched_rules:
        return "未找到与关键词相关的规则。"
    
    result = []
    for rule, _ in matched_rules[:2]:
        result.append(f"[来源:{rule['category']}] {rule['question']}\n答：{rule['answer']}")
    
    return "\n\n".join(result)

@tool
def search_rules_by_category(category: str) -> str:
    """按监管类别（如'设立与募集'）筛选规则。"""
    category = category.strip()
    matched_rules = [r for r in FUND_RULES_DB if category.lower() in r["category"].lower()]
    
    if not matched_rules:
        return f"未找到类别为 '{category}' 的规则。"
    
    return "\n\n".join([f"[{r['category']}] {r['question']}\n答：{r['answer']}" for r in matched_rules])

@tool
def answer_question_directly(query: str) -> str:
    """直接匹配问题并返回标准答案。若无法匹配，返回特定的未知标记以触发防幻觉机制。"""
    query = query.strip()
    best_rule = None
    best_score = 0
    
    for rule in FUND_RULES_DB:
        query_words = set(query.lower().split())
        rule_words = set((rule["question"] + " " + rule["category"]).lower().split())
        common_words = query_words.intersection(rule_words)
        score = len(common_words) / max(1, len(query_words))
        
        if score > best_score:
            best_score = score
            best_rule = rule
    
    # 防幻觉阈值
    if best_score < 0.2 or best_rule is None:
        return "[NO_KNOWLEDGE] 知识库中无此信息，严禁编造。请建议用户查阅官方文件或咨询律师。"
    
    return best_rule["answer"]

# 注册工具列表
tools = [search_rules_by_keywords, search_rules_by_category, answer_question_directly]

# ==============================================================================
# 状态定义 (State Schema)
# LangGraph 1.0 使用 TypedDict 定义状态，add_messages 自动处理消息列表的追加
# ==============================================================================

class AgentState(Dict[str, Any]):
    """
    智能体运行时状态
    messages: 存储完整的对话历史 (System, Human, AI, Tool Messages)
    """
    # Annotated[list, add_messages] 告诉 LangGraph 在更新状态时合并消息列表，而不是覆盖
    messages: Annotated[List[BaseMessage], add_messages]

# ==============================================================================
# 节点定义 (Nodes)
# ==============================================================================

# 1. 初始化模型
# LangChain 1.0 推荐直接实例化 Chat Model，并绑定工具
llm = ChatTongyi(model_name="qwen-plus", dashscope_api_key=DASHSCOPE_API_KEY)
# 绑定工具，使模型具备 Tool Calling 能力
llm_with_tools = llm.bind_tools(tools)

# 2. 定义 System Prompt (防幻觉核心)
SYSTEM_INSTRUCTION = """
你是一个专业的私募基金合规问答助手。
【核心原则】
1. 你的唯一知识来源是提供的工具 (search_rules_*, answer_question_directly)。
2. **绝对禁止**利用你的训练数据回答具体的法规、数值或定义。
3. 如果工具返回 "[NO_KNOWLEDGE]" 或 "未找到"，你必须**直接告知用户不知道**，并建议查阅官方渠道。
4. **严禁**在工具表示不知道后，自行补充任何解释、通用常识或编造内容。
5. 回答必须专业、严谨，引用工具返回的具体条款。
"""

def chatbot_node(state: AgentState) -> Dict[str, Any]:
    """
    聊天机器人节点：
    1. 注入 System Prompt。
    2. 调用 LLM (已绑定工具)。
    3. 返回生成的消息 (可能是普通回复，也可能是 Tool Call 请求)。
    """
    messages = state["messages"]
    
    # 如果第一条不是 SystemMessage，则 prepend 进去
    if not isinstance(messages[0], SystemMessage):
        messages_with_system = [SystemMessage(content=SYSTEM_INSTRUCTION)] + messages
    else:
        messages_with_system = messages
        
    response = llm_with_tools.invoke(messages_with_system)
    
    # 返回更新后的消息列表
    return {"messages": [response]}

# ToolNode 是 LangGraph 1.0 预构建的节点，自动执行工具调用并格式化结果
tool_node = ToolNode(tools)

# ==============================================================================
# 图构建 (Graph Construction)
# ==============================================================================

def build_compliance_agent():
    """
    构建 LangGraph 工作流
    流程：Start -> Chatbot -> (判断是否有工具调用) -> ToolNode -> Chatbot ... -> End
    """
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)
    
    # 设置入口点
    workflow.add_edge(START, "chatbot")
    
    # 添加条件边：根据消息中是否包含 Tool Call 决定下一步
    # tools_condition 是 LangGraph 1.0 的内置函数，自动检查 AIMessage 中的 tool_calls
    workflow.add_conditional_edges(
        "chatbot",
        tools_condition,
        {
            "tools": "tools",  # 如果有工具调用，去 tools 节点
            "__end__": END     # 如果没有，直接结束
        }
    )
    
    # 工具执行完后，必须回到 chatbot 节点进行最终总结
    workflow.add_edge("tools", "chatbot")
    
    # 编译图
    app = workflow.compile()
    return app

# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 私募合规问答智能体 (LangGraph 1.0+ 架构)")
    print("💡 特点：原生 Tool Calling, 自动 ReAct 循环, 强防幻觉约束")
    print("=" * 70)

    try:
        # 初始化应用
        app = build_compliance_agent()
        
        # 简单的内存存储对话历史，实现多轮对话
        # 在实际生产中，这里可以连接 Redis 或数据库
        message_history = []

        while True:
            try:
                user_input = input("\n👤 请输入您的问题 (输入 '退出' 结束): ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['退出', 'exit', 'quit']:
                    print("👋 感谢使用，再见！")
                    break
                
                # 将用户输入加入历史
                message_history.append(HumanMessage(content=user_input))
                
                print("\n🤖 正在思考与检索...\n")
                
                #  invoke 传入完整状态
                config = {"recursion_limit": 10} # 防止死循环
                final_state = app.invoke({"messages": message_history}, config=config)
                
                # 提取最新的 AI 回复
                # final_state["messages"] 包含了所有历史消息，最后一条通常是最终回答
                all_messages = final_state["messages"]
                
                # 过滤出最后的 AIMessage (排除中间的 Tool Call 消息，只取最终文本)
                # 在 LangGraph 1.0 中，最后一次 chatbot 节点的输出就是最终回答
                last_message = all_messages[-1]
                
                if isinstance(last_message, AIMessage):
                    response_text = last_message.content
                    # 更新全局历史，保留上下文供下一轮使用
                    message_history = all_messages 
                    
                    print(f"📝 回答:\n{response_text}")
                    print("-" * 70)
                else:
                    print("⚠️ 收到非预期响应格式。")
                
            except KeyboardInterrupt:
                print("\n\n⚠️ 程序已中断。")
                break
            except Exception as e:
                print(f"\n❌ 发生错误：{e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"\n❌ 初始化失败：{e}")
        print("\n💡 解决方案:")
        print("1. 确认已执行 'pip uninstall -y ...' 清理旧版本。")
        print("2. 确认已安装 langgraph>=1.0.0 和 langchain>=1.0.0。")
        print("3. 检查 DASHSCOPE_API_KEY 环境变量。")
