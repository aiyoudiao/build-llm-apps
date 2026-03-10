#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
私募合规问答智能体 (LangChain 0.2+ 新版架构)

功能描述：
    本模块基于 LangChain 最新架构（0.2.x - 0.3.x）重构了“私募基金运作指引问答助手”。
    相比 v1.0 版本，核心变化如下：
    1. 弃用 LLMSingleActionAgent 和手动正则解析器，采用原生 Tool Calling 机制。
    2. 利用 create_tool_calling_agent / create_react_agent 自动构建 ReAct 循环。
    3. 引入强化的“防幻觉机制”，通过 System Prompt 严格限制模型在知识库缺失时的行为。
    4. 修复了 v2.0 中的缩进错误和导入兼容性问题，确保在不同 LangChain 版本下稳定运行。

    特别设计了“知识边界锁定”，当工具返回无匹配信息时，强制模型停止生成，
    避免利用通用训练数据胡乱编造法律法规定义。

核心工作流程：
    [用户输入] 
       ↓
    [ReAct Agent (内部循环)] -> 模型自动分析意图，决定调用哪个 Tool
       ↓
    [工具执行 (Action)] -> 
        1. 关键词搜索 (模糊匹配规则库)
        2. 类别查询 (按监管维度筛选)
        3. 直接回答 (相似度匹配或触发兜底策略)
       ↓
    [观察结果 (Observation)] -> 获取检索到的规则片段或标准化的“未知”回复
       ↓
    [最终生成 (Final Answer)] -> 整合上下文，输出专业合规建议（严禁越界补充）

依赖环境：
    - Python 3.9+
    - langchain >= 0.2.0
    - langchain-community >= 0.2.0
    - langchain-core >= 0.2.0
    - dashscope (阿里云通义千问 SDK)
    - 环境变量：DASHSCOPE_API_KEY
================================================================================
"""

import re
import os
from typing import List, Dict, Any, Union, Optional, Type
from dotenv import load_dotenv

# ==============================================================================
# LangChain 核心组件导入
# ==============================================================================

# 新版工具与提示词导入
from langchain_core.tools import BaseTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor

# 尝试导入 ChatTongyi (推荐用于 Tool Calling，支持 bind_tools)
try:
    from langchain_community.chat_models import ChatTongyi
    USE_CHAT_MODEL = True
except ImportError:
    # 如果没有 ChatTongyi，尝试用 LLM 类兜底（注意：部分旧版 LLM 包装器可能不支持 bind_tools）
    from langchain_community.llms import Tongyi as ChatTongyi
    USE_CHAT_MODEL = False

load_dotenv()

# ==============================================================================
# 配置与数据层
# ==============================================================================

# 获取阿里云通义千问 API 密钥
# 注意：运行前需在终端执行 export DASHSCOPE_API_KEY="your_key"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 简化的私募基金规则数据库 (模拟向量数据库或 SQL 数据库)
# 在实际生产环境中，此处应替换为从数据库或向量检索系统动态加载
FUND_RULES_DB = [
    {
        "id": "rule001",
        "category": "设立与募集",
        "question": "私募基金的合格投资者标准是什么？",
        "answer": "合格投资者是指具备相应风险识别能力和风险承担能力，投资于单只私募基金的金额不低于100万元且符合下列条件之一的单位和个人：\n1. 净资产不低于1000万元的单位\n2. 金融资产不低于300万元或者最近三年个人年均收入不低于50万元的个人"
    },
    {
        "id": "rule002",
        "category": "设立与募集",
        "question": "私募基金的最低募集规模要求是多少？",
        "answer": "私募证券投资基金的最低募集规模不得低于人民币1000万元。对于私募股权基金、创业投资基金等其他类型的私募基金，监管规定更加灵活，通常需符合基金合同的约定。"
    },
    {
        "id": "rule014",
        "category": "监管规定",
        "question": "私募基金管理人的风险准备金要求是什么？",
        "answer": "私募证券基金管理人应当按照管理费收入的10%计提风险准备金，主要用于赔偿因管理人违法违规、违反基金合同、操作错误等给基金财产或者投资者造成的损失。"
    }
]

# ==============================================================================
# 工具层：使用 @tool 装饰器 (新版推荐写法)
# ==============================================================================

@tool
def search_rules_by_keywords(keywords: str) -> str:
    """
    工具 1: 关键词搜索
    逻辑：将输入关键词拆分，与规则库中的类别和问题进行模糊匹配，返回匹配度最高的前 2 条。
    适用场景：用户询问具体概念（如'合格投资者'、'100万'）时。
    """
    keywords = keywords.strip().lower()
    # 支持中英文逗号及空格分隔
    keyword_list = re.split(r'[,，\s]+', keywords)
    
    matched_rules = []
    for rule in FUND_RULES_DB:
        # 构建待搜索文本
        rule_text = (rule["category"] + " " + rule["question"]).lower()
        # 计算命中关键词的数量
        match_count = sum(1 for kw in keyword_list if kw and kw in rule_text)
        if match_count > 0:
            matched_rules.append((rule, match_count))
    
    # 按匹配度降序排列
    matched_rules.sort(key=lambda x: x[1], reverse=True)
    
    if not matched_rules:
        return "未找到与关键词相关的规则。"
    
    # 格式化输出结果
    result = []
    for rule, _ in matched_rules[:2]:
        result.append(f"类别：{rule['category']}\n问题：{rule['question']}\n答案：{rule['answer']}")
    
    return "\n\n".join(result)

@tool
def search_rules_by_category(category: str) -> str:
    """
    工具 2: 类别查询
    逻辑：根据预设的监管类别（如'设立与募集'）筛选所有相关规则。
    适用场景：用户询问某一大类规定（如'设立与募集'、'监管规定'）时。
    """
    category = category.strip()
    matched_rules = []
    
    for rule in FUND_RULES_DB:
        if category.lower() in rule["category"].lower():
            matched_rules.append(rule)
    
    if not matched_rules:
        return f"未找到类别为 '{category}' 的规则。"
    
    result = []
    for rule in matched_rules:
        result.append(f"问题：{rule['question']}\n答案：{rule['answer']}")
    
    return "\n\n".join(result)

@tool
def answer_question_directly(query: str) -> str:
    """
    工具 3: 直接回答/兜底策略
    逻辑：
    1. 计算用户问题与库中问题的词法相似度。
    2. 若相似度高：直接返回检索到的标准答案。
    3. 若相似度低：触发兜底机制，返回带有特殊标记的“未知”回复，防止模型幻觉。
    
    适用场景：当问题比较宽泛或不确定用哪个特定工具时，直接使用此工具。
    """
    query = query.strip()
    best_rule = None
    best_score = 0
    
    # 简单的词交集相似度算法
    for rule in FUND_RULES_DB:
        query_words = set(query.lower().split())
        rule_words = set((rule["question"] + " " + rule["category"]).lower().split())
        common_words = query_words.intersection(rule_words)
        
        # 避免除以零，计算重合比例
        score = len(common_words) / max(1, len(query_words))
        if score > best_score:
            best_score = score
            best_rule = rule
    
    # 阈值判断：如果相似度低于 0.2 或未匹配到规则，视为超出知识库范围
    if best_score < 0.2 or best_rule is None:
        missing_topic = _identify_missing_topic(query)
        # 关键修改：用明确的标记包裹，暗示模型不要续写，强化防幻觉机制
        return (f"[NO_KNOWLEDGE]\n"
                f"对不起，在我的知识库中没有关于 [{missing_topic}] 的详细信息。\n"
                f"建议查阅中国证券投资基金业协会官网或咨询专业法律人士。\n"
                f"[END_OF_RESPONSE]")
    
    # 在知识库范围内，直接返回标准答案
    return best_rule["answer"]

def _identify_missing_topic(query: str) -> str:
    """
    辅助方法：识别查询中缺失的具体知识主题
    作用：在兜底回复中告诉用户具体缺了什么信息，提升用户体验，体现专业性。
    """
    query = query.lower()
    if "投资" in query and "资产" in query:
        return "私募基金可投资的资产类别"
    elif "公募" in query and "区别" in query:
        return "私募基金与公募基金的区别"
    elif "退出" in query and ("机制" in query or "方式" in query):
        return "创业投资基金的退出机制"
    elif "费用" in query and "结构" in query:
        return "私募基金的费用结构"
    elif "托管" in query:
        return "私募基金资产托管"
    return "您所询问的具体主题"

# 注册工具列表，供 Agent 调用
tools = [search_rules_by_keywords, search_rules_by_category, answer_question_directly]

# ==============================================================================
# Agent 核心架构：新版 ReAct 实现
# ==============================================================================

def create_fund_qa_agent():
    """
    工厂函数：创建并配置完整的问答智能体 (兼容 LangChain 0.2.x - 0.3.x)
    
    核心步骤：
    1. 动态检测并导入合适的 Agent 创建器 (create_tool_calling_agent 或 create_react_agent)。
    2. 初始化支持 Tool Calling 的大语言模型。
    3. 构建包含强约束规则的 System Prompt，防止模型幻觉。
    4. 组装 Agent 与 Executor。
    """
    # 1. 动态导入 Agent 创建器 (适配不同版本的 LangChain)
    try:
        # 优先尝试新版 API (LangChain 0.2.0+)
        from langchain.agents import create_tool_calling_agent
        agent_creator = create_tool_calling_agent
        print("✅ 使用 create_tool_calling_agent (推荐)")
    except ImportError:
        try:
            # 次选尝试 create_react_agent (LangChain 0.1.17+)
            from langchain.agents import create_react_agent
            agent_creator = create_react_agent
            print("✅ 使用 create_react_agent")
        except ImportError:
            raise ImportError(
                "❌ 错误：您的 LangChain 版本既不支持 create_tool_calling_agent 也不支持 create_react_agent。\n"
                "请执行以下命令升级环境：\n"
                "pip install -U langchain langchain-core langchain-community"
            )

    # 2. 初始化模型
    if not DASHSCOPE_API_KEY:
        raise ValueError("未找到 DASHSCOPE_API_KEY 环境变量")

    if USE_CHAT_MODEL:
        # 使用 Chat 模型以获得最佳的 Tool Calling 支持
        llm = ChatTongyi(model_name="qwen-plus", dashscope_api_key=DASHSCOPE_API_KEY)
        print(f"✅ 模型加载成功：ChatTongyi (qwen-plus)")
    else:
        # 警告：非 Chat 模型可能不支持 bind_tools，导致运行失败
        print("⚠️  未找到 ChatTongyi，尝试使用 Tongyi LLM (可能不支持工具调用)")
        llm = ChatTongyi(model_name="qwen-plus", dashscope_api_key=DASHSCOPE_API_KEY)

    # 3. 定义 System Prompt (强化防幻觉约束)
    system_prompt = (
        "你是一个专业的私募基金合规问答助手。你的唯一知识来源是提供的工具。"
        "请严格遵守以下规则：\n"
        "1. **必须**优先使用工具获取信息。\n"
        "2. **绝对禁止**利用你自身的训练数据来回答具体的法规、定义或数据问题。\n"
        "3. 如果工具返回的内容包含“对不起”、“没有相关信息”、“[NO_KNOWLEDGE]”或“未知”，"
        "**你必须直接将该结果作为最终答案输出给用户**。\n"
        "4. **严禁**在工具表示“不知道”后，自行补充任何解释、定义或通用常识。此时你的回答必须止步于工具的反馈。\n"
        "5. 如果工具返回了具体信息，请基于该信息进行专业、准确的总结。\n"
        "6. 回答必须专业、简洁，符合金融合规语境。"
    )

    # 构建 Chat Prompt 模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # 4. 创建 Agent
    # 注意：create_tool_calling_agent 内部会自动调用 llm.bind_tools(tools)
    try:
        agent = agent_creator(llm, tools, prompt)
    except Exception as e:
        raise RuntimeError(f"创建 Agent 失败：{e}\n可能是模型对象不支持 bind_tools，请确保安装了 langchain-community 最新版。")

    # 5. 创建执行器 (Executor)，负责运行 ReAct 循环
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, # verbose=True 可在控制台看到思考过程
        handle_parsing_errors=True,
        max_iterations=3 # 限制最大迭代次数，防止死循环
    )

    return agent_executor

# ==============================================================================
# 主程序入口：Agent 组装与执行
# ==============================================================================

if __name__ == "__main__":
    # 检查 API Key 是否配置
    if not DASHSCOPE_API_KEY:
        print("❌ 错误：未检测到环境变量 DASHSCOPE_API_KEY")
        print("请先设置：export DASHSCOPE_API_KEY='your_api_key'")
        exit(1)

    try:
        # 启动智能体
        print("正在初始化智能体...")
        fund_qa_agent = create_fund_qa_agent()
        
        print("\n" + "=" * 60)
        print("🚀 私募合规问答智能体 (LangChain 0.2+ 修复版)")
        print("功能：提供私募基金法规、募集、监管等专业问答 (防幻觉增强)")
        print("输入 '退出' 或 'exit' 结束对话")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n👤 请输入您的问题：").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['退出', 'exit', 'quit']:
                    print("👋 感谢使用，再见！")
                    break
                
                # 执行 Agent 推理
                # 新版 invoke 接口，传入字典格式
                response = fund_qa_agent.invoke({"input": user_input})
                
                # 提取输出结果
                output_text = response.get('output', '无响应')
                print(f"\n🤖 回答：{output_text}")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  程序已中断，感谢使用！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误：{e}")
                print("💡 建议：检查 API Key 或网络连接，或尝试简化您的问题。")
                
    except Exception as e:
        print(f"\n❌ 初始化失败：{e}")
        print("\n💡 解决方案：")
        print("1. 检查依赖版本：pip show langchain langchain-core langchain-community")
        print("2. 强制重装兼容版本：pip install -U langchain==0.2.5 langchain-core==0.2.9 langchain-community==0.2.5")
