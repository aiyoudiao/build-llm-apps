#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
模块名称：私募合规问答智能体 (LangChain 0.3.x 稳定架构版)
版本重要提示：
    【关键】本代码专为 LangChain 0.3.x 系列设计 (推荐 0.3.20)。
    ❌ 不兼容 LangChain 1.0+：
       在 LangChain 1.0 中，`create_tool_calling_agent` 已被移除并迁移至 LangGraph。
       若您希望继续使用纯 LangChain (AgentExecutor) 架构而不引入 LangGraph，
       请务必将依赖锁定在 0.3.x 版本。

功能描述：
    本模块基于 LangChain 0.2.x - 0.3.x 成熟架构重构了“私募基金运作指引问答助手”。
    相比 v1.0 (旧版正则解析) 和 v2.0 (存在兼容性缺陷)，本版本的核心升级如下：
    
    1. 【原生 Tool Calling 机制】
       - 弃用 LLMSingleActionAgent 和脆弱的手动正则解析器。
       - 采用 LLM 原生的 Function Calling / Tool Calling 能力 (`bind_tools`)。
       - 优势：参数提取更精准，复杂意图识别率大幅提升。
    
    2. 【自动化 ReAct 循环构建】
       - 利用 `create_tool_calling_agent` (或降级兼容 `create_react_agent`) 自动组装 Agent。
       - 内部自动管理 "思考 (Thought) -> 行动 (Action) -> 观察 (Observation)" 的闭环。
       - 无需手动编写复杂的循环逻辑，代码更简洁健壮。
    
    3. 【强化防幻觉机制 (Anti-Hallucination)】
       - System Prompt 强约束：明确禁止模型利用预训练知识回答法规细节。
       - 知识边界锁定：当工具返回无匹配信息时，注入特殊标记 `[NO_KNOWLEDGE]`。
       - 强制截断策略：Prompt 指令要求模型在检测到该标记时，必须直接输出兜底回复，
         严禁进行任何发散性解释或编造法律定义。
    
    4. 【兼容性与稳定性修复】
       - 修复了 v2.0 中的缩进错误和导入路径问题。
       - 增加了动态导入检测逻辑，若环境版本不匹配（如误装 1.0+），将抛出明确的指引错误。

特别设计 - 知识边界锁定流程：
    当用户提问超出知识库范围 -> 工具返回 "[NO_KNOWLEDGE]..." -> 
    Agent 读取到标记 -> 触发 System Prompt 中的"停止生成"指令 -> 
    最终输出："对不起，知识库中无此信息..." (杜绝胡编乱造)

核心工作流程：
    [用户输入] 
       ↓
    [ReAct Agent (内部循环)] 
       ├─> 意图分析：模型判断需要调用哪个 Tool (Search/Category/Direct)
       └─> 参数提取：自动从自然语言中提取搜索关键词
       ↓
    [工具执行 (Action)] 
       ├─> 1. 关键词搜索：模糊匹配规则库 (支持中英文分词)
       ├─> 2. 类别查询：按监管维度 (设立/募集/监管) 筛选
       └─> 3. 直接回答：相似度匹配 或 触发兜底策略 ([NO_KNOWLEDGE])
       ↓
    [观察结果 (Observation)] 
       ├─> 成功：获取具体的法规片段
       └─> 失败：获取标准化的“未知”回复标记
       ↓
    [最终生成 (Final Answer)] 
       ├─> 成功场景：整合上下文，输出专业、准确的合规建议
       └─> 失败场景：直接复读工具的兜底回复，严禁越界补充

================================================================================
【环境安装指令】 (请务必执行以下命令以确保版本兼容)
================================================================================
# 推荐方案：锁定 LangChain 0.3.x 系列 (最后支持 create_tool_calling_agent 的纯 LC 版本)
pip install "langchain==0.3.20" "langchain-core==0.3.29" "langchain-community==0.3.13" dashscope python-dotenv

# 注意：
# 1. 不要执行 `pip install -U langchain`，这会升级到 1.0+ 导致代码报错。
# 2. langchain-community 目前最高为 0.4.x，但为了配合 0.3.x 的核心架构，
#    建议统一使用 0.3.x 全套以避免潜在的 API 行为差异。
================================================================================

依赖环境：
    - Python 3.9+ (推荐 3.10+)
    - langchain == 0.3.x (核心框架)
    - langchain-community == 0.3.x (模型集成，含 ChatTongyi)
    - langchain-core == 0.3.x (基础接口)
    - dashscope (阿里云通义千问 SDK)
    - python-dotenv (环境变量管理)
    - 环境变量：DASHSCOPE_API_KEY (需在终端 export 或 .env 文件中配置)

================================================================================
"""

import re
import os
from typing import List, Dict, Any, Union, Optional, Type
from dotenv import load_dotenv

# ==============================================================================
# LangChain 核心组件导入
# ==============================================================================

# 工具与提示词 (这些在 0.3.x 和 1.x 中路径基本一致)
from langchain_core.tools import BaseTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor

# 尝试导入 ChatTongyi (推荐用于 Tool Calling，支持 bind_tools)
# 在 langchain-community 0.3.x 中，路径为 langchain_community.chat_models
try:
    from langchain_community.chat_models import ChatTongyi
    USE_CHAT_MODEL = True
except ImportError:
    # 兜底方案：尝试旧版 LLM 包装器 (不推荐，可能不支持 bind_tools)
    try:
        from langchain_community.llms import Tongyi as ChatTongyi
        USE_CHAT_MODEL = False
    except ImportError:
        raise ImportError("❌ 错误：未找到 ChatTongyi。请确保已安装 langchain-community。")

load_dotenv()

# ==============================================================================
# 配置与数据层
# ==============================================================================

# 获取阿里云通义千问 API 密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 简化的私募基金规则数据库 (模拟生产环境中的向量数据库或 SQL 数据库)
# 实际应用中，这里应替换为真实的检索接口
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
# 工具层：使用 @tool 装饰器
# ==============================================================================

@tool
def search_rules_by_keywords(keywords: str) -> str:
    """
    工具 1: 关键词搜索
    逻辑：将输入关键词拆分，与规则库中的类别和问题进行模糊匹配，返回匹配度最高的前 2 条。
    适用场景：用户询问具体概念（如'合格投资者'、'100 万'）时。
    """
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
    
    for rule in FUND_RULES_DB:
        query_words = set(query.lower().split())
        rule_words = set((rule["question"] + " " + rule["category"]).lower().split())
        common_words = query_words.intersection(rule_words)
        
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
    
    return best_rule["answer"]

def _identify_missing_topic(query: str) -> str:
    """辅助方法：识别查询中缺失的具体知识主题"""
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

# 注册工具列表
tools = [search_rules_by_keywords, search_rules_by_category, answer_question_directly]

# ==============================================================================
# Agent 核心架构：新版 ReAct 实现
# ==============================================================================

def create_fund_qa_agent():
    """
    工厂函数：创建并配置完整的问答智能体
    
    版本兼容性检查逻辑：
    1. 优先尝试从 langchain.agents 导入 create_tool_calling_agent (适用于 0.2.x - 0.3.x)。
    2. 如果失败，尝试导入 create_react_agent (适用于部分 0.1.x 版本)。
    3. 如果都失败，说明是 LangChain 1.0+，此时抛出明确错误，提示用户降级或迁移至 LangGraph。
    """
    
    agent_creator = None
    creator_name = ""

    # 1. 动态导入 Agent 创建器
    try:
        # LangChain 0.2.0 - 0.3.x 的标准入口
        from langchain.agents import create_tool_calling_agent
        agent_creator = create_tool_calling_agent
        creator_name = "create_tool_calling_agent (推荐)"
    except ImportError:
        try:
            # 较旧版本的入口
            from langchain.agents import create_react_agent
            agent_creator = create_react_agent
            creator_name = "create_react_agent"
        except ImportError:
            # LangChain 1.0+ 会进入这里
            raise ImportError(
                "❌ 版本不兼容错误：\n"
                "当前环境检测到 LangChain 1.0 或更高版本，该版本已移除 'create_tool_calling_agent'。\n"
                "本代码设计为纯 LangChain 架构 (不使用 LangGraph)，因此必须使用 0.3.x 版本。\n\n"
                "✅ 解决方案：请执行以下命令降级依赖:\n"
                "pip install \"langchain==0.3.20\" \"langchain-core==0.3.29\" \"langchain-community==0.3.13\""
            )

    print(f"✅ 成功加载 Agent 创建器：{creator_name}")

    # 2. 初始化模型
    if not DASHSCOPE_API_KEY:
        raise ValueError("❌ 错误：未找到 DASHSCOPE_API_KEY 环境变量")

    if USE_CHAT_MODEL:
        llm = ChatTongyi(model_name="qwen-plus", dashscope_api_key=DASHSCOPE_API_KEY)
        print(f"✅ 模型加载成功：ChatTongyi (qwen-plus)")
    else:
        print("⚠️  警告：使用非 Chat 模型包装器，可能影响 Tool Calling 效果。")
        llm = ChatTongyi(model_name="qwen-plus", dashscope_api_key=DASHSCOPE_API_KEY)

    # 3. 定义 System Prompt (强化防幻觉约束)
    system_prompt = (
        "你是一个专业的私募基金合规问答助手。你的唯一知识来源是提供的工具。\n"
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
    # MessagesPlaceholder 用于在多轮对话中动态插入历史记录和 Agent 的思考过程 (agent_scratchpad)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # 4. 创建 Agent
    # create_tool_calling_agent 内部会自动执行 llm.bind_tools(tools)
    try:
        agent = agent_creator(llm, tools, prompt)
    except Exception as e:
        raise RuntimeError(f"创建 Agent 实例失败：{e}\n可能是模型对象不支持 bind_tools，请检查 langchain-community 版本。")

    # 5. 创建执行器 (Executor)
    # AgentExecutor 负责驱动 ReAct 循环：调用 Agent -> 执行工具 -> 返回结果 -> 再次调用 Agent
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,             # 开启详细日志，可在控制台看到思考过程
        handle_parsing_errors=True, # 自动处理 LLM 输出格式错误
        max_iterations=3,         # 限制最大循环次数，防止死循环
        max_execution_time=60     # 限制单次执行总时长 (秒)
    )

    return agent_executor

# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":
    # 预检查 API Key
    if not DASHSCOPE_API_KEY:
        print("❌ 错误：未检测到环境变量 DASHSCOPE_API_KEY")
        print("💡 请先设置：export DASHSCOPE_API_KEY='your_api_key'")
        exit(1)

    try:
        print("正在初始化智能体...")
        fund_qa_agent = create_fund_qa_agent()
        
        print("\n" + "=" * 70)
        print("🚀 私募合规问答智能体 (LangChain 0.3.x 稳定版)")
        print("💡 模式：ReAct Agent + Tool Calling + 防幻觉增强")
        print("⚠️  注意：本代码依赖 LangChain 0.3.x，请勿升级到 1.0+")
        print("=" * 70)

        while True:
            try:
                user_input = input("\n👤 请输入您的问题 (输入 '退出' 结束): ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['退出', 'exit', 'quit']:
                    print("👋 感谢使用，再见！")
                    break
                
                # 执行 Agent 推理
                # 注意：invoke 接收字典，其中 'input' 是必填项
                # 如果需要多轮对话，还需传入 'chat_history' 列表 (本示例为简化版，每轮独立，如需多轮需在外部维护 history)
                response = fund_qa_agent.invoke({"input": user_input})
                
                # 提取输出结果
                output_text = response.get('output', '无响应')
                print(f"\n🤖 回答:\n{output_text}")
                print("-" * 70)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  程序已中断，感谢使用！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误：{e}")
                print("💡 建议：检查 API Key、网络连接或简化问题。")
                
    except ImportError as ie:
        # 捕获版本不兼容的错误并打印友好提示
        print(f"\n{ie}")
    except Exception as e:
        print(f"\n❌ 初始化失败：{e}")
        print("\n💡 解决方案：")
        print("1. 确认依赖版本是否为 0.3.x 系列。")
        print("2. 重新安装：pip install \"langchain==0.3.20\" \"langchain-community==0.3.13\"")
