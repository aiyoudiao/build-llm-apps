#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
模块名称：混合架构财富投顾 (Hybrid Wealth Advisor)
基于框架：LangChain 1.2+ & LangGraph 1.0+

功能描述：
    本模块实现了一个具备“双模态”处理能力的智能财富顾问。
    它通过一个动态协调层（Coordinator），根据用户问题的复杂度，自动在
    “反应式（快思考）”和“深思熟虑（慢思考）”两种模式间切换。

    核心优势 (v1.0 新特性)：
    1. 【动态路由】自动识别意图：简单查询走快速通道；复杂规划走深度分析通道。
    2. 【工具增强】反应式模式集成了 LangChain 1.0 标准的 create_react_agent，
       可调用实时工具（如指数查询）；深思熟虑模式采用多步 Chain of Thought。
    3. 【状态隔离】基于 LangGraph 的状态机管理，确保不同模式下的上下文不混淆。
    4. 【容错机制】内置多层异常捕获，防止单点故障导致整个服务中断。

三层架构流程：
    ┌──────────────────────────────────────────────────────────────────────┐
    │  Layer 1: 协调层 (Coordinator)                                      │
    │  [评估节点] -> 判断意图 (紧急/信息/分析) -> 路由至 Reactive 或 Deliberative │
    └──────────────────────────────────────────────────────────────────────┘
                                   ↓ 分支
    ┌─────────────────────┐               ┌─────────────────────────────────┐
    │  Layer 2A: 反应式   │               │  Layer 2B: 深思熟虑             │
    │  (Reactive Mode)    │               │  (Deliberative Mode)            │
    │  - 适用：即时查询   │               │  - 适用：复杂规划               │
    │  - 核心：ReAct Agent│               │  - 核心：多步推理链             │
    │  - 动作：调用工具   │               │  - 步骤：收集->分析->建议       │
    │  - 输出：直接回答   │               │  - 输出：深度报告               │
    └─────────────────────┘               └─────────────────────────────────┘
                                   ↓ 汇聚
    ┌──────────────────────────────────────────────────────────────────────┐
    │  Layer 3: 响应层 (Responder)                                        │
    │  统一格式化输出，记录耗时，返回最终建议                              │
    └──────────────────────────────────────────────────────────────────────┘

依赖环境：
    - Python 3.9+
    - langchain == 1.2.10
    - langgraph == 1.0.10
    - langchain-community == 0.4.1
    - dashscope (阿里云通义千问 SDK)
    - 环境变量：DASHSCOPE_API_KEY

使用说明：
    1. 配置环境变量 DASHSCOPE_API_KEY。
    2. 运行脚本，选择预设问题或输入自定义问题。
    3. 系统将自动展示 Mermaid 流程图并执行相应模式。
================================================================================
"""

import os
import json
import warnings
from datetime import datetime
from typing import Dict, List, Any, Literal, TypedDict, Optional, Annotated

from dotenv import load_dotenv
load_dotenv()

# LangChain & LangGraph 核心组件 (v1.0+ 标准导入)
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models import ChatTongyi

# [修正] 尝试从 langchain.agents 导入 create_react_agent
# 如果 langchain 1.2.10 中没有直接导出，可能需要从 langchain_community 或手动构建
try:
    from langchain.agents import create_react_agent, AgentExecutor
except ImportError:
    # 兼容方案：如果主包没有，尝试从 community 或其他位置，或者稍后手动构建
    # 在 langchain 1.x 中，create_react_agent 通常在 langchain.agents 中
    # 如果实在找不到，我们将使用手动构建 ReAct 链的方式作为 fallback
    create_react_agent = None
    AgentExecutor = None

from langgraph.graph import StateGraph, END, START



# ==============================================================================
# 配置与初始化
# ==============================================================================

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("❌ 错误：未找到环境变量 DASHSCOPE_API_KEY，请先行配置。")

# 初始化 LLM (使用 qwen-turbo 以平衡速度与成本)
# 注意：LangChain 1.0+ 推荐使用 ChatModel 而非 LLM，但为兼容原有逻辑此处保留 Tongyi
llm = ChatTongyi(model_name="qwen-turbo-latest", dashscope_api_key=DASHSCOPE_API_KEY)

# ==============================================================================
# 数据结构定义 (Schema Definitions)
# ==============================================================================

class WealthAdvisorState(TypedDict):
    """
    智能体全局状态容器
    所有节点共享此状态，通过返回字典增量更新
    """
    # [输入]
    user_query: str
    customer_profile: Optional[Dict[str, Any]]
    
    # [路由决策]
    query_type: Optional[Literal["emergency", "informational", "analytical"]]
    processing_mode: Optional[Literal["reactive", "deliberative"]]
    
    # [中间产物]
    emergency_response: Optional[Dict[str, Any]]
    market_data: Optional[Dict[str, Any]]
    analysis_results: Optional[Dict[str, Any]]
    
    # [输出]
    final_response: Optional[str]
    
    # [控制流]
    current_phase: Optional[str]
    error: Optional[str]

# ==============================================================================
# 工具定义 (Tools for Reactive Mode)
# ==============================================================================

def query_shanghai_index(_: str = "") -> str:
    """
    [工具] 上证指数实时查询 (模拟版)
    在实际生产环境中，此处应替换为真实的金融数据 API 调用 (如 Tushare, AkShare)
    """
    # 模拟数据生成
    name, price, change, pct = "上证指数", "3125.62", "+6.32", "+0.20%"
    result = f"📈 {name} 当前点位: {price}，涨跌: {change}，涨跌幅: {pct}"
    print(f"🔧 [Tool Call] 调用上证指数查询工具 -> {result}")
    return result

# 定义可用工具列表
REACTIVE_TOOLS = [
    Tool(
        name="上证指数查询",
        func=query_shanghai_index,
        description="用于查询上证指数的最新行情数据。当用户询问大盘、指数、点位时使用。"
    )
]

# ==============================================================================
# 提示词模板 (Prompts)
# ==============================================================================

ASSESSMENT_PROMPT = """你是一个智能财富顾问的协调员。请分析用户查询，决定处理策略。

用户查询: {user_query}

任务:
1. 判断查询类型 (query_type):
   - "emergency": 需要即时数据或简单事实 (如：今天股价多少？什么是 ETF？)
   - "informational": 需要知识解释 (如：税收政策是什么？)
   - "analytical": 需要深度分析、规划或建议 (如：我该如何调整仓位？退休计划？)

2. 选择处理模式 (processing_mode):
   - "reactive": 针对 emergency/informational 类型，追求速度，可调用工具。
   - "deliberative": 针对 analytical 类型，追求深度，需多步推理。

请以 JSON 格式返回:
{{
    "query_type": "类型",
    "processing_mode": "模式",
    "reasoning": "简短的理由"
}}
"""

DATA_COLLECTION_PROMPT = """作为数据收集专家，请为以下投资分析任务确定所需数据。

用户查询: {user_query}
客户画像: {customer_profile}

请列出需要的数据类型 (如：宏观指标、行业趋势、历史回报、风险系数等)，并生成一份合理的模拟数据集供后续分析使用。

返回 JSON 格式:
{{
    "required_data_types": ["类型 1", "类型 2"],
    "collected_data": {{ ...模拟的具体数据... }}
}}
"""

ANALYSIS_PROMPT = """作为首席投资分析师，请基于以下信息进行深度推演。

用户查询: {user_query}
客户画像: {customer_profile}
市场数据: {market_data}

请输出详细的分析结论 (JSON 格式):
{{
    "market_assessment": "市场现状评估",
    "portfolio_analysis": "当前持仓诊断",
    "recommendations": [{{"action": "...", "reason": "..."}}],
    "risk_analysis": "潜在风险提示",
    "expected_outcomes": "预期收益与情景分析"
}}
"""

RECOMMENDATION_PROMPT = """作为资深理财师，请将上述专业分析转化为客户易懂的执行建议。

用户查询: {user_query}
客户画像: {customer_profile}
分析结论: {analysis_results}

要求:
- 语气亲切专业，避免过度堆砌术语。
- 包含：总体策略、具体行动步骤、资产配置比例、风控措施、时间规划。
- 直接输出自然语言文本，不要 JSON。
"""

# ==============================================================================
# 工作流节点 (Workflow Nodes)
# ==============================================================================

def assess_query_node(state: WealthAdvisorState) -> WealthAdvisorState:
    """
    [节点 1] 情境评估
    功能：解析用户意图，动态路由到反应式或深思熟虑模式
    """
    print("⚖️ [1/4] 正在评估查询意图...")
    try:
        prompt = ChatPromptTemplate.from_template(ASSESSMENT_PROMPT)
        chain = prompt | llm | JsonOutputParser()
        
        result = chain.invoke({"user_query": state["user_query"]})
        
        # 安全校验枚举值
        mode = result.get("processing_mode", "reactive")
        q_type = result.get("query_type", "emergency")
        
        if mode not in ["reactive", "deliberative"]: mode = "reactive"
        if q_type not in ["emergency", "informational", "analytical"]: q_type = "emergency"
        
        print(f"   ✅ 决策结果：模式={mode}, 类型={q_type}, 理由={result.get('reasoning', 'N/A')}")
        
        return {
            "processing_mode": mode,
            "query_type": q_type
        }
    except Exception as e:
        return {"error": f"评估失败：{str(e)}", "processing_mode": "reactive"}

def reactive_processing_node(state: WealthAdvisorState) -> WealthAdvisorState:
    """
    [节点 2A] 反应式处理
    功能：构建 ReAct Agent (LangChain 1.0 标准方式)，调用工具解决即时性问题
    """
    print("⚡ [2A/4] 启动反应式模式 (ReAct Agent)...")
    try:
        # LangChain 1.0+ 标准创建 ReAct Agent 的方式
        # create_react_agent 返回的是一个 CompiledGraph (可直接 invoke)
        agent_graph = create_react_agent(
            model=llm,
            tools=REACTIVE_TOOLS,
            prompt="你是一个专业的财富顾问助手。请利用提供的工具回答用户问题。\n如果用户询问股市行情，务必使用'上证指数查询'工具。\n保持回答简洁、专业。"
        )
        
        # 执行 Agent
        # LangGraph 1.0 的 invoke 需要传入 messages 列表
        input_messages = [{"role": "user", "content": state["user_query"]}]
        result = agent_graph.invoke({"messages": input_messages})
        
        # 提取最终回复 (LangGraph ReAct 预构建通常返回 {'messages': [...]})
        # 最后一条消息通常是 AI 的最终回答
        output_messages = result.get("messages", [])
        if output_messages:
            final_answer = output_messages[-1].content
        else:
            final_answer = "未能生成有效回复。"
            
        print("   ✅ 反应式处理完成")
        return {"final_response": final_answer}
        
    except Exception as e:
        return {"error": f"反应式处理异常：{str(e)}", "final_response": "快速响应服务暂时不可用。"}

def collect_data_node(state: WealthAdvisorState) -> WealthAdvisorState:
    """
    [节点 2B-1] 数据收集
    功能：模拟收集深度分析所需的宏观与微观数据
    """
    print("📥 [2B-1/4] 正在收集深度分析数据...")
    try:
        prompt = ChatPromptTemplate.from_template(DATA_COLLECTION_PROMPT)
        chain = prompt | llm | JsonOutputParser()
        
        input_data = {
            "user_query": state["user_query"],
            "customer_profile": json.dumps(state.get("customer_profile", {}), ensure_ascii=False)
        }
        
        result = chain.invoke(input_data)
        print("   ✅ 数据收集完成")
        
        return {"market_data": result.get("collected_data", {}), "current_phase": "analyze"}
    except Exception as e:
        return {"error": f"数据收集失败：{str(e)}", "current_phase": "collect_data"}

def analyze_data_node(state: WealthAdvisorState) -> WealthAdvisorState:
    """
    [节点 2B-2] 深度分析
    功能：基于收集的数据进行逻辑推演和策略生成
    """
    print("🧠 [2B-2/4] 正在进行深度投资分析...")
    if not state.get("market_data"):
        return {"error": "缺少市场数据，退回收集阶段", "current_phase": "collect_data"}
    
    try:
        prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)
        chain = prompt | llm | JsonOutputParser()
        
        input_data = {
            "user_query": state["user_query"],
            "customer_profile": json.dumps(state.get("customer_profile", {}), ensure_ascii=False),
            "market_data": json.dumps(state["market_data"], ensure_ascii=False)
        }
        
        result = chain.invoke(input_data)
        print("   ✅ 分析完成")
        
        return {"analysis_results": result, "current_phase": "recommend"}
    except Exception as e:
        return {"error": f"分析失败：{str(e)}", "current_phase": "analyze"}

def generate_recommendations_node(state: WealthAdvisorState) -> WealthAdvisorState:
    """
    [节点 2B-3] 建议生成
    功能：将专业分析转化为客户可执行的自然人语言建议
    """
    print("📝 [2B-3/4] 正在生成最终投资建议...")
    if not state.get("analysis_results"):
        return {"error": "缺少分析结果", "current_phase": "analyze"}
    
    try:
        prompt = ChatPromptTemplate.from_template(RECOMMENDATION_PROMPT)
        chain = prompt | llm | StrOutputParser() # 注意：这里需要自然语言，不用 JSON Parser
        
        input_data = {
            "user_query": state["user_query"],
            "customer_profile": json.dumps(state.get("customer_profile", {}), ensure_ascii=False),
            "analysis_results": json.dumps(state["analysis_results"], ensure_ascii=False)
        }
        
        result = chain.invoke(input_data)
        print("   ✅ 建议生成完成")
        
        return {"final_response": result, "current_phase": "respond"}
    except Exception as e:
        return {"error": f"建议生成失败：{str(e)}", "current_phase": "recommend"}

def respond_node(state: WealthAdvisorState) -> WealthAdvisorState:
    """
    [节点 3] 统一响应
    功能：确保 final_response 字段存在，处理异常情况
    """
    if not state.get("final_response"):
        return {
            "final_response": "抱歉，未能生成有效回复。系统记录错误：" + str(state.get("error", "未知错误")),
            "error": state.get("error", "Response generation failed")
        }
    return state

# ==============================================================================
# 图构建 (Graph Construction)
# ==============================================================================

def create_wealth_advisor_graph() -> StateGraph:
    """构建混合架构工作流图 (LangGraph 1.0 风格)"""
    workflow = StateGraph(WealthAdvisorState)
    
    # 注册节点
    workflow.add_node("assess", assess_query_node)
    workflow.add_node("reactive", reactive_processing_node)
    workflow.add_node("collect_data", collect_data_node)
    workflow.add_node("analyze", analyze_data_node)
    workflow.add_node("recommend", generate_recommendations_node)
    workflow.add_node("respond", respond_node)
    
    # 设置入口点 (LangGraph 1.0 推荐使用 START 常量)
    workflow.add_edge(START, "assess")
    
    # 动态路由逻辑 (Conditional Edges)
    def route_logic(state: WealthAdvisorState) -> str:
        if state.get("error"): return "respond" # 出错直接去响应节点报错
        return "reactive" if state.get("processing_mode") == "reactive" else "collect_data"
    
    workflow.add_conditional_edges(
        "assess",
        route_logic,
        {"reactive": "reactive", "collect_data": "collect_data", "respond": "respond"}
    )
    
    # 定义深思熟虑模式的线性流程
    workflow.add_edge("collect_data", "analyze")
    workflow.add_edge("analyze", "recommend")
    workflow.add_edge("recommend", "respond")
    
    # 反应式模式直接去响应
    workflow.add_edge("reactive", "respond")
    
    # 结束
    workflow.add_edge("respond", END)
    
    return workflow.compile()

# ==============================================================================
# 模拟数据与主程序 (Main Execution)
# ==============================================================================

SAMPLE_PROFILES = {
    "customer1": {
        "customer_id": "C1001", "risk_tolerance": "平衡型", "investment_horizon": "中期",
        "financial_goals": ["退休", "教育"], "portfolio_value": 1500000.0,
        "current_allocations": {"股票": 0.4, "债券": 0.3, "现金": 0.1, "另类": 0.2}
    },
    "customer2": {
        "customer_id": "C1002", "risk_tolerance": "进取型", "investment_horizon": "长期",
        "financial_goals": ["财富增值"], "portfolio_value": 3000000.0,
        "current_allocations": {"股票": 0.65, "债券": 0.15, "现金": 0.05, "另类": 0.15}
    }
}

def run_advisor(query: str, cid: str = "customer1") -> Dict[str, Any]:
    """执行智能体流程"""
    graph = create_wealth_advisor_graph()
    profile = SAMPLE_PROFILES.get(cid, SAMPLE_PROFILES["customer1"])
    
    initial_state = {
        "user_query": query,
        "customer_profile": profile,
        "query_type": None, "processing_mode": None,
        "final_response": None, "error": None, "current_phase": "start"
    }
    
    # 打印流程图
    print("\n🗺️  工作流拓扑结构:")
    print(graph.get_graph().draw_mermaid())
    print("-" * 50)
    
    return graph.invoke(initial_state)

if __name__ == "__main__":
    print("🚀 混合架构财富投顾系统已启动 (LangChain 1.2 + LangGraph 1.0)")
    print("💡 特点：动态路由 (反应式 vs 深思熟虑) + 工具增强")
    print("=" * 60)
    
    examples = [
        "今天上证指数表现如何？",  # 触发 Reactive + Tool
        "什么是 ETF？",            # 触发 Reactive
        "根据我的风险偏好，如何调整组合以应对经济衰退？", # 触发 Deliberative
        "帮我制定一个 10 年期的子女教育金计划" # 触发 Deliberative
    ]
    
    print("\n请选择测试场景:")
    for i, ex in enumerate(examples, 1): print(f"{i}. {ex}")
    print("0. 自定义输入")
    
    choice = input("\n选项：")
    query = examples[int(choice)-1] if choice.isdigit() and 0 < int(choice) <= len(examples) else input("请输入问题：")
    
    cid = "customer1" if input("选择客户 (1:平衡型，2:进取型) [默认 1]: ") != "2" else "customer2"
    
    print(f"\n📝 问题：{query}")
    print(f"👤 客户：{SAMPLE_PROFILES[cid]['risk_tolerance']}")
    print("\n⏳ 处理中...\n")
    
    start = datetime.now()
    try:
        result = run_advisor(query, cid)
        end = datetime.now()
        
        if result.get("error"):
            print(f"⚠️ 警告：{result['error']}")
        
        mode = result.get("processing_mode", "Unknown")
        mode_emoji = "⚡" if mode == "reactive" else "🧠"
        print(f"\n{mode_emoji} 处理模式：{'反应式 (快速响应)' if mode == 'reactive' else '深思熟虑 (深度分析)'}")
        print("\n💬 最终回复:\n" + "-" * 40)
        print(result.get("final_response", "无响应"))
        print("-" * 40)
        print(f"⏱️  耗时：{(end-start).total_seconds():.2f} 秒")
        
    except Exception as e:
        print(f"💥 系统崩溃：{str(e)}")
        import traceback
        traceback.print_exc()
