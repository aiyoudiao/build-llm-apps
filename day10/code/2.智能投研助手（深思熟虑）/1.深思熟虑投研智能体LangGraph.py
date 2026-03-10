#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
模块名称：深思熟虑投研智能体 (LangGraph 架构版)
功能描述：
    基于 LangGraph 框架构建的强类型、多阶段智能工作流。
    本版本专为 LangChain 0.2.x 生态优化，将“感知 - 建模 - 推理 - 决策 - 报告”
    五步法固化为确定的状态机流转，确保长链路任务不丢失上下文。

    相比 ReAct 架构，本版本的优势在于：
    1. 确定性：严格执行预设流程，避免模型跳过关键分析步骤。
    2. 状态持久化：通过 TypedDict 显式管理中间产物，便于调试和断点续传。
    3. 可观测性：原生支持 Mermaid 流程图生成，清晰展示数据流向。

核心工作流程 (StateGraph)：
    [Start] 
       ↓
    [Perception Node] (收集情报) --(更新 state.perception_data)--> 
       ↓
    [Modeling Node]   (构建模型) --(更新 state.world_model)--> 
       ↓
    [Reasoning Node]  (发散方案) --(更新 state.reasoning_plans)--> 
       ↓
    [Decision Node]   (收敛决策) --(更新 state.selected_plan)--> 
       ↓
    [Report Node]     (撰写报告) --(更新 state.final_report)--> 
       ↓
    [End]

依赖环境：
    - Python 3.9+
    - langchain == 0.2.5
    - langchain-core == 0.2.9
    - langchain-community == 0.2.5
    - langgraph == 0.1.9 (关键依赖)
    - dashscope
    - 环境变量：DASHSCOPE_API_KEY
================================================================================
"""

import os
import json
from typing import Dict, List, Any, Literal, TypedDict, Optional
from datetime import datetime
from dotenv import load_dotenv

# ==============================================================================
# LangChain & LangGraph 核心组件导入 (0.2.x 风格)
# ==============================================================================

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END

load_dotenv()

# ==============================================================================
# 配置与初始化
# ==============================================================================

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("❌ 错误：未找到环境变量 DASHSCOPE_API_KEY")

# 初始化模型 (使用 qwen-plus 增强逻辑推理)
llm = ChatTongyi(model_name="qwen-plus", dashscope_api_key=DASHSCOPE_API_KEY)

# ==============================================================================
# 状态定义 (State Schema)
# 使用 TypedDict 定义全局共享状态，确保类型安全
# ==============================================================================

class ResearchAgentState(TypedDict):
    """
    智能体运行时状态容器
    所有节点共享此状态，通过返回字典更新特定字段
    """
    # [输入参数]
    research_topic: str
    industry_focus: str
    time_horizon: str
    
    # [中间产物] (由各个节点逐步填充)
    perception_data: Optional[Dict[str, Any]]
    world_model: Optional[Dict[str, Any]]
    reasoning_plans: Optional[List[Dict[str, Any]]]
    selected_plan: Optional[Dict[str, Any]]
    
    # [最终产出]
    final_report: Optional[str]
    
    # [控制流]
    current_phase: Literal["perception", "modeling", "reasoning", "decision", "report", "completed"]
    error: Optional[str]

# ==============================================================================
# 提示词模板 (Prompts)
# 保持简洁，专注于指令清晰
# ==============================================================================

PERCEPTION_PROMPT = """你是一级市场研究员。请针对主题 "{topic}" (行业:{industry}, 视角:{horizon}) 进行市场感知。
输出 JSON：{{ "market_overview": "...", "key_indicators": {{...}}, "recent_news": [...], "industry_trends": {{...}} }}"""

MODELING_PROMPT = """你是宏观策略师。基于以下数据构建市场模型：{data}
输出 JSON：{{ "market_state": "...", "economic_cycle": "...", "risk_factors": [...], "opportunity_areas": [...], "market_sentiment": "..." }}"""

REASONING_PROMPT = """你是首席投资官。基于模型 {model} 生成 3 个不同的投资方案。
输出 JSON 数组：[{{ "plan_id": "...", "hypothesis": "...", "confidence_level": 0.0-1.0, "pros": [...], "cons": [...] }}, ...]"""

DECISION_PROMPT = """你是投决会主席。评估方案 {plans}，选出最优解。
输出 JSON：{{ "selected_plan_id": "...", "investment_thesis": "...", "recommendation": "...", "timeframe": "..." }}"""

REPORT_PROMPT = """你是首席分析师。整合以下信息撰写研报：
1. 背景：{perception}
2. 模型：{model}
3. 决策：{decision}
要求：结构完整（标题/摘要/观点/论证/风险/结论），专业客观，Markdown 格式。"""

# ==============================================================================
# 工作流节点 (Nodes)
# 每个函数接收当前 State，返回需要更新的字段字典
# ==============================================================================

def perception_node(state: ResearchAgentState) -> Dict[str, Any]:
    """节点 1: 感知 - 收集市场情报"""
    print(">>> [1/5] 正在感知市场环境...")
    try:
        prompt = ChatPromptTemplate.from_template(PERCEPTION_PROMPT)
        chain = prompt | llm | JsonOutputParser()
        
        result = chain.invoke({
            "topic": state["research_topic"],
            "industry": state["industry_focus"],
            "horizon": state["time_horizon"]
        })
        
        return {
            "perception_data": result,
            "current_phase": "modeling"
        }
    except Exception as e:
        return {"error": f"感知失败: {str(e)}", "current_phase": "perception"}

def modeling_node(state: ResearchAgentState) -> Dict[str, Any]:
    """节点 2: 建模 - 构建分析框架"""
    print(">>> [2/5] 正在构建市场模型...")
    if not state.get("perception_data"):
        return {"error": "缺失感知数据", "current_phase": "perception"}
    
    try:
        prompt = ChatPromptTemplate.from_template(MODELING_PROMPT)
        chain = prompt | llm | JsonOutputParser()
        
        result = chain.invoke({
            "data": json.dumps(state["perception_data"], ensure_ascii=False)
        })
        
        return {
            "world_model": result,
            "current_phase": "reasoning"
        }
    except Exception as e:
        return {"error": f"建模失败: {str(e)}", "current_phase": "modeling"}

def reasoning_node(state: ResearchAgentState) -> Dict[str, Any]:
    """节点 3: 推理 - 生成候选方案"""
    print(">>> [3/5] 正在推演投资方案...")
    if not state.get("world_model"):
        return {"error": "缺失世界模型", "current_phase": "modeling"}
    
    try:
        prompt = ChatPromptTemplate.from_template(REASONING_PROMPT)
        chain = prompt | llm | JsonOutputParser()
        
        result = chain.invoke({
            "model": json.dumps(state["world_model"], ensure_ascii=False)
        })
        
        return {
            "reasoning_plans": result,
            "current_phase": "decision"
        }
    except Exception as e:
        return {"error": f"推理失败: {str(e)}", "current_phase": "reasoning"}

def decision_node(state: ResearchAgentState) -> Dict[str, Any]:
    """节点 4: 决策 - 选定最优方案"""
    print(">>> [4/5] 正在进行投资决策...")
    if not state.get("reasoning_plans"):
        return {"error": "缺失候选方案", "current_phase": "reasoning"}
    
    try:
        prompt = ChatPromptTemplate.from_template(DECISION_PROMPT)
        chain = prompt | llm | JsonOutputParser()
        
        result = chain.invoke({
            "plans": json.dumps(state["reasoning_plans"], ensure_ascii=False)
        })
        
        return {
            "selected_plan": result,
            "current_phase": "report"
        }
    except Exception as e:
        return {"error": f"决策失败: {str(e)}", "current_phase": "decision"}

def report_node(state: ResearchAgentState) -> Dict[str, Any]:
    """节点 5: 报告 - 撰写最终研报"""
    print(">>> [5/5] 正在撰写研究报告...")
    if not state.get("selected_plan"):
        return {"error": "缺失最终决策", "current_phase": "decision"}
    
    try:
        prompt = ChatPromptTemplate.from_template(REPORT_PROMPT)
        # 报告是长文本，使用 StrOutputParser
        chain = prompt | llm | StrOutputParser()
        
        result = chain.invoke({
            "perception": json.dumps(state["perception_data"], ensure_ascii=False),
            "model": json.dumps(state["world_model"], ensure_ascii=False),
            "decision": json.dumps(state["selected_plan"], ensure_ascii=False)
        })
        
        return {
            "final_report": result,
            "current_phase": "completed"
        }
    except Exception as e:
        return {"error": f"报告失败: {str(e)}", "current_phase": "report"}

# ==============================================================================
# 图构建 (Graph Construction)
# 定义节点连接关系
# ==============================================================================

def build_research_workflow() -> StateGraph:
    """构建并编译 LangGraph 工作流"""
    
    # 初始化状态图
    workflow = StateGraph(ResearchAgentState)
    
    # 注册节点
    workflow.add_node("perception", perception_node)
    workflow.add_node("modeling", modeling_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("report", report_node)
    
    # 设置入口
    workflow.set_entry_point("perception")
    
    # 定义顺序边 (Sequential Edges)
    # 这里的逻辑是确定性的：A -> B -> C -> D -> E -> END
    workflow.add_edge("perception", "modeling")
    workflow.add_edge("modeling", "reasoning")
    workflow.add_edge("reasoning", "decision")
    workflow.add_edge("decision", "report")
    workflow.add_edge("report", END)
    
    # 编译为可执行应用
    return workflow.compile()

# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 深思熟虑投研智能体 (LangGraph 架构版) 已启动")
    print("💡 特点：确定性流程、状态持久化、防幻觉增强")
    print("=" * 70)

    try:
        # 1. 构建工作流
        app = build_research_workflow()
        
        # 2. 打印流程图 (Mermaid)
        print("\n📊 工作流拓扑结构:")
        print(app.get_graph().draw_mermaid())
        print("-" * 70)
        
        # 3. 用户交互
        topic = input("\n📝 研究主题 (例：固态电池技术突破): ").strip() or "固态电池技术突破"
        industry = input("🏭 行业焦点 (例：新能源汽车产业链): ").strip() or "新能源汽车"
        horizon = input("⏳ 时间范围 (短期/中期/长期): ").strip() or "中期"
        
        # 4. 初始化状态
        initial_state = {
            "research_topic": topic,
            "industry_focus": industry,
            "time_horizon": horizon,
            "perception_data": None,
            "world_model": None,
            "reasoning_plans": None,
            "selected_plan": None,
            "final_report": None,
            "current_phase": "perception",
            "error": None
        }
        
        print(f"\n⏳ 智能体开始执行全流程分析，请稍候...\n")
        
        # 5. 执行 invoke
        final_state = app.invoke(initial_state)
        
        # 6. 结果处理
        if final_state.get("error"):
            print(f"\n❌ 流程中断：{final_state['error']}")
        else:
            print("\n" + "=" * 70)
            print("📄 生成研究报告:")
            print("=" * 70)
            print(final_state["final_report"])
            
            # 保存文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"研报_{topic[:10]}_{timestamp}.md"
            # 简单清洗文件名
            filename = "".join([c for c in filename if c.isalnum() or c in ('_', '.', '-')])
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(final_state["final_report"])
            
            print(f"\n💾 报告已保存：{filename}")
            
    except Exception as e:
        print(f"\n💥 系统异常：{e}")
        import traceback
        traceback.print_exc()
