#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
模块名称：深思熟虑投研智能体 (Deliberative Investment Research Agent)
基于框架：Qwen-Agent (阿里云通义千问智能体框架)

功能描述：
    本模块实现了一个具备“深度思考”能力的专业投资研究智能体。
    不同于传统的单步问答，该智能体模拟人类分析师的思维链路，强制通过
    五个严谨的阶段来处理复杂的研究需求，确保输出结果的逻辑性、深度和可靠性。

    核心优势：
    1. 【思维链固化】将分析过程固化为“感知→建模→推理→决策→报告”的标准作业程序 (SOP)。
    2. 【上下文记忆】通过会话隔离机制 (_last_analysis_dict)，在多轮对话或工具调用间保持状态一致。
    3. 【双重模式】支持“分步交互式分析”（用户可干预每一步）和“一键完整分析”（高效生成报告）。
    4. 【容错处理】内置 JSON 解析防御机制，防止因模型输出格式微调导致的流程中断。

核心工作流程 (The 5-Step Deliberative Process)：
    ┌──────────────────────────────────────────────────────────────────────┐
    │  1. 感知 (Perception)                                                │
    │     ├─ 动作：收集市场概况、关键指标 (GDP/CPI/PMI)、新闻、行业趋势      │
    │     └─ 产出：原始情报数据包 (perception_data)                        │
    └──────────────────────────────────────────────────────────────────────┘
                                   ↓
    ┌──────────────────────────────────────────────────────────────────────┐
    │  2. 建模 (Modeling)                                                  │
    │     ├─ 动作：基于情报构建内部世界模型，判断经济周期、风险与机会        │
    │     └─ 产出：结构化市场模型 (world_model)                            │
    └──────────────────────────────────────────────────────────────────────┘
                                   ↓
    ┌──────────────────────────────────────────────────────────────────────┐
    │  3. 推理 (Reasoning)                                                 │
    │     ├─ 动作：发散思维，生成 3 个不同策略的候选方案 (成长/价值/创新)     │
    │     └─ 产出：候选方案列表及置信度评估 (reasoning_plans)               │
    └──────────────────────────────────────────────────────────────────────┘
                                   ↓
    ┌──────────────────────────────────────────────────────────────────────┐
    │  4. 决策 (Decision)                                                  │
    │     ├─ 动作：收敛思维，评估优劣，选定最优策略并制定配置建议            │
    │     └─ 产出：最终投资决策书 (selected_plan)                          │
    └──────────────────────────────────────────────────────────────────────┘
                                   ↓
    ┌──────────────────────────────────────────────────────────────────────┐
    │  5. 报告 (Reporting)                                                 │
    │     ├─ 动作：整合全链路数据，撰写结构化、专业化的 Markdown 研报         │
    │     └─ 产出：完整投资研究报告 (文件保存 + 文本输出)                    │
    └──────────────────────────────────────────────────────────────────────┘

依赖环境：
    - Python 3.8+
    - qwen-agent (阿里云官方智能体 SDK)
    - dashscope (通义千问 API SDK)
    - matplotlib (用于配置中文字体，虽本例未绘图但保留以防扩展)
    - 环境变量：DASHSCOPE_API_KEY

使用说明：
    1. 确保已设置环境变量：export DASHSCOPE_API_KEY="your_key"
    2. 运行脚本默认启动 WebUI 图形界面。
    3. 在终端输入 'quit' 可退出 TUI 模式（若修改入口为 TUI）。
================================================================================
"""

import os
import json
import dashscope
from datetime import datetime
from typing import Dict, List, Any, Optional

# Qwen-Agent 核心组件
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool

from dotenv import load_dotenv
load_dotenv()

# ==============================================================================
# 全局配置与初始化
# ==============================================================================

# 配置 Matplotlib 中文字体，防止后续扩展绘图功能时出现乱码
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置资源根目录
ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

# 初始化 DashScope API
# 优先读取环境变量，若无则留空（会在运行时报错提示）
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
dashscope.timeout = 30  # 设置请求超时时间为 30 秒

# ==============================================================================
# 会话状态管理 (Session State Management)
# ==============================================================================

# 用于在不同工具调用间共享中间分析结果
# 结构：{ session_id: { 'perception_data': ..., 'world_model': ..., ... } }
_last_analysis_dict: Dict[int, Dict[str, Any]] = {}

def get_session_id(kwargs: Dict) -> Optional[int]:
    """
    根据传入的 kwargs 提取当前会话的唯一标识符。
    利用 messages 对象的内存地址 ID 作为临时的 Session ID，实现会话隔离。
    """
    messages = kwargs.get('messages')
    if messages is not None:
        return id(messages)
    return None

# ==============================================================================
# 系统提示词 (System Prompt)
# ==============================================================================

system_prompt = """
你是由深思熟虑型智能体架构驱动的专业投资研究助手。
你的核心任务是协助用户完成高质量的投资研究报告。

【工作流程规范】
你必须严格遵循以下五个阶段进行分析，不可跳跃：
1. **感知 (Perception)**：全面收集市场数据、宏观指标及行业动态。
2. **建模 (Modeling)**：基于数据构建市场世界观，识别周期位置与风险机会。
3. **推理 (Reasoning)**：发散思维，生成至少三个不同视角的投资策略方案。
4. **决策 (Decision)**：收敛思维，评估方案优劣，选定最优解并给出配置建议。
5. **报告 (Reporting)**：整合上述所有步骤的成果，输出结构严谨的专业研报。

【工具使用策略】
- **一键模式**：若用户希望快速获得结果，请优先调用 `complete_analysis` 工具，它将自动执行全流程。
- **分步模式**：若用户希望深入探讨某一步骤，可按顺序调用 `market_perception` -> `market_modeling` -> `investment_reasoning` -> `investment_decision` -> `generate_report`。

【回答风格】
- 专业、客观、数据驱动。
- 严禁编造数据，若缺乏具体实时数据，请基于逻辑推演并注明“模拟推演”。
- 输出报告必须包含风险提示。
"""

# ==============================================================================
# 工具定义层 (Tool Definitions)
# 每个工具对应思维链中的一个节点
# ==============================================================================

# --- 阶段 1: 感知 ---
@register_tool('market_perception')
class MarketPerceptionTool(BaseTool):
    """感知阶段：收集市场数据和信息"""
    description = '收集市场概况、关键经济指标、重要新闻及行业趋势，为后续分析提供数据基础。'
    parameters = [{
        'name': 'research_topic', 'type': 'string', 'description': '研究主题', 'required': True
    }, {
        'name': 'industry_focus', 'type': 'string', 'description': '行业焦点', 'required': True
    }, {
        'name': 'time_horizon', 'type': 'string', 'description': '时间范围 (短期/中期/长期)', 'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        topic, industry, horizon = args['research_topic'], args['industry_focus'], args['time_horizon']
        session_id = get_session_id(kwargs)

        # [模拟] 实际场景中此处应调用搜索 API 或数据库
        perception_data = {
            "market_overview": f"针对{topic}的分析显示，{industry}领域在{horizon}内呈现稳步增长态势。",
            "key_indicators": {
                "GDP 增长率": "5.2% (经济基本面稳健)",
                "CPI 指数": "2.1% (通胀可控)",
                "PMI 指数": "51.2 (制造业扩张)",
                "市场情绪": "中性偏乐观"
            },
            "recent_news": [
                f"政策面持续利好{industry}发展",
                f"{industry}核心技术取得突破性进展",
                "全球供应链重构带来新机遇"
            ],
            "industry_trends": {
                "技术迭代": "加速向智能化、绿色化转型",
                "竞争格局": "头部效应显著，马太效应加剧",
                "政策环境": "监管框架日益完善"
            }
        }

        # 状态持久化
        if session_id:
            _last_analysis_dict.setdefault(session_id, {})['perception_data'] = perception_data

        # 格式化输出供 LLM 阅读
        return self._format_output("感知阶段完成", perception_data)

    def _format_output(self, title: str, data: dict) -> str:
        """辅助方法：格式化字典数据为 Markdown 字符串"""
        lines = [f"## {title}"]
        lines.append(f"**市场概况**: {data['market_overview']}")
        lines.append("**关键指标**:")
        lines.extend([f"- {k}: {v}" for k, v in data['key_indicators'].items()])
        lines.append("**重要新闻**:")
        lines.extend([f"- {n}" for n in data['recent_news']])
        lines.append("**行业趋势**:")
        lines.extend([f"- {k}: {v}" for k, v in data['industry_trends'].items()])
        return "\n".join(lines)

# --- 阶段 2: 建模 ---
@register_tool('market_modeling')
class MarketModelingTool(BaseTool):
    """建模阶段：构建内部世界模型"""
    description = '基于感知数据，分析市场状态、经济周期、风险因素及潜在机会，构建结构化世界模型。'
    parameters = [{
        'name': 'research_topic', 'type': 'string', 'description': '研究主题', 'required': True
    }, {
        'name': 'industry_focus', 'type': 'string', 'description': '行业焦点', 'required': True
    }, {
        'name': 'time_horizon', 'type': 'string', 'description': '时间范围', 'required': True
    }, {
        'name': 'perception_data', 'type': 'string', 'description': '上一阶段输出的 JSON 数据', 'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        # 安全解析上游数据，防止格式错误导致流程中断
        try:
            perception_data = json.loads(args['perception_data'])
        except (json.JSONDecodeError, TypeError):
            perception_data = {} # 降级处理
            
        session_id = get_session_id(kwargs)
        industry, horizon = args['industry_focus'], args['time_horizon']

        # [逻辑] 基于感知数据构建模型
        world_model = {
            "market_state": f"{industry}处于成长期向成熟期过渡阶段，{horizon}内预计保持 CAGR 15%+ 增长。",
            "economic_cycle": "复苏后期至过热初期，流动性充裕但边际收紧。",
            "risk_factors": ["技术路线颠覆风险", "地缘政治影响", "原材料价格波动", "估值回调压力"],
            "opportunity_areas": ["国产替代加速", "出海业务拓展", "数字化转型红利", "并购整合机会"],
            "market_sentiment": "长期看好，短期存在博弈震荡。"
        }

        if session_id:
            _last_analysis_dict.setdefault(session_id, {})['world_model'] = world_model

        return (f"## 建模阶段完成\n\n"
                f"**市场状态**: {world_model['market_state']}\n"
                f"**经济周期**: {world_model['economic_cycle']}\n"
                f"**风险因素**: {', '.join(world_model['risk_factors'])}\n"
                f"**机会领域**: {', '.join(world_model['opportunity_areas'])}\n"
                f"**市场情绪**: {world_model['market_sentiment']}")

# --- 阶段 3: 推理 ---
@register_tool('investment_reasoning')
class InvestmentReasoningTool(BaseTool):
    """推理阶段：生成候选投资方案"""
    description = '基于世界模型，发散生成多个（通常 3 个）不同策略的投资方案，并评估其优劣势。'
    parameters = [{
        'name': 'research_topic', 'type': 'string', 'description': '研究主题', 'required': True
    }, {
        'name': 'industry_focus', 'type': 'string', 'description': '行业焦点', 'required': True
    }, {
        'name': 'time_horizon', 'type': 'string', 'description': '时间范围', 'required': True
    }, {
        'name': 'world_model', 'type': 'string', 'description': '上一阶段输出的 JSON 数据', 'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        session_id = get_session_id(kwargs)
        horizon = args['time_horizon']

        # [逻辑] 生成差异化策略
        reasoning_plans = [
            {
                "plan_id": "growth_focused", "hypothesis": "高增长驱动，拥抱龙头",
                "confidence_level": 0.75,
                "pros": ["收益弹性大", "顺应产业趋势"], "cons": ["估值容忍度低", "波动大"]
            },
            {
                "plan_id": "value_focused", "hypothesis": "低估值修复，防守反击",
                "confidence_level": 0.65,
                "pros": ["安全边际高", "下行风险小"], "cons": ["上涨爆发力弱", "时间成本高"]
            },
            {
                "plan_id": "innovation_focused", "hypothesis": "押注技术变革，高风险高回报",
                "confidence_level": 0.55,
                "pros": ["潜在十倍股", "先发优势"], "cons": ["失败率高", "技术不确定性"]
            }
        ]

        if session_id:
            _last_analysis_dict.setdefault(session_id, {})['reasoning_plans'] = reasoning_plans

        # 格式化输出
        result = ["## 推理阶段完成\n\n**候选方案对比**:\n"]
        for i, plan in enumerate(reasoning_plans, 1):
            result.append(f"### 方案{i}: {plan['plan_id']} (置信度:{plan['confidence_level']})")
            result.append(f"- **假设**: {plan['hypothesis']}")
            result.append(f"- **优势**: {', '.join(plan['pros'])}")
            result.append(f"- **劣势**: {', '.join(plan['cons'])}\n")
        return "\n".join(result)

# --- 阶段 4: 决策 ---
@register_tool('investment_decision')
class InvestmentDecisionTool(BaseTool):
    """决策阶段：选择最优方案"""
    description = '综合评估候选方案，选定最优投资策略，形成具体的配置建议和风控措施。'
    parameters = [{
        'name': 'research_topic', 'type': 'string', 'description': '研究主题', 'required': True
    }, {
        'name': 'industry_focus', 'type': 'string', 'description': '行业焦点', 'required': True
    }, {
        'name': 'time_horizon', 'type': 'string', 'description': '时间范围', 'required': True
    }, {
        'name': 'world_model', 'type': 'string', 'description': '世界模型 JSON', 'required': True
    }, {
        'name': 'reasoning_plans', 'type': 'string', 'description': '候选方案 JSON', 'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        session_id = get_session_id(kwargs)
        industry, horizon = args['industry_focus'], args['time_horizon']

        # [逻辑] 默认选择成长型策略作为最优解（可根据模型动态调整）
        selected_plan = {
            "selected_plan_id": "growth_focused",
            "investment_thesis": f"鉴于{industry}的高确定性与政策共振，成长型策略性价比最高。",
            "supporting_evidence": ["行业渗透率快速提升", "龙头企业护城河加深", "盈利增速匹配估值"],
            "risk_assessment": "需警惕宏观流动性收紧带来的估值杀跌，建议设置 15% 止损线。",
            "recommendation": f"建议{horizon}内采取'核心 + 卫星'策略：70% 配置龙头白马，30% 配置弹性标的。",
            "timeframe": f"{horizon}周期，季度再平衡。"
        }

        if session_id:
            _last_analysis_dict.setdefault(session_id, {})['selected_plan'] = selected_plan

        return (f"## 决策阶段完成\n\n"
                f"**选定策略**: {selected_plan['selected_plan_id']}\n"
                f"**核心论点**: {selected_plan['investment_thesis']}\n"
                f"**配置建议**: {selected_plan['recommendation']}\n"
                f"**风控措施**: {selected_plan['risk_assessment']}")

# --- 阶段 5: 报告生成 ---
@register_tool('generate_report')
class GenerateReportTool(BaseTool):
    """报告阶段：生成完整研报"""
    description = '整合前四个阶段的所有成果，撰写一份结构完整、格式专业的投资研究报告。'
    parameters = [{
        'name': 'research_topic', 'type': 'string', 'description': '研究主题', 'required': True
    }, {
        'name': 'industry_focus', 'type': 'string', 'description': '行业焦点', 'required': True
    }, {
        'name': 'time_horizon', 'type': 'string', 'description': '时间范围', 'required': True
    }, {
        'name': 'perception_data', 'type': 'string', 'description': '感知数据 JSON', 'required': True
    }, {
        'name': 'world_model', 'type': 'string', 'description': '世界模型 JSON', 'required': True
    }, {
        'name': 'selected_plan', 'type': 'string', 'description': '最终决策 JSON', 'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        topic, industry, horizon = args['research_topic'], args['industry_focus'], args['time_horizon']
        
        # 安全解析所有上游数据
        def safe_load(key):
            try: return json.loads(args.get(key, '{}'))
            except: return {}
            
        perception = safe_load('perception_data')
        model = safe_load('world_model')
        decision = safe_load('selected_plan')
        
        timestamp = datetime.now().strftime("%Y年%m月%d日")
        filename = f"研报_{topic[:10]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # 构建 Markdown 报告内容
        report_content = f"""
# {topic} 深度投资研究报告

**生成时间**: {timestamp} | **行业**: {industry} | **周期**: {horizon}

---

## 1. 执行摘要
{decision.get('investment_thesis', '暂无核心论点')}
> **核心建议**: {decision.get('recommendation', '暂无具体建议')}

## 2. 市场环境感知
- **概况**: {perception.get('market_overview', '无数据')}
- **关键指标**: 
{chr(10).join([f"  - {k}: {v}" for k,v in perception.get('key_indicators', {}).items()])}
- **最新动态**: {', '.join(perception.get('recent_news', []))}

## 3. 深度建模分析
- **市场状态**: {model.get('market_state', '无数据')}
- **风险预警**: {', '.join(model.get('risk_factors', []))}
- **机会洞察**: {', '.join(model.get('opportunity_areas', []))}

## 4. 投资决策详情
- **选定策略**: {decision.get('selected_plan_id', '未知')}
- **支撑证据**: 
{chr(10).join([f"  - {e}" for e in decision.get('supporting_evidence', [])])}
- **风险评估**: {decision.get('risk_assessment', '无')}

## 5. 结论与风险提示
投资有风险，入市需谨慎。本报告基于 AI 深思熟虑流程生成，仅供参考，不构成绝对投资建议。
---
*Generated by Deliberative Investment Research Agent*
"""
        # 保存文件
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_content)
            file_msg = f"\n📄 **报告已保存至**: `{filename}`"
        except Exception as e:
            file_msg = f"\n⚠️ 报告保存失败：{str(e)}"

        return f"## 报告生成完成{file_msg}\n\n{report_content}"

# --- 额外工具：一键全流程 ---
@register_tool('complete_analysis')
class CompleteAnalysisTool(BaseTool):
    """一键完成：自动串联感知、建模、推理、决策、报告全过程"""
    description = '高效模式：一次性执行全部五个分析阶段，直接输出完整研报，无需分步调用。'
    parameters = [{
        'name': 'research_topic', 'type': 'string', 'description': '研究主题', 'required': True
    }, {
        'name': 'industry_focus', 'type': 'string', 'description': '行业焦点', 'required': True
    }, {
        'name': 'time_horizon', 'type': 'string', 'description': '时间范围', 'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        """
        内部模拟执行全流程，复用各阶段逻辑，最后调用报告生成逻辑。
        为避免代码重复，此处直接内联了各阶段的数据生成逻辑（生产环境中可调用类方法）。
        """
        args = json.loads(params)
        topic, industry, horizon = args['research_topic'], args['industry_focus'], args['time_horizon']
        session_id = get_session_id(kwargs)

        # [模拟全流程数据生成] (简化版，实际可调用上述类的内部逻辑)
        perception_data = {"market_overview": f"{topic}全景分析...", "key_indicators": {"GDP": "5.2%"}, "recent_news": ["新闻 A", "新闻 B"], "industry_trends": {"趋势": "向上"}}
        world_model = {"market_state": "成长期", "risk_factors": ["风险 A"], "opportunity_areas": ["机会 A"]}
        selected_plan = {"investment_thesis": "强烈推荐", "recommendation": "买入", "supporting_evidence": ["证据 1"], "risk_assessment": "注意波动", "selected_plan_id": "Growth"}

        # 更新会话状态
        if session_id:
            _last_analysis_dict[session_id] = {
                'perception_data': perception_data,
                'world_model': world_model,
                'selected_plan': selected_plan
            }

        # 构造 generate_report 所需的参数字符串
        report_params = json.dumps({
            "research_topic": topic,
            "industry_focus": industry,
            "time_horizon": horizon,
            "perception_data": json.dumps(perception_data),
            "world_model": json.dumps(world_model),
            "selected_plan": json.dumps(selected_plan)
        })
        
        # 复用报告生成逻辑
        report_tool = GenerateReportTool()
        # 注意：这里需要构造符合 call 签名的参数，由于是直接调用，我们模拟一下
        # 为了简化，直接返回一个简化的成功消息，实际会触发上面的逻辑
        return f"## 完整分析执行中...\n\n正在为您生成 {topic} 的深度报告...\n\n(系统内部已自动串联 5 个阶段)\n\n" + \
               GenerateReportTool().call(report_params, **kwargs)

# ==============================================================================
# 服务启动与入口 (Service Entry Points)
# ==============================================================================

def init_agent_service():
    """初始化 Qwen-Agent 助手实例"""
    llm_cfg = {
        'model': 'qwen3-max', # 使用较新的 Turbo 模型以平衡速度与成本
        'timeout': 30,
        'retry_count': 3,
    }
    
    # 注册所有可用工具
    tool_names = [
        'market_perception', 'market_modeling', 'investment_reasoning', 
        'investment_decision', 'generate_report', 'complete_analysis'
    ]

    return Assistant(
        llm=llm_cfg,
        name='深思熟虑投研助手',
        description='专业的一站式投资研究分析智能体',
        system_message=system_prompt,
        function_list=tool_names,
    )

def app_gui():
    """启动 Web 图形界面 (基于 Gradio/Ray 封装)"""
    print("🚀 正在启动深思熟虑投研智能体 Web 界面...")
    bot = init_agent_service()
    
    # 配置预设问题，引导用户提问
    chatbot_config = {
        'prompt.suggestions': [
            '请分析新能源汽车行业在中期时间框架下的投资机会',
            '帮我研究人工智能技术在短期内的投资前景',
            '分析医疗健康行业在长期投资周期中的发展机会',
            '对半导体行业进行完整的深思熟虑分析'
        ]
    }
    
    try:
        WebUI(bot, chatbot_config=chatbot_config).run()
    except Exception as e:
        print(f"❌ Web 界面启动失败：{e}")
        print("请检查网络连接及 DASHSCOPE_API_KEY 配置。")

if __name__ == '__main__':
    # 默认启动图形界面，如需终端模式可修改为 app_tui() (需自行实现 TUI 逻辑)
    app_gui()
