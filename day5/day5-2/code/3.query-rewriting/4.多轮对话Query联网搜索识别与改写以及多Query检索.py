"""
## 混合检索 RAG 系统 - 本地知识库 + 互联网搜索

结合 Query 改写、MultiQuery 检索、联网搜索判断的完整问答系统

系统架构：
  用户输入 → 联网搜索判断 → 路由决策
     │
     ├─ 需要联网 → 搜索引擎 API → 整合结果
     │
     └─ 不需要联网 → 本地 RAG 系统
          ├─ Query 改写（理解真实意图）
          ├─ MultiQuery 检索（提高召回率）
          └─ LLM 生成答案 + 页码溯源
核心特性：
  1. 智能路由：自动判断使用本地知识库还是互联网搜索
  2. 多轮对话：支持上下文理解、指代消解
  3. 高召回率：MultiQuery 生成多个相似查询
  4. 可溯源：答案可追溯到具体 PDF 页码
  5. 成本追踪：记录每次 API 调用的费用和 Token

┌─────────────────────────────────────────────────────────────────┐
│                    混合检索 RAG 系统                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  用户输入                                                        │
│     │                                                           │
│     ▼                                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ConversationManager  对话历史管理                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│     │                                                           │
│     ▼                                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  WebSearchJudger  联网搜索判断                          │   │
│  │  - 8 大场景识别（时效性、价格、营业、活动等）              │   │
│  └─────────────────────────────────────────────────────────┘   │
│     │                                                           │
│     ├─ 需要联网 ──────────────────────────────────┐           │
│     │   ▼                                         │           │
│     │  ┌─────────────────────────────────────┐   │           │
│     │  │ 改写搜索查询 + 生成搜索策略          │   │           │
│     │  │ 返回建议来源和关键词                 │   │           │
│     │  └─────────────────────────────────────┘   │           │
│     │                                           │           │
│     └─ 不需要联网 ──────────────────────────────┤           │
│         ▼                                       │           │
│      ┌─────────────────────────────────────┐   │           │
│      │  QueryRewriter  Query 改写           │   │           │
│      │  - 上下文依赖/对比/指代/多意图/反问  │   │           │
│      └─────────────────────────────────────┘   │           │
│         │                                       │           │
│         ▼                                       │           │
│      ┌─────────────────────────────────────┐   │           │
│      │  MultiQueryRetriever  多查询检索     │   │           │
│      │  - LLM 生成多个相似表述               │   │           │
│      │  - 分别检索向量库                    │   │           │
│      │  - 合并去重                          │   │           │
│      └─────────────────────────────────────┘   │           │
│         │                                       │           │
│         ▼                                       │           │
│      ┌─────────────────────────────────────┐   │           │
│      │  DocumentChain  LLM 生成答案          │   │           │
│      │  + 页码溯源                          │   │           │
│      └─────────────────────────────────────┘   │           │
│         │                                       │           │
│         └───────────────────┬───────────────────┘           │
│                             ▼                               │
│                    输出：答案 + 来源 + 成本信息                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""

# -----------------------------------------------------------------------------
# 导入依赖库
# -----------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# PDF 处理
from PyPDF2 import PdfReader

# LangChain 核心组件
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Tongyi
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import MultiQueryRetriever

# DashScope 原生 API（用于 Query 改写和联网搜索判断）
import dashscope
import json

# 工具库
from typing import List, Dict, Any, Set, Tuple, Optional
from datetime import datetime
import os
import re

# 成本追踪
from langchain_community.callbacks.manager import get_openai_callback


# =============================================================================
# 配置管理
# =============================================================================

class Config:
    """
    系统配置管理类
    
    集中管理所有配置参数，便于统一调整和版本控制
    """
    
    # API 配置
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    
    if not DASHSCOPE_API_KEY:
        raise ValueError("请先设置环境变量 DASHSCOPE_API_KEY")
    
    # 设置 DashScope API Key
    dashscope.api_key = DASHSCOPE_API_KEY
    
    # 模型配置
    EMBEDDING_MODEL = "text-embedding-v1"      # 向量嵌入模型
    LLM_MODEL = "deepseek-v3"                   # 问答生成模型
    REWRITE_MODEL = "qwen-turbo-latest"         # Query 改写模型
    
    # 文件路径配置
    PDF_PATH = "./浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf"
    VECTOR_DB_PATH = "./vector_db"
    
    # 文本处理配置
    CHUNK_SIZE = 1000          # 文本块大小（字符数）
    CHUNK_OVERLAP = 100        # 文本块重叠（字符数）
    
    # 检索配置
    TOP_K = 5                  # 每个 Query 检索的文档数量
    
    # 对话配置
    MAX_CONVERSATION_HISTORY = 10  # 保留的最大对话轮数
    
    # 联网搜索配置
    WEB_SEARCH_ENABLED = True      # 是否启用联网搜索功能
    WEB_SEARCH_API = None          # 搜索引擎 API（如需要可配置）


# =============================================================================
# 对话历史管理
# =============================================================================

class ConversationManager:
    """
    对话历史管理器
    
    功能：管理多轮对话的历史记录，支持添加、查询、限制长度等操作
    """
    
    def __init__(self, max_history: int = Config.MAX_CONVERSATION_HISTORY):
        """
        初始化对话管理器
        
        参数:
            max_history: 保留的最大对话轮数
        """
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []
    
    def add_turn(self, role: str, content: str) -> None:
        """
        添加一轮对话
        
        参数:
            role: 角色（user 或 assistant）
            content: 对话内容
        """
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # 如果超过最大长度，移除最早的对话
        while len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_formatted_history(self, include_last_user_query: bool = False) -> str:
        """
        获取格式化的对话历史字符串
        
        参数:
            include_last_user_query: 是否包含最后一轮用户查询
        
        返回:
            格式化的对话历史字符串
        """
        lines = []
        for turn in self.history:
            role_label = "用户" if turn["role"] == "user" else "AI"
            lines.append(f"{role_label}: \"{turn['content']}\"")
        
        return "\n".join(lines)
    
    def get_last_user_query(self) -> Optional[str]:
        """
        获取最后一次用户查询
        
        返回:
            最后一次用户查询内容，如果没有则返回 None
        """
        for turn in reversed(self.history):
            if turn["role"] == "user":
                return turn["content"]
        return None
    
    def clear(self) -> None:
        """
        清空对话历史
        """
        self.history = []


# =============================================================================
# Query 改写模块（多轮对话理解）
# =============================================================================

class QueryRewriter:
    """
    智能 Query 改写器（多轮对话场景）
    
    功能：将多轮对话中的复杂查询改写为清晰、独立、可检索的标准查询
    
    支持的改写类型：
        - 上下文依赖型：补全省略的上下文信息
        - 对比型：明确比较对象和维度
        - 模糊指代型：消除代词歧义
        - 多意图型：分解为多个独立问题
        - 反问型：提取真实意图
    """
    
    def __init__(self, model: str = Config.REWRITE_MODEL):
        """
        初始化 Query 改写器
        
        参数:
            model: 使用的 LLM 模型名称
        """
        self.model = model
    
    def _call_llm(self, prompt: str) -> str:
        """
        调用大语言模型生成文本回复
        
        参数:
            prompt: 用户输入的提示词或问题
        
        返回:
            模型生成的文本内容
        """
        messages = [{"role": "user", "content": prompt}]
        response = dashscope.Generation.call(
            model=self.model,
            messages=messages,
            result_format='message',
            temperature=0,
        )
        return response.output.choices[0].message.content
    
    def rewrite_context_dependent_query(self, current_query: str, conversation_history: str) -> str:
        """
        处理上下文依赖型查询
        
        参数:
            current_query: 当前用户问题
            conversation_history: 前序对话历史
        
        返回:
            改写后的完整问题
        """
        instruction = """
你是一个智能的查询优化助手。请分析用户的当前问题以及前序对话历史，判断当前问题是否依赖于上下文。
如果依赖，请将当前问题改写成一个独立的、包含所有必要上下文信息的完整问题。
如果不依赖，直接返回原问题。
只返回改写后的问题，不要包含其他说明文字。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前问题 ###
{current_query}

### 改写后的问题 ###
"""
        
        return self._call_llm(prompt)
    
    def rewrite_comparative_query(self, query: str, context_info: str) -> str:
        """
        处理对比型查询
        
        参数:
            query: 原始用户问题
            context_info: 对话历史或上下文信息
        
        返回:
            改写后的对比性查询
        """
        instruction = """
你是一个查询分析专家。请分析用户的输入和相关的对话上下文，识别出问题中需要进行比较的多个对象。
然后，将原始问题改写成一个更明确、更适合在知识库中检索的对比性查询。
只返回改写后的查询，不要包含其他说明文字。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史/上下文信息 ###
{context_info}

### 原始问题 ###
{query}

### 改写后的查询 ###
"""
        
        return self._call_llm(prompt)
    
    def rewrite_ambiguous_reference_query(self, current_query: str, conversation_history: str) -> str:
        """
        处理模糊指代型查询
        
        参数:
            current_query: 当前用户问题
            conversation_history: 前序对话历史
        
        返回:
            消除歧义后的清晰问题
        """
        instruction = """
你是一个消除语言歧义的专家。请分析用户的当前问题和对话历史，找出问题中 "都"、"它"、"这个" 等模糊指代词具体指向的对象。
然后，将这些指代词替换为明确的对象名称，生成一个清晰、无歧义的新问题。
只返回改写后的问题，不要包含其他说明文字。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前问题 ###
{current_query}

### 改写后的问题 ###
"""
        
        return self._call_llm(prompt)
    
    def rewrite_multi_intent_query(self, query: str) -> List[str]:
        """
        处理多意图型查询
        
        参数:
            query: 原始用户问题
        
        返回:
            分解后的问题列表
        """
        instruction = """
你是一个任务分解机器人。请将用户的复杂问题分解成多个独立的、可以单独回答的简单问题。
以 JSON 数组格式输出，例如：["问题 1", "问题 2", "问题 3"]
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 原始问题 ###
{query}

### 分解后的问题列表 ###
"""
        
        response = self._call_llm(prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return [response]
    
    def rewrite_rhetorical_query(self, current_query: str, conversation_history: str) -> str:
        """
        处理反问型查询
        
        参数:
            current_query: 当前用户问题
            conversation_history: 前序对话历史
        
        返回:
            中立、客观的可检索问题
        """
        instruction = """
你是一个沟通理解大师。请分析用户的反问或带有情绪的陈述，识别其背后真实的意图和问题。
然后，将这个反问改写成一个中立、客观、可以直接用于知识库检索的问题。
只返回改写后的问题，不要包含其他说明文字。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前问题 ###
{current_query}

### 改写后的问题 ###
"""
        
        return self._call_llm(prompt)
    
    def auto_rewrite_query(self, query: str, conversation_history: str = "", context_info: str = "") -> Dict[str, Any]:
        """
        自动识别 Query 类型并进行初步改写
        
        参数:
            query: 用户查询
            conversation_history: 对话历史
            context_info: 上下文信息
        
        返回:
            包含查询类型、改写结果和置信度的字典
        """
        instruction = """
你是一个智能的查询分析专家。请分析用户的查询，识别其属于以下哪种类型：
1. 上下文依赖型 - 包含"还有"、"其他"等需要上下文理解的词汇
2. 对比型 - 包含"哪个"、"比较"、"更"、"哪个更好"等比较词汇
3. 模糊指代型 - 包含"它"、"他们"、"都"、"这个"等指代词
4. 多意图型 - 包含多个独立问题，用"、"或"？"分隔
5. 反问型 - 包含"不会"、"难道"等反问语气
说明：如果同时存在多意图型、模糊指代型，优先级为多意图型>模糊指代型

请返回 JSON 格式的结果：
{
    "query_type": "查询类型",
    "rewritten_query": "改写后的查询",
    "confidence": 0.9
}
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 上下文信息 ###
{context_info}

### 原始查询 ###
{query}

### 分析结果 ###
"""
        
        response = self._call_llm(prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "query_type": "未知类型",
                "rewritten_query": query,
                "confidence": 0.5
            }
    
    def auto_rewrite_and_execute(self, query: str, conversation_history: str = "", context_info: str = "") -> Dict[str, Any]:
        """
        自动识别 Query 类型并调用相应的改写方法
        
        参数:
            query: 用户查询
            conversation_history: 对话历史
            context_info: 上下文信息
        
        返回:
            包含原始查询、检测类型、置信度和改写结果的字典
        """
        # 第一步：自动识别查询类型
        result = self.auto_rewrite_query(query, conversation_history, context_info)
        
        # 提取识别出的查询类型
        query_type = result.get('query_type', '')
        
        # 第二步：根据类型调用相应的改写方法
        if '上下文依赖' in query_type:
            final_result = self.rewrite_context_dependent_query(query, conversation_history)
        elif '对比' in query_type:
            final_result = self.rewrite_comparative_query(query, context_info or conversation_history)
        elif '模糊指代' in query_type:
            final_result = self.rewrite_ambiguous_reference_query(query, conversation_history)
        elif '多意图' in query_type:
            final_result = self.rewrite_multi_intent_query(query)
        elif '反问' in query_type:
            final_result = self.rewrite_rhetorical_query(query, conversation_history)
        else:
            final_result = result.get('rewritten_query', query)
        
        # 第三步：组装并返回完整结果
        return {
            "original_query": query,
            "detected_type": query_type,
            "confidence": result.get('confidence', 0.5),
            "rewritten_query": final_result,
            "auto_rewrite_result": result
        }


# =============================================================================
# 联网搜索判断模块
# =============================================================================

class WebSearchJudger:
    """
    联网搜索判断器
    
    功能：判断用户查询是否需要联网搜索来获取最新、最准确的信息
    
    支持的搜索场景（8 大类）：
        1. 时效性信息 - 最新、今天、现在、实时等
        2. 价格信息 - 多少钱、价格、费用、票价等
        3. 营业信息 - 营业时间、开放时间、是否开放等
        4. 活动信息 - 活动、表演、演出、节日等
        5. 天气信息 - 天气、下雨、温度等
        6. 交通信息 - 怎么去、交通、地铁、公交等
        7. 预订信息 - 预订、预约、购票、订票等
        8. 实时状态 - 排队、拥挤、人流量等
    """
    
    def __init__(self, model: str = Config.REWRITE_MODEL):
        """
        初始化联网搜索判断器
        
        参数:
            model: 使用的 LLM 模型名称
        """
        self.model = model
    
    def _call_llm(self, prompt: str) -> str:
        """
        调用大语言模型生成文本回复
        
        参数:
            prompt: 用户输入的提示词或问题
        
        返回:
            模型生成的文本内容
        """
        messages = [{"role": "user", "content": prompt}]
        response = dashscope.Generation.call(
            model=self.model,
            messages=messages,
            result_format='message',
            temperature=0,
        )
        return response.output.choices[0].message.content
    
    def identify_web_search_needs(self, query: str, conversation_history: str = "") -> Dict[str, Any]:
        """
        识别查询是否需要联网搜索
        
        参数:
            query: 用户查询
            conversation_history: 对话历史
        
        返回:
            包含是否需要搜索、原因和置信度的字典
        """
        instruction = """
你是一个智能的查询分析专家。请分析用户的查询，判断是否需要联网搜索来获取最新、最准确的信息。

需要联网搜索的情况包括：
1. 时效性信息 - 包含"最新"、"今天"、"现在"、"实时"、"当前"等时间相关词汇
2. 价格信息 - 包含"多少钱"、"价格"、"费用"、"票价"等价格相关词汇
3. 营业信息 - 包含"营业时间"、"开放时间"、"闭园时间"、"是否开放"等营业状态
4. 活动信息 - 包含"活动"、"表演"、"演出"、"节日"、"庆典"等动态信息
5. 天气信息 - 包含"天气"、"下雨"、"温度"等天气相关
6. 交通信息 - 包含"怎么去"、"交通"、"地铁"、"公交"等交通方式
7. 预订信息 - 包含"预订"、"预约"、"购票"、"订票"等预订相关
8. 实时状态 - 包含"排队"、"拥挤"、"人流量"等实时状态

请返回 JSON 格式：
{
    "need_web_search": true/false,
    "search_reason": "需要搜索的原因",
    "confidence": 0.9
}
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 用户查询 ###
{query}

### 分析结果 ###
"""
        
        response = self._call_llm(prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "need_web_search": False,
                "search_reason": "无法解析",
                "confidence": 0.5
            }
    
    def rewrite_for_web_search(self, query: str, search_type: str = "general") -> Dict[str, Any]:
        """
        将用户查询改写为更适合搜索引擎检索的形式
        
        参数:
            query: 原始用户查询
            search_type: 搜索类型
        
        返回:
            包含改写查询、关键词、意图和建议来源的字典
        """
        instruction = """
你是一个专业的搜索查询优化专家。请将用户的查询改写为更适合搜索引擎检索的形式。

改写技巧：
1. 添加具体地点 - 如"上海迪士尼乐园"、"香港迪士尼乐园"
2. 添加时间范围 - 如"2024 年"、"今天"、"本周"
3. 使用关键词组合 - 将长句拆分为关键词
4. 添加搜索意图 - 明确搜索目的
5. 去除口语化表达 - 转换为标准搜索词
6. 添加相关词汇 - 增加同义词或相关词

请返回 JSON 格式：
{
    "rewritten_query": "改写后的搜索查询",
    "search_keywords": ["关键词 1", "关键词 2", "关键词 3"],
    "search_intent": "搜索意图",
    "suggested_sources": ["建议搜索的网站类型"]
}
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 原始查询 ###
{query}

### 搜索类型 ###
{search_type}

### 改写结果 ###
"""
        
        response = self._call_llm(prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "rewritten_query": query,
                "search_keywords": [query],
                "search_intent": "信息查询",
                "suggested_sources": ["官方网站", "旅游网站"]
            }
    
    def auto_web_search_rewrite(self, query: str, conversation_history: str = "") -> Dict[str, Any]:
        """
        自动识别并改写为联网搜索查询
        
        参数:
            query: 用户查询
            conversation_history: 对话历史
        
        返回:
            完整的搜索结果字典
        """
        # 第一步：识别是否需要联网搜索
        search_analysis = self.identify_web_search_needs(query, conversation_history)
        
        if not search_analysis.get('need_web_search', False):
            return {
                "need_web_search": False,
                "reason": "查询不需要联网搜索",
                "original_query": query
            }
        
        # 第二步：改写查询
        rewritten_result = self.rewrite_for_web_search(query)
        
        # 第三步：组装并返回完整结果
        return {
            "need_web_search": True,
            "search_reason": search_analysis.get('search_reason', ''),
            "confidence": search_analysis.get('confidence', 0.5),
            "original_query": query,
            "rewritten_query": rewritten_result.get('rewritten_query', query),
            "search_keywords": rewritten_result.get('search_keywords', []),
            "search_intent": rewritten_result.get('search_intent', ''),
            "suggested_sources": rewritten_result.get('suggested_sources', [])
        }


# =============================================================================
# 文档处理模块
# =============================================================================

class DocumentProcessor:
    """
    文档处理器
    
    功能：处理 PDF 文档，包括文本提取、切块、向量化和向量库管理
    """
    
    def __init__(self, pdf_path: str, vector_db_path: str):
        """
        初始化文档处理器
        
        参数:
            pdf_path: PDF 文件路径
            vector_db_path: 向量数据库保存路径
        """
        self.pdf_path = pdf_path
        self.vector_db_path = vector_db_path
        self.vectorstore = None
        self.embeddings = None
    
    def extract_pages_text(self) -> List[Dict[str, Any]]:
        """
        逐页提取 PDF 文本，并保留页码信息
        
        返回:
            包含页码和文本的列表
        """
        print("[步骤 1] 提取 PDF 文本...")
        
        pdf_reader = PdfReader(self.pdf_path)
        pages = []
        
        for page_number, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text() or ""
            
            if not page_text.strip():
                print(f"  [提示] 第 {page_number} 页未提取到文本")
            
            pages.append({
                "page_number": page_number,
                "page_text": page_text,
            })
        
        print(f"  [完成] 成功提取 {len(pages)} 页文本")
        return pages
    
    def split_pages_to_chunks(self, pages: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        把每页文本切分成多个 chunk，并绑定来源页码 metadata
        
        参数:
            pages: 提取的页面文本列表
        
        返回:
            texts: 所有 chunk 文本列表
            metadatas: 对应的 metadata 列表
        """
        print("[步骤 2] 文本切块...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ".", " ", ""],
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )
        
        texts = []
        metadatas = []
        
        for page in pages:
            page_number = page["page_number"]
            page_text = page["page_text"]
            
            if not page_text.strip():
                continue
            
            page_chunks = text_splitter.split_text(page_text)
            
            for chunk_text in page_chunks:
                texts.append(chunk_text)
                metadatas.append({
                    "page": page_number,
                    "source": os.path.basename(self.pdf_path)
                })
        
        print(f"  [完成] 文本被分割成 {len(texts)} 个块")
        return texts, metadatas
    
    def build_or_load_faiss(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> FAISS:
        """
        构建或加载 FAISS 向量数据库
        
        参数:
            texts: chunk 文本列表
            metadatas: 对应的 metadata 列表
        
        返回:
            FAISS 向量数据库对象
        """
        print("[步骤 3] 构建/加载向量数据库...")
        
        # 初始化嵌入模型
        self.embeddings = DashScopeEmbeddings(
            model=Config.EMBEDDING_MODEL,
            dashscope_api_key=Config.DASHSCOPE_API_KEY,
        )
        
        # 检查是否已有保存的向量库
        if os.path.exists(self.vector_db_path) and os.path.isdir(self.vector_db_path) and os.listdir(self.vector_db_path):
            print(f"  [目录] 发现现有向量数据库：{self.vector_db_path}")
            self.vectorstore = FAISS.load_local(
                self.vector_db_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"  [完成] 已加载向量数据库")
            return self.vectorstore
        
        # 创建新的向量库
        print(f"  [构建] 开始构建向量数据库...")
        self.vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        
        # 保存到本地
        os.makedirs(self.vector_db_path, exist_ok=True)
        self.vectorstore.save_local(self.vector_db_path)
        print(f"  [完成] 向量数据库构建并保存完成：{self.vector_db_path}")
        
        return self.vectorstore
    
    def get_vectorstore(self) -> FAISS:
        """
        获取向量数据库对象
        
        返回:
            FAISS 向量数据库对象
        """
        return self.vectorstore


# =============================================================================
# 检索与问答模块
# =============================================================================

class RAGEngine:
    """
    RAG 引擎
    
    功能：结合 Query 改写和 MultiQuery 检索，执行完整的问答流程
    """
    
    def __init__(self, vectorstore: FAISS, embeddings: DashScopeEmbeddings):
        """
        初始化 RAG 引擎
        
        参数:
            vectorstore: FAISS 向量数据库
            embeddings: 嵌入模型
        """
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.llm = None
        self.query_rewriter = None
        self.web_search_judger = None
        self.document_chain = None
    
    def initialize(self) -> None:
        """
        初始化 LLM、Query 改写器、联网搜索判断器和问答链
        """
        print("[步骤 4] 初始化 LLM 和组件...")
        
        # 初始化 LLM
        self.llm = Tongyi(
            model_name=Config.LLM_MODEL,
            dashscope_api_key=Config.DASHSCOPE_API_KEY,
        )
        
        # 初始化 Query 改写器（多轮对话理解）
        self.query_rewriter = QueryRewriter(model=Config.REWRITE_MODEL)
        
        # 初始化联网搜索判断器
        self.web_search_judger = WebSearchJudger(model=Config.REWRITE_MODEL)
        
        # 创建问答链
        prompt = ChatPromptTemplate.from_template("""你是一个专业的文档问答助手。请根据以下检索到的上下文（Context）回答问题。

要求：
1. 如果上下文中没有答案，请直接说"未找到相关信息"，不要编造
2. 回答要准确、简洁，引用具体数据时请注明来源
3. 如果上下文中有多个相关信息，请综合整理后回答

<context>
{context}
</context>

问题：{input}

回答：""")
        
        self.document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        print(f"  [完成] LLM 模型：{Config.LLM_MODEL}")
        print(f"  [完成] 改写模型：{Config.REWRITE_MODEL}")
    
    def create_multi_query_retriever(self) -> MultiQueryRetriever:
        """
        创建 MultiQuery 检索器
        
        返回:
            MultiQueryRetriever 对象
        """
        base_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": Config.TOP_K}
        )
        
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm,
        )
        
        return retriever
    
    def process_local_query(self, query: str, conversation_history: str = "") -> Dict[str, Any]:
        """
        处理本地知识库查询的完整流程
        
        流程：
            1. Query 改写（理解真实意图）
            2. MultiQuery 检索（提高召回率）
            3. LLM 生成答案
            4. 收集来源页码
        
        参数:
            query: 用户查询
            conversation_history: 对话历史
        
        返回:
            包含答案、来源页码、改写信息等的字典
        """
        result = {
            "search_type": "local",
            "original_query": query,
            "rewritten_query": None,
            "query_type": None,
            "answer": None,
            "source_pages": [],
            "retrieved_docs": [],
            "cost": None,
        }
        
        # 第一步：Query 改写（理解真实意图）
        print("\n[阶段 1] Query 改写...")
        rewrite_result = self.query_rewriter.auto_rewrite_and_execute(
            query=query,
            conversation_history=conversation_history
        )
        
        result["query_type"] = rewrite_result["detected_type"]
        
        # 获取改写后的查询
        rewritten_query = rewrite_result["rewritten_query"]
        if isinstance(rewritten_query, list):
            # 多意图型：使用第一个问题作为主查询
            result["rewritten_query"] = rewritten_query[0]
            print(f"  [类型] 多意图型（已分解为 {len(rewritten_query)} 个子问题）")
        else:
            result["rewritten_query"] = rewritten_query
            print(f"  [类型] {rewrite_result['detected_type']}")
        
        print(f"  [改写] {result['rewritten_query']}")
        
        # 第二步：MultiQuery 检索（提高召回率）
        print("\n[阶段 2] MultiQuery 检索...")
        retriever = self.create_multi_query_retriever()
        docs = retriever.invoke(result["rewritten_query"])
        
        result["retrieved_docs"] = docs
        print(f"  [完成] 检索到 {len(docs)} 个相关文档块")
        
        # 第三步：LLM 生成答案
        print("\n[阶段 3] 生成答案...")
        with get_openai_callback() as cost:
            response_text = self.document_chain.invoke({
                "context": docs,
                "input": result["rewritten_query"]
            })
        
        result["answer"] = response_text
        result["cost"] = cost
        print(f"  [完成] 答案已生成")
        print(f"  [成本] 总费用：${cost.total_cost:.4f}")
        
        # 第四步：收集来源页码
        source_pages = set()
        for doc in docs:
            page = (doc.metadata or {}).get("page")
            if page is not None:
                source_pages.add(page)
        
        result["source_pages"] = sorted(list(source_pages))
        
        return result
    
    def process_web_query(self, query: str, conversation_history: str = "") -> Dict[str, Any]:
        """
        处理联网搜索查询的完整流程
        
        流程：
            1. 判断是否需要联网搜索
            2. 改写为适合搜索引擎的查询
            3. 生成搜索策略
            4. 返回搜索建议（实际搜索需调用搜索引擎 API）
        
        参数:
            query: 用户查询
            conversation_history: 对话历史
        
        返回:
            包含搜索建议、关键词、策略等的字典
        """
        result = {
            "search_type": "web",
            "original_query": query,
            "rewritten_query": None,
            "search_keywords": [],
            "search_intent": None,
            "suggested_sources": [],
            "answer": None,
            "note": "需要调用搜索引擎 API 获取实际结果",
        }
        
        # 第一步：判断并改写联网搜索查询
        print("\n[阶段 1] 联网搜索判断...")
        web_search_result = self.web_search_judger.auto_web_search_rewrite(
            query=query,
            conversation_history=conversation_history
        )
        
        result["rewritten_query"] = web_search_result.get("rewritten_query", query)
        result["search_keywords"] = web_search_result.get("search_keywords", [])
        result["search_intent"] = web_search_result.get("search_intent", "")
        result["suggested_sources"] = web_search_result.get("suggested_sources", [])
        
        print(f"  [搜索原因] {web_search_result.get('search_reason', '')}")
        print(f"  [改写查询] {result['rewritten_query']}")
        print(f"  [关键词] {result['search_keywords']}")
        print(f"  [建议来源] {result['suggested_sources']}")
        
        # 注意：实际项目中需要在这里调用搜索引擎 API
        # 例如：Google Search API、Bing Search API、SerpAPI 等
        result["answer"] = "[提示] 此问题需要联网搜索获取最新信息，建议访问：" + ", ".join(result["suggested_sources"])
        
        return result
    
    def process_query(self, query: str, conversation_history: str = "") -> Dict[str, Any]:
        """
        处理用户查询的完整流程（自动路由）
        
        流程：
            1. 判断是否需要联网搜索
            2. 如果需要联网 → 调用 process_web_query
            3. 如果不需要联网 → 调用 process_local_query
        
        参数:
            query: 用户查询
            conversation_history: 对话历史
        
        返回:
            包含答案、来源、成本等的完整结果字典
        """
        # 第一步：判断是否需要联网搜索
        print("\n[路由决策] 判断查询类型...")
        web_search_analysis = self.web_search_judger.identify_web_search_needs(
            query=query,
            conversation_history=conversation_history
        )
        
        need_web_search = web_search_analysis.get('need_web_search', False)
        
        if need_web_search:
            print(f"  [决策] 需要联网搜索")
            print(f"  [原因] {web_search_analysis.get('search_reason', '')}")
            result = self.process_web_query(query, conversation_history)
        else:
            print(f"  [决策] 使用本地知识库")
            result = self.process_local_query(query, conversation_history)
        
        return result


# =============================================================================
# 主系统类
# =============================================================================

class HybridRAGSystem:
    """
    混合检索 RAG 系统
    
    功能：整合所有模块，提供完整的对话式问答体验
    支持：本地知识库检索 + 互联网搜索智能路由
    """
    
    def __init__(self):
        """
        初始化系统
        """
        self.doc_processor = None
        self.rag_engine = None
        self.conversation_manager = None
        self.is_initialized = False
    
    def initialize(self) -> None:
        """
        初始化系统（文档处理、向量库、LLM 等）
        """
        print("\n" + "=" * 60)
        print("混合检索 RAG 系统 - 初始化")
        print("=" * 60)
        
        # 初始化文档处理器
        self.doc_processor = DocumentProcessor(
            pdf_path=Config.PDF_PATH,
            vector_db_path=Config.VECTOR_DB_PATH
        )
        
        # 提取文本并切块
        pages = self.doc_processor.extract_pages_text()
        texts, metadatas = self.doc_processor.split_pages_to_chunks(pages)
        
        # 构建或加载向量库
        vectorstore = self.doc_processor.build_or_load_faiss(texts, metadatas)
        
        # 初始化 RAG 引擎
        self.rag_engine = RAGEngine(
            vectorstore=vectorstore,
            embeddings=self.doc_processor.embeddings
        )
        self.rag_engine.initialize()
        
        # 初始化对话管理器
        self.conversation_manager = ConversationManager()
        
        self.is_initialized = True
        
        print("\n" + "=" * 60)
        print("系统初始化完成")
        print("=" * 60)
    
    def chat(self, query: str) -> Dict[str, Any]:
        """
        处理用户对话
        
        参数:
            query: 用户输入
        
        返回:
            包含答案和元数据的字典
        """
        if not self.is_initialized:
            raise RuntimeError("系统未初始化，请先调用 initialize() 方法")
        
        # 获取对话历史
        conversation_history = self.conversation_manager.get_formatted_history()
        
        # 处理查询（自动路由到本地或联网搜索）
        result = self.rag_engine.process_query(query, conversation_history)
        
        # 添加到对话历史
        self.conversation_manager.add_turn("user", query)
        self.conversation_manager.add_turn("assistant", result["answer"])
        
        return result
    
    def print_result(self, result: Dict[str, Any]) -> None:
        """
        打印问答结果
        
        参数:
            result: process_query 返回的结果字典
        """
        print("\n" + "=" * 60)
        print("问答结果")
        print("=" * 60)
        
        print(f"\n[搜索类型]")
        print(f"    {'互联网搜索' if result['search_type'] == 'web' else '本地知识库'}")
        
        print(f"\n[原始问题]")
        print(f"    {result['original_query']}")
        
        if result.get('rewritten_query'):
            print(f"\n[改写后的查询]")
            print(f"    {result['rewritten_query']}")
        
        if result.get('query_type'):
            print(f"\n[查询类型]")
            print(f"    {result['query_type']}")
        
        print(f"\n[答案]")
        print("-" * 60)
        print(result["answer"])
        print("-" * 60)
        
        if result["search_type"] == "local":
            if result.get("source_pages"):
                print(f"\n[来源页码] (共 {len(result['source_pages'])} 页)")
                for page in result["source_pages"]:
                    print(f"    - 第 {page} 页")
            
            print(f"\n[检索文档数]")
            print(f"    {len(result.get('retrieved_docs', []))} 个")
        
        if result.get("search_keywords"):
            print(f"\n[搜索关键词]")
            for kw in result["search_keywords"]:
                print(f"    - {kw}")
        
        if result.get("suggested_sources"):
            print(f"\n[建议来源]")
            for src in result["suggested_sources"]:
                print(f"    - {src}")
        
        if result.get("cost"):
            print(f"\n[API 成本]")
            print(f"    总费用：${result['cost'].total_cost:.4f}")
            print(f"    总 Token: {result['cost'].total_tokens}")
        
        print("=" * 60)
    
    def clear_conversation(self) -> None:
        """
        清空对话历史
        """
        self.conversation_manager.clear()
        print("[提示] 对话历史已清空")
    
    def get_conversation_history(self) -> str:
        """
        获取当前对话历史
        
        返回:
            格式化的对话历史字符串
        """
        return self.conversation_manager.get_formatted_history()


# =============================================================================
# 主程序入口
# =============================================================================

def main():
    """
    主程序：演示混合检索 RAG 系统的完整功能
    """
    print("\n")
    print("*" * 60)
    print("*  混合检索 RAG 系统")
    print("*  本地知识库 + 互联网搜索智能路由")
    print("*" * 60)
    
    # 初始化系统
    system = HybridRAGSystem()
    system.initialize()
    
    # 预设测试问题（覆盖本地知识库和联网搜索场景）
    test_queries = [
        # 本地知识库可回答的问题
        "客户经理的考核标准是什么？",
        "被投诉了，投诉一次扣多少分？",
        "还有其他扣分项吗？",  # 上下文依赖型
        
        # 需要联网搜索的问题
        "今天的天气怎么样？",  # 天气信息
        "下周六的门票多少钱？",  # 价格信息
    ]
    
    print("\n" + "=" * 60)
    print("开始混合检索测试")
    print("=" * 60)
    
    # 逐个处理查询
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'#' * 60}")
        print(f"#  第 {i} 轮对话")
        print(f"#" * 60)
        
        # 处理查询
        result = system.chat(query)
        
        # 打印结果
        system.print_result(result)
    
    # 显示完整对话历史
    print("\n\n" + "=" * 60)
    print("完整对话历史")
    print("=" * 60)
    print(system.get_conversation_history())
    
    print("\n\n")
    print("*" * 60)
    print("*  所有测试完成")
    print("*" * 60)
    print("\n")


if __name__ == "__main__":
    main()
