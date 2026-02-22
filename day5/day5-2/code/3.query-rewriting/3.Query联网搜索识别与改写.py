"""
## 联网搜索 Query 识别与改写系统

判断用户查询是否需要联网搜索，并生成优化的搜索策略

1. 识别查询是否需要联网搜索（8 大场景判断）
2. 将查询改写为适合搜索引擎的形式
3. 生成详细的搜索策略（关键词、平台、时间范围）

### 适用场景：
- 混合检索系统（本地知识库 + 互联网搜索）
- 需要时效性信息的问答系统
- 智能路由（离线/在线检索决策）

"""

# -----------------------------------------------------------------------------
# 导入依赖库
# -----------------------------------------------------------------------------
import dashscope  # 阿里云 DashScope SDK，用于调用通义千问等大模型
import os         # 操作系统接口模块，用于读取环境变量
import json       # JSON 数据处理模块，用于解析和生成 JSON 格式数据
import re         # 正则表达式模块，用于文本匹配和处理
from datetime import datetime  # 日期时间模块，用于获取当前日期
from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------------------------------------
# 配置 API Key
# -----------------------------------------------------------------------------
# 从环境变量中获取 DashScope API Key
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

# 检查 API Key 是否设置成功
if not dashscope.api_key:
    raise ValueError("请先设置环境变量 DASHSCOPE_API_KEY")


# -----------------------------------------------------------------------------
# 核心工具函数
# -----------------------------------------------------------------------------

def get_completion(prompt, model="qwen-turbo-latest"):
    """
    调用大语言模型生成文本回复
    
    参数:
        prompt: 用户输入的提示词或问题
        model: 使用的模型名称，默认为 qwen-turbo-latest
    
    返回:
        模型生成的文本内容
    """
    # 构造消息列表，符合 DashScope API 的消息格式
    messages = [{"role": "user", "content": prompt}]
    
    # 调用 DashScope Generation API 生成回复
    response = dashscope.Generation.call(
        model=model,              # 指定使用的模型
        messages=messages,        # 传入对话消息
        result_format='message',  # 返回格式为 message 类型
        temperature=0,            # 温度设为 0，保证输出确定性，适合任务型场景
    )
    
    # 提取并返回生成的文本内容
    return response.output.choices[0].message.content


# -----------------------------------------------------------------------------
# 联网搜索 Query 改写器类
# -----------------------------------------------------------------------------

class WebSearchQueryRewriter:
    """
    联网搜索 Query 改写器
    
    功能：判断用户查询是否需要联网搜索，并在需要时生成优化的搜索策略
    
    支持的搜索场景：
        - 时效性信息：最新、今天、现在、实时等
        - 价格信息：多少钱、价格、费用、票价等
        - 营业信息：营业时间、开放时间、是否开放等
        - 活动信息：活动、表演、演出、节日等
        - 天气信息：天气、下雨、温度等
        - 交通信息：怎么去、交通、地铁、公交等
        - 预订信息：预订、预约、购票、订票等
        - 实时状态：排队、拥挤、人流量等
    """
    
    def __init__(self, model="qwen-turbo-latest"):
        """
        初始化联网搜索 Query 改写器
        
        参数:
            model: 使用的 LLM 模型名称，默认为 qwen-turbo-latest
        """
        self.model = model  # 保存模型配置，供后续方法使用
    
    # -------------------------------------------------------------------------
    # 识别是否需要联网搜索
    # -------------------------------------------------------------------------
    
    def identify_web_search_needs(self, query, conversation_history=""):
        """
        识别查询是否需要联网搜索
        
        判断依据（8 大场景）：
            1. 时效性信息 - 包含"最新"、"今天"、"现在"等时间相关词汇
            2. 价格信息 - 包含"多少钱"、"价格"、"费用"等价格相关词汇
            3. 营业信息 - 包含"营业时间"、"开放时间"等营业状态
            4. 活动信息 - 包含"活动"、"表演"、"演出"等动态信息
            5. 天气信息 - 包含"天气"、"下雨"、"温度"等天气相关
            6. 交通信息 - 包含"怎么去"、"交通"、"地铁"等交通方式
            7. 预订信息 - 包含"预订"、"预约"、"购票"等预订相关
            8. 实时状态 - 包含"排队"、"拥挤"、"人流量"等实时状态
        
        参数:
            query: 用户查询
            conversation_history: 对话历史（可选）
        
        返回:
            包含是否需要搜索、原因和置信度的字典
            {
                "need_web_search": true/false,
                "search_reason": "需要搜索的原因",
                "confidence": 0.9
            }
        """
        # 定义系统指令，指导 LLM 如何判断是否需要联网搜索
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
        
        # 构造完整的提示词，包含指令、对话历史和用户查询
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 用户查询 ###
{query}

### 分析结果 ###
"""
        
        # 调用 LLM 生成分析结果
        response = get_completion(prompt, self.model)
        
        # 尝试解析 JSON 格式的输出
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认结果
            return {
                "need_web_search": False,
                "search_reason": "无法解析",
                "confidence": 0.5
            }
    
    # -------------------------------------------------------------------------
    # 为联网搜索改写查询
    # -------------------------------------------------------------------------
    
    def rewrite_for_web_search(self, query, search_type="general"):
        """
        将用户查询改写为更适合搜索引擎检索的形式
        
        改写技巧：
            1. 添加具体地点 - 如"上海迪士尼乐园"、"香港迪士尼乐园"
            2. 添加时间范围 - 如"2026 年"、"今天"、"本周"
            3. 使用关键词组合 - 将长句拆分为关键词
            4. 添加搜索意图 - 明确搜索目的
            5. 去除口语化表达 - 转换为标准搜索词
            6. 添加相关词汇 - 增加同义词或相关词
        
        参数:
            query: 原始用户查询
            search_type: 搜索类型，默认为 general
        
        返回:
            包含改写查询、关键词、意图和建议来源的字典
            {
                "rewritten_query": "改写后的搜索查询",
                "search_keywords": ["关键词 1", "关键词 2"],
                "search_intent": "搜索意图",
                "suggested_sources": ["建议搜索的网站类型"]
            }
        """
        # 定义系统指令，指导 LLM 如何改写搜索查询
        instruction = """
你是一个专业的搜索查询优化专家。请将用户的查询改写为更适合搜索引擎检索的形式。

改写技巧：
1. 添加具体地点 - 如"上海迪士尼乐园"、"香港迪士尼乐园"
2. 添加时间范围 - 如"2026 年"、"今天"、"本周"
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
        
        # 构造完整的提示词
        prompt = f"""
### 指令 ###
{instruction}

### 原始查询 ###
{query}

### 搜索类型 ###
{search_type}

### 改写结果 ###
"""
        
        # 调用 LLM 生成改写结果
        response = get_completion(prompt, self.model)
        
        # 尝试解析 JSON 格式的输出
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认结果
            return {
                "rewritten_query": query,
                "search_keywords": [query],
                "search_intent": "信息查询",
                "suggested_sources": ["官方网站", "旅游网站"]
            }
    
    # -------------------------------------------------------------------------
    # 生成搜索策略
    # -------------------------------------------------------------------------
    
    def generate_search_strategy(self, query, search_type="general"):
        """
        为用户查询制定详细的搜索策略
        
        搜索策略包括：
            1. 主要搜索词 - 核心关键词
            2. 扩展搜索词 - 相关词汇和同义词
            3. 搜索网站 - 推荐的搜索平台
            4. 时间范围 - 具体的搜索时间范围
        
        参数:
            query: 用户查询
            search_type: 搜索类型，默认为 general
        
        返回:
            包含主要关键词、扩展关键词、搜索平台和时间范围的字典
            {
                "primary_keywords": ["主要关键词"],
                "extended_keywords": ["扩展关键词"],
                "search_platforms": ["搜索平台"],
                "time_range": "具体的时间范围"
            }
        """
        # 获取当前日期，用于生成时间相关的搜索策略
        current_date = datetime.now().strftime("%Y年%m月%d日")
        
        # 定义系统指令，指导 LLM 如何生成搜索策略
        instruction = f"""
你是一个搜索策略专家。请为用户的查询制定详细的搜索策略。

当前日期：{current_date}

搜索策略包括：
1. 主要搜索词 - 核心关键词
2. 扩展搜索词 - 相关词汇和同义词
3. 搜索网站 - 推荐的搜索平台
4. 时间范围 - 具体的搜索时间范围

请返回 JSON 格式：
{{
    "primary_keywords": ["主要关键词"],
    "extended_keywords": ["扩展关键词"],
    "search_platforms": ["搜索平台"],
    "time_range": "具体的时间范围"
}}
"""
        
        # 构造完整的提示词
        prompt = f"""
### 指令 ###
{instruction}

### 用户查询 ###
{query}

### 搜索类型 ###
{search_type}

### 搜索策略 ###
"""
        
        # 调用 LLM 生成搜索策略
        response = get_completion(prompt, self.model)
        
        # 尝试解析 JSON 格式的输出
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认结果
            return {
                "primary_keywords": [query],
                "extended_keywords": [],
                "search_platforms": ["百度", "谷歌"],
                "time_range": "最近一周"
            }
    
    # -------------------------------------------------------------------------
    # 自动识别并改写
    # -------------------------------------------------------------------------
    
    def auto_web_search_rewrite(self, query, conversation_history=""):
        """
        自动识别查询是否需要联网搜索，并生成完整的搜索方案
        
        工作流程：
            1. 调用 identify_web_search_needs 判断是否需要联网搜索
            2. 如果需要，调用 rewrite_for_web_search 改写查询
            3. 调用 generate_search_strategy 生成搜索策略
            4. 返回完整的搜索结果
        
        参数:
            query: 用户查询
            conversation_history: 对话历史（可选）
        
        返回:
            完整的搜索结果字典
            如果需要搜索：
            {
                "need_web_search": True,
                "search_reason": "需要搜索的原因",
                "confidence": 0.9,
                "original_query": "原始查询",
                "rewritten_query": "改写后的查询",
                "search_keywords": ["关键词"],
                "search_intent": "搜索意图",
                "suggested_sources": ["建议来源"],
                "search_strategy": {搜索策略}
            }
            
            如果不需要搜索：
            {
                "need_web_search": False,
                "reason": "不需要搜索的原因",
                "original_query": "原始查询"
            }
        """
        # 第一步：识别是否需要联网搜索
        search_analysis = self.identify_web_search_needs(query, conversation_history)
        
        # 如果不需要联网搜索，直接返回
        if not search_analysis.get('need_web_search', False):
            return {
                "need_web_search": False,
                "reason": "查询不需要联网搜索",
                "original_query": query
            }
        
        # 第二步：改写查询为适合搜索引擎的形式
        rewritten_result = self.rewrite_for_web_search(query)
        
        # 第三步：生成详细的搜索策略
        search_strategy = self.generate_search_strategy(query)
        
        # 第四步：组装并返回完整结果
        return {
            "need_web_search": True,
            "search_reason": search_analysis.get('search_reason', ''),
            "confidence": search_analysis.get('confidence', 0.5),
            "original_query": query,
            "rewritten_query": rewritten_result.get('rewritten_query', query),
            "search_keywords": rewritten_result.get('search_keywords', []),
            "search_intent": rewritten_result.get('search_intent', ''),
            "suggested_sources": rewritten_result.get('suggested_sources', []),
            "search_strategy": search_strategy
        }


# -----------------------------------------------------------------------------
# 主程序入口
# -----------------------------------------------------------------------------

def main():
    """
    主函数：演示联网搜索 Query 改写器的各种使用场景
    """
    # 初始化联网搜索 Query 改写器
    web_searcher = WebSearchQueryRewriter()
    
    # 打印标题
    print("=" * 60)
    print("Query 联网搜索识别与改写示例（迪士尼主题乐园）")
    print("=" * 60)
    print()
    
    # -------------------------------------------------------------------------
    # 示例 1: 时效性信息查询
    # -------------------------------------------------------------------------
    print("示例 1: 时效性信息查询")
    print("-" * 60)
    
    # 构造对话历史，模拟真实的多轮对话场景
    conversation_history1 = """
用户："我想去上海迪士尼乐园玩"
AI："上海迪士尼乐园是一个很棒的选择！"
"""
    # 当前查询包含"今天"和"现在"，需要联网获取实时信息
    query1 = "上海迪士尼乐园今天开放吗？现在人多不多？"
    
    print(f"对话历史：{conversation_history1}")
    print(f"当前查询：{query1}")
    print()
    
    # 执行自动识别和改写
    result1 = web_searcher.auto_web_search_rewrite(query1, conversation_history1)
    
    # 根据结果打印相应信息
    if result1['need_web_search']:
        print("[结果] 需要联网搜索")
        print(f"  搜索原因：{result1['search_reason']}")
        print(f"  置信度：{result1['confidence']}")
        print(f"  改写查询：{result1['rewritten_query']}")
        print(f"  搜索关键词：{result1['search_keywords']}")
        print(f"  搜索意图：{result1['search_intent']}")
        print(f"  建议来源：{result1['suggested_sources']}")
        print("  搜索策略:")
        print(f"    - 主要关键词：{result1['search_strategy']['primary_keywords']}")
        print(f"    - 扩展关键词：{result1['search_strategy']['extended_keywords']}")
        print(f"    - 搜索平台：{result1['search_strategy']['search_platforms']}")
        print(f"    - 时间范围：{result1['search_strategy']['time_range']}")
    else:
        print("[结果] 不需要联网搜索")
        print(f"  原因：{result1['reason']}")
    
    print("\n" + "=" * 60 + "\n")
    
    # -------------------------------------------------------------------------
    # 示例 2: 价格和预订信息查询
    # -------------------------------------------------------------------------
    print("示例 2: 价格和预订信息查询")
    print("-" * 60)
    
    # 当前查询包含价格和预订相关信息，需要联网获取最新价格
    query2 = "下周六的门票多少钱？需要提前多久预订？"
    
    print(f"当前查询：{query2}")
    print()
    
    # 执行自动识别和改写
    result2 = web_searcher.auto_web_search_rewrite(query2)
    
    # 根据结果打印相应信息
    if result2['need_web_search']:
        print("[结果] 需要联网搜索")
        print(f"  搜索原因：{result2['search_reason']}")
        print(f"  置信度：{result2['confidence']}")
        print(f"  改写查询：{result2['rewritten_query']}")
        print(f"  搜索关键词：{result2['search_keywords']}")
        print(f"  搜索意图：{result2['search_intent']}")
        print(f"  建议来源：{result2['suggested_sources']}")
        print("  搜索策略:")
        print(f"    - 主要关键词：{result2['search_strategy']['primary_keywords']}")
        print(f"    - 扩展关键词：{result2['search_strategy']['extended_keywords']}")
        print(f"    - 搜索平台：{result2['search_strategy']['search_platforms']}")
        print(f"    - 时间范围：{result2['search_strategy']['time_range']}")
    else:
        print("[结果] 不需要联网搜索")
        print(f"  原因：{result2['reason']}")
    
    # 打印完成提示
    print("\n" + "=" * 60)
    print("所有示例执行完成")
    print("=" * 60)


# -----------------------------------------------------------------------------
# 程序入口
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
