"""
## 智能 Query 改写系统 - 处理多轮对话中的复杂查询

根据类型调用对应改写方法
• 上下文依赖 → rewrite_context_dependent_query()
• 对比型 → rewrite_comparative_query()
• 模糊指代 → rewrite_ambiguous_reference_query()
• 多意图 → rewrite_multi_intent_query()
• 反问型 → rewrite_rhetorical_query()
"""

# -----------------------------------------------------------------------------
# 导入依赖库
# -----------------------------------------------------------------------------
import dashscope  # 阿里云 DashScope SDK，用于调用通义千问等大模型
import os         # 操作系统接口模块，用于读取环境变量
import json       # JSON 数据处理模块，用于解析和生成 JSON 格式数据

# -----------------------------------------------------------------------------
# 配置 API Key
# -----------------------------------------------------------------------------
# 从环境变量中获取 DashScope API Key
# 建议将敏感信息放在环境变量中，避免硬编码在代码里
from dotenv import load_dotenv
load_dotenv()
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
# Query 改写器类
# -----------------------------------------------------------------------------

class QueryRewriter:
    """
    智能 Query 改写器
    
    功能：将多轮对话中的复杂查询改写为清晰、独立、可检索的标准查询
    支持的改写类型：
        - 上下文依赖型：补全省略的上下文信息
        - 对比型：明确比较对象和维度
        - 模糊指代型：消除代词歧义
        - 多意图型：分解为多个独立问题
        - 反问型：提取真实意图
    """
    
    def __init__(self, model="qwen-turbo-latest"):
        """
        初始化 Query 改写器
        
        参数:
            model: 使用的 LLM 模型名称
        """
        self.model = model  # 保存模型配置，供后续方法使用
    
    # -------------------------------------------------------------------------
    # 上下文依赖型 Query 改写
    # -------------------------------------------------------------------------
    
    def rewrite_context_dependent_query(self, current_query, conversation_history):
        """
        处理上下文依赖型查询
        
        场景：用户问题中包含"还有"、"其他"等词汇，需要参考前文才能理解
        
        参数:
            current_query: 当前用户问题
            conversation_history: 前序对话历史
        
        返回:
            改写后的完整问题
        """
        # 定义系统指令，指导 LLM 如何进行改写
        instruction = """
你是一个智能的查询优化助手。请分析用户的当前问题以及前序对话历史，判断当前问题是否依赖于上下文。
如果依赖，请将当前问题改写成一个独立的、包含所有必要上下文信息的完整问题。
如果不依赖，直接返回原问题。
"""
        
        # 构造完整的提示词，包含指令、对话历史和当前问题
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前问题 ###
{current_query}

### 改写后的问题 ###
"""
        
        # 调用 LLM 生成改写结果
        return get_completion(prompt, self.model)
    
    # -------------------------------------------------------------------------
    # 对比型 Query 改写
    # -------------------------------------------------------------------------
    
    def rewrite_comparative_query(self, query, context_info):
        """
        处理对比型查询
        
        场景：用户问题中包含"哪个"、"比较"、"更"等比较词汇
        
        参数:
            query: 原始用户问题
            context_info: 对话历史或上下文信息，用于识别比较对象
        
        返回:
            改写后的对比性查询
        """
        # 定义系统指令，指导 LLM 识别比较对象并改写
        instruction = """
你是一个查询分析专家。请分析用户的输入和相关的对话上下文，识别出问题中需要进行比较的多个对象。
然后，将原始问题改写成一个更明确、更适合在知识库中检索的对比性查询。
"""
        
        # 构造完整的提示词
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史/上下文信息 ###
{context_info}

### 原始问题 ###
{query}

### 改写后的查询 ###
"""
        
        # 调用 LLM 生成改写结果
        return get_completion(prompt, self.model)
    
    # -------------------------------------------------------------------------
    # 模糊指代型 Query 改写
    # -------------------------------------------------------------------------
    
    def rewrite_ambiguous_reference_query(self, current_query, conversation_history):
        """
        处理模糊指代型查询
        
        场景：用户问题中包含"它"、"他们"、"都"、"这个"等指代词
        
        参数:
            current_query: 当前用户问题
            conversation_history: 前序对话历史，用于确定指代对象
        
        返回:
            消除歧义后的清晰问题
        """
        # 定义系统指令，指导 LLM 消除指代歧义
        instruction = """
你是一个消除语言歧义的专家。请分析用户的当前问题和对话历史，找出问题中 "都"、"它"、"这个" 等模糊指代词具体指向的对象。
然后，将这些指代词替换为明确的对象名称，生成一个清晰、无歧义的新问题。
"""
        
        # 构造完整的提示词
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前问题 ###
{current_query}

### 改写后的问题 ###
"""
        
        # 调用 LLM 生成改写结果
        return get_completion(prompt, self.model)
    
    # -------------------------------------------------------------------------
    # 多意图型 Query 改写
    # -------------------------------------------------------------------------
    
    def rewrite_multi_intent_query(self, query):
        """
        处理多意图型查询
        
        场景：一个问题中包含多个独立子问题，需要分解后分别处理
        
        参数:
            query: 原始用户问题
        
        返回:
            分解后的问题列表（JSON 数组格式）
        """
        # 定义系统指令，指导 LLM 分解复杂问题
        instruction = """
你是一个任务分解机器人。请将用户的复杂问题分解成多个独立的、可以单独回答的简单问题。以 JSON 数组格式输出。
"""
        
        # 构造完整的提示词
        prompt = f"""
### 指令 ###
{instruction}

### 原始问题 ###
{query}

### 分解后的问题列表 ###
请以 JSON 数组格式输出，例如：["问题 1", "问题 2", "问题 3"]
"""
        
        # 调用 LLM 生成分解结果
        response = get_completion(prompt, self.model)
        
        # 尝试解析 JSON 格式的输出
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果解析失败，将原始响应包装为列表返回
            return [response]
    
    # -------------------------------------------------------------------------
    # 反问型 Query 改写
    # -------------------------------------------------------------------------
    
    def rewrite_rhetorical_query(self, current_query, conversation_history):
        """
        处理反问型查询
        
        场景：用户问题带有情绪或反问语气，需要提取真实意图
        
        参数:
            current_query: 当前用户问题
            conversation_history: 前序对话历史，用于理解情绪背景
        
        返回:
            中立、客观的可检索问题
        """
        # 定义系统指令，指导 LLM 理解反问背后的真实意图
        instruction = """
你是一个沟通理解大师。请分析用户的反问或带有情绪的陈述，识别其背后真实的意图和问题。
然后，将这个反问改写成一个中立、客观、可以直接用于知识库检索的问题。
"""
        
        # 构造完整的提示词
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前问题 ###
{current_query}

### 改写后的问题 ###
"""
        
        # 调用 LLM 生成改写结果
        return get_completion(prompt, self.model)
    
    # -------------------------------------------------------------------------
    # 自动识别 Query 类型
    # -------------------------------------------------------------------------
    
    def auto_rewrite_query(self, query, conversation_history="", context_info=""):
        """
        自动识别 Query 类型并进行初步改写
        
        支持的类型：
            1. 上下文依赖型 - 包含"还有"、"其他"等词汇
            2. 对比型 - 包含"哪个"、"比较"、"更"等词汇
            3. 模糊指代型 - 包含"它"、"他们"、"都"、"这个"等指代词
            4. 多意图型 - 包含多个独立问题
            5. 反问型 - 包含"不会"、"难道"等反问语气
        
        参数:
            query: 用户查询
            conversation_history: 对话历史（可选）
            context_info: 上下文信息（可选）
        
        返回:
            包含查询类型、改写结果和置信度的字典
        """
        # 定义系统指令，指导 LLM 识别查询类型
        instruction = """
你是一个智能的查询分析专家。请分析用户的查询，识别其属于以下哪种类型：
1. 上下文依赖型 - 包含"还有"、"其他"等需要上下文理解的词汇
2. 对比型 - 包含"哪个"、"比较"、"更"、"哪个更好"、"哪个更"等比较词汇
3. 模糊指代型 - 包含"它"、"他们"、"都"、"这个"等指代词
4. 多意图型 - 包含多个独立问题，用"、"或"？"分隔
5. 反问型 - 包含"不会"、"难道"等反问语气
说明：如果同时存在多意图型、模糊指代型，优先级为多意图型>模糊指代型

请返回 JSON 格式的结果：
{
    "query_type": "查询类型",
    "rewritten_query": "改写后的查询",
    "confidence": "置信度 (0-1)"
}
"""
        
        # 构造完整的提示词
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
        
        # 调用 LLM 生成分析结果
        response = get_completion(prompt, self.model)
        
        # 尝试解析 JSON 格式的输出
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认结果
            return {
                "query_type": "未知类型",
                "rewritten_query": query,
                "confidence": 0.5
            }
    
    # -------------------------------------------------------------------------
    # 自动识别并执行改写
    # -------------------------------------------------------------------------
    
    def auto_rewrite_and_execute(self, query, conversation_history="", context_info=""):
        """
        自动识别 Query 类型并调用相应的改写方法
        
        工作流程：
            1. 调用 auto_rewrite_query 识别查询类型
            2. 根据识别结果调用对应的专用改写方法
            3. 返回完整的改写结果
        
        参数:
            query: 用户查询
            conversation_history: 对话历史（可选）
            context_info: 上下文信息（可选）
        
        返回:
            包含原始查询、检测类型、置信度和改写结果的字典
        """
        # 第一步：自动识别查询类型
        result = self.auto_rewrite_query(query, conversation_history, context_info)
        
        # 提取识别出的查询类型
        query_type = result.get('query_type', '')
        
        # 第二步：根据类型调用相应的改写方法
        if '上下文依赖' in query_type:
            # 上下文依赖型：补全上下文信息
            final_result = self.rewrite_context_dependent_query(query, conversation_history)
        elif '对比' in query_type:
            # 对比型：明确比较对象
            final_result = self.rewrite_comparative_query(query, context_info or conversation_history)
        elif '模糊指代' in query_type:
            # 模糊指代型：消除指代歧义
            final_result = self.rewrite_ambiguous_reference_query(query, conversation_history)
        elif '多意图' in query_type:
            # 多意图型：分解为多个问题
            final_result = self.rewrite_multi_intent_query(query)
        elif '反问' in query_type:
            # 反问型：提取真实意图
            final_result = self.rewrite_rhetorical_query(query, conversation_history)
        else:
            # 其他类型：返回自动识别的改写结果
            final_result = result.get('rewritten_query', query)
        
        # 第三步：组装并返回完整结果
        return {
            "original_query": query,           # 原始查询
            "detected_type": query_type,       # 检测到的类型
            "confidence": result.get('confidence', 0.5),  # 置信度
            "rewritten_query": final_result,   # 改写后的查询
            "auto_rewrite_result": result      # 自动识别的原始结果
        }


# -----------------------------------------------------------------------------
# 主程序入口
# -----------------------------------------------------------------------------

def main():
    """
    主函数：演示 Query 改写器的各种使用场景
    """
    # 初始化 Query 改写器
    rewriter = QueryRewriter()
    
    # 打印标题
    print("=" * 60)
    print("Query 改写功能使用示例（迪士尼主题乐园）")
    print("=" * 60)
    print()
    
    # -------------------------------------------------------------------------
    # 示例 1: 上下文依赖型 Query
    # -------------------------------------------------------------------------
    print("示例 1: 上下文依赖型 Query")
    print("-" * 60)
    
    # 构造对话历史，模拟真实的多轮对话场景
    conversation_history = """
用户："我想了解一下上海迪士尼乐园的最新项目。"
AI："上海迪士尼乐园最新推出了'疯狂动物城'主题园区，这里有朱迪警官和尼克狐的互动体验。"
用户："这个园区有什么游乐设施？"
AI："'疯狂动物城'园区目前有疯狂动物城警察局、朱迪警官训练营和尼克狐的冰淇淋店等设施。"
"""
    # 当前查询包含"其他"，需要参考上下文才能理解
    current_query = "还有其他设施吗？"
    
    print(f"对话历史：{conversation_history}")
    print(f"当前查询：{current_query}")
    
    # 执行改写
    result = rewriter.rewrite_context_dependent_query(current_query, conversation_history)
    print(f"改写结果：{result}")
    print()
    
    # -------------------------------------------------------------------------
    # 示例 2: 对比型 Query
    # -------------------------------------------------------------------------
    print("示例 2: 对比型 Query")
    print("-" * 60)
    
    conversation_history = """
用户："我想了解一下上海迪士尼乐园的最新项目。"
AI："上海迪士尼乐园最新推出了疯狂动物城主题园区，还有蜘蛛侠主题园区"
"""
    # 当前查询包含"哪个"、"比较"，需要明确比较对象
    current_query = "哪个游玩的时间比较长，比较有趣"
    
    print(f"对话历史：{conversation_history}")
    print(f"当前查询：{current_query}")
    
    # 执行改写
    result = rewriter.rewrite_comparative_query(current_query, conversation_history)
    print(f"改写结果：{result}")
    print()
    
    # -------------------------------------------------------------------------
    # 示例 3: 模糊指代型 Query
    # -------------------------------------------------------------------------
    print("示例 3: 模糊指代型 Query")
    print("-" * 60)
    
    conversation_history = """
用户："我想了解一下上海迪士尼乐园和香港迪士尼乐园的烟花表演。"
AI："好的，上海迪士尼乐园和香港迪士尼乐园都有精彩的烟花表演。"
"""
    # 当前查询包含"都"，需要确定指代对象
    current_query = "都什么时候开始？"
    
    print(f"对话历史：{conversation_history}")
    print(f"当前查询：{current_query}")
    
    # 执行改写
    result = rewriter.rewrite_ambiguous_reference_query(current_query, conversation_history)
    print(f"改写结果：{result}")
    print()
    
    # -------------------------------------------------------------------------
    # 示例 4: 多意图型 Query
    # -------------------------------------------------------------------------
    print("示例 4: 多意图型 Query")
    print("-" * 60)
    
    # 当前查询包含三个独立问题，需要分解
    query = "门票多少钱？需要提前预约吗？停车费怎么收？"
    
    print(f"原始查询：{query}")
    
    # 执行改写（分解）
    result = rewriter.rewrite_multi_intent_query(query)
    print(f"分解结果：{result}")
    print()
    
    # -------------------------------------------------------------------------
    # 示例 5: 反问型 Query
    # -------------------------------------------------------------------------
    print("示例 5: 反问型 Query")
    print("-" * 60)
    
    conversation_history = """
用户："你好，我想预订下周六上海迪士尼乐园的门票。"
AI："正在为您查询... 查询到下周六的门票已经售罄。"
用户："售罄是什么意思？我朋友上周去还能买到当天的票。"
"""
    # 当前查询带有反问语气，需要提取真实意图
    current_query = "这不会也要提前一个月预订吧？"
    
    print(f"对话历史：{conversation_history}")
    print(f"当前查询：{current_query}")
    
    # 执行改写
    result = rewriter.rewrite_rhetorical_query(current_query, conversation_history)
    print(f"改写结果：{result}")
    print()
    
    # -------------------------------------------------------------------------
    # 示例 6: 自动识别 Query 类型
    # -------------------------------------------------------------------------
    print("示例 6: 自动识别 Query 类型")
    print("-" * 60)
    
    # 测试查询列表，覆盖各种类型
    test_queries = [
        "还有其他游乐项目吗？",        # 上下文依赖型
        "哪个园区更好玩？",            # 对比型
        "都适合小朋友吗？",            # 模糊指代型
        "有什么餐厅？价格怎么样？",    # 多意图型
        "这不会也要排队两小时吧？",    # 反问型
    ]
    
    # 逐个测试
    for i, query in enumerate(test_queries, 1):
        print(f"测试查询 {i}: {query}")
        result = rewriter.auto_rewrite_query(query)
        print(f"  识别类型：{result['query_type']}")
        print(f"  改写结果：{result['rewritten_query']}")
        print(f"  置信度：{result['confidence']}")
        print()
    
    # 打印完成提示
    print("=" * 60)
    print("所有示例执行完成")
    print("=" * 60)


# -----------------------------------------------------------------------------
# 程序入口
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
