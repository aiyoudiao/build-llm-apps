"""
## 对话知识提取与沉淀系统

> 对话记录 → AI提取知识 → 批量处理 → 合并相似知识 → 沉淀到知识库
> 从大量客服对话中自动提取事实性知识，过滤掉临时性信息（如用户需求、具体问题），将重复出现的知识合并成结构化条目。

### 功能说明

1. 从客服对话记录中自动提取有价值的知识点
2. 支持批量处理多条对话
3. 过滤临时性信息（用户需求、具体问题）
4. 使用大语言模型合并相似知识
5. 输出结构化的知识库条目

### 技术栈

- 通义千问 API：知识提取与合并
- dashscope SDK：阿里云百炼服务原生接口
- collections.Counter：知识频率统计


步骤1: 输入对话记录
   ↓
步骤2: AI提取知识（事实/需求/问题/流程/注意）
   ↓
步骤3: 批量处理多条对话
   ↓
步骤4: 过滤掉"需求"和"问题"类型（临时性信息）
   ↓
步骤5: 按知识类型分组
   ↓
步骤6: 使用LLM合并相似知识
   ↓
步骤7: 输出结构化知识库
"""

# -----------------------------------------------------------------------------
# 导入依赖库
# -----------------------------------------------------------------------------
import dashscope        # 阿里云百炼SDK，用于调用通义千问模型
import os               # 环境变量操作，用于获取API密钥
import json             # JSON数据解析与生成
from datetime import datetime  # 时间处理（本代码中未直接使用）
from collections import Counter  # 计数器，用于统计知识出现频率

# -----------------------------------------------------------------------------
# 配置API密钥
# -----------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

# -----------------------------------------------------------------------------
# 工具函数定义
# -----------------------------------------------------------------------------

def preprocess_json_response(response):
    """
    预处理大模型返回的JSON响应
    
    功能：
        移除响应中可能包含的markdown代码块标记，确保JSON可被正确解析
    
    参数：
        response: 大模型返回的原始字符串
    
    返回：
        清理后的纯JSON字符串
    """
    # 空响应直接返回空字符串
    if not response:
        return ""
    
    # 移除开头的markdown代码块标记
    if response.startswith('```json'):
        response = response[7:]  # 移除 ```json 前缀（7个字符）
    elif response.startswith('```'):
        response = response[3:]  # 移除 ``` 前缀（3个字符）
    
    # 移除结尾的markdown代码块标记
    if response.endswith('```'):
        response = response[:-3]  # 移除 ``` 后缀
    
    # 去除首尾空白字符后返回
    return response.strip()


def get_completion(prompt, model="qwen-turbo-latest"):
    """
    调用通义千问模型生成文本响应
    
    参数：
        prompt: 用户输入的提示词
        model: 使用的模型名称，默认使用通义千问turbo版本
    
    返回：
        模型生成的文本内容
    """
    # 构建符合dashscope格式的消息列表
    messages = [{"role": "user", "content": prompt}]
    
    # 调用dashscope API生成响应
    response = dashscope.Generation.call(
        model=model,            # 指定模型名称
        messages=messages,      # 对话消息
        result_format='message', # 返回格式为message
        temperature=0.3,        # 温度参数，0.3降低随机性提高稳定性
    )
    
    # 提取并返回生成的文本内容
    return response.output.choices[0].message.content


# =============================================================================
# 核心类：对话知识提取器
# =============================================================================

class ConversationKnowledgeExtractor:
    """
    对话知识提取与沉淀器
    
    核心功能：
        1. 从单次对话中提取结构化知识
        2. 批量处理多条对话记录
        3. 过滤临时性知识（需求/问题）
        4. 使用LLM合并相似知识
        5. 统计知识出现频率
    
    属性：
        model: 使用的大模型名称
        extracted_knowledge: 提取的知识列表
        knowledge_frequency: 知识出现频率计数器
    """
    
    def __init__(self, model="qwen-turbo-latest"):
        """
        初始化知识提取器
        
        参数：
            model: 使用的大模型名称，默认使用通义千问turbo版本
        """
        self.model = model
        self.extracted_knowledge = []
        self.knowledge_frequency = Counter()
    
    def extract_knowledge_from_conversation(self, conversation):
        """
        从单次对话中提取知识
        
        功能：
            使用大模型分析对话内容，提取事实、需求、问题、流程、注意等知识点
        
        参数：
            conversation: 对话文本内容
        
        返回：
            包含提取知识、对话摘要和用户意图的字典
        """
        # 定义知识提取的指令模板
        instruction = """
你是一个专业的知识提取专家。请从给定的对话中提取有价值的知识点，包括：
1. 事实性信息（地点、时间、价格、规则等）
2. 用户需求和偏好
3. 常见问题和解答
4. 操作流程和步骤
5. 注意事项和提醒

请返回JSON格式：
{
    "extracted_knowledge": [
        {
            "knowledge_type": "知识类型（事实/需求/问题/流程/注意）",
            "content": "知识内容",
            "confidence": "置信度(0-1)",
            "source": "来源（用户/AI/对话）",
            "keywords": ["关键词1", "关键词2"],
            "category": "分类"
        }
    ],
    "conversation_summary": "对话摘要",
    "user_intent": "用户意图"
}
"""
        
        # 构建完整的提示词，包含指令和对话内容
        prompt = f"""
### 指令 ###
{instruction}

### 对话内容 ###
{conversation}

### 提取结果 ###
"""
        
        # 调用大模型提取知识
        response = get_completion(prompt, self.model)
        
        # 预处理响应，移除markdown格式
        response = preprocess_json_response(response)
        
        # 尝试解析JSON响应
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            # JSON解析失败时的容错处理
            print(f"对话知识提取JSON解析失败：{e}")
            print(f"AI返回内容：{response[:200]}...")
            # 返回空结果作为降级方案
            return {
                "extracted_knowledge": [],
                "conversation_summary": "无法解析对话",
                "user_intent": "未知"
            }
    
    def batch_extract_knowledge(self, conversations):
        """
        批量提取知识
        
        功能：
            遍历多条对话，逐一提取知识并统计出现频率
        
        参数：
            conversations: 对话列表
        
        返回：
            所有提取的知识列表
        """
        all_knowledge = []
        
        # 遍历每条对话
        for i, conversation in enumerate(conversations):
            print(f"正在处理对话 {i+1}/{len(conversations)}...")
            
            # 从当前对话提取知识
            result = self.extract_knowledge_from_conversation(conversation)
            
            # 将提取的知识添加到总列表
            all_knowledge.extend(result.get('extracted_knowledge', []))
            
            # 更新频率统计
            for knowledge in result.get('extracted_knowledge', []):
                # 构建知识唯一键（类型 + 内容前50字符）
                key = f"{knowledge['knowledge_type']}:{knowledge['content'][:50]}"
                self.knowledge_frequency[key] += 1
        
        return all_knowledge
    
    def merge_similar_knowledge(self, knowledge_list):
        """
        合并相似的知识点
        
        功能：
            1. 过滤掉需求和问题类型的知识（临时性、个性化信息）
            2. 按知识类型分组
            3. 使用LLM合并同组内的相似知识
        
        参数：
            knowledge_list: 知识列表
        
        返回：
            合并后的知识列表
        """
        # 过滤掉需求和问题类型的知识，因为它们是临时的、个性化的
        # 只保留可复用的通用知识（事实、流程、注意）
        filtered_knowledge = [
            knowledge for knowledge in knowledge_list 
            if knowledge.get('knowledge_type') not in ['需求', '问题']
        ]
        
        # 输出过滤统计信息
        print(f"过滤前知识点数量：{len(knowledge_list)}")
        print(f"过滤后知识点数量：{len(filtered_knowledge)}")
        print(f"过滤掉的'需求'和'问题'类型知识点：{len(knowledge_list) - len(filtered_knowledge)}")
        
        # 按知识类型分组
        knowledge_by_type = {}
        for knowledge in filtered_knowledge:
            knowledge_type = knowledge.get('knowledge_type', '其他')
            if knowledge_type not in knowledge_by_type:
                knowledge_by_type[knowledge_type] = []
            knowledge_by_type[knowledge_type].append(knowledge)
        
        merged_knowledge = []
        
        # 对每个知识类型分别进行LLM合并
        for knowledge_type, knowledge_group in knowledge_by_type.items():
            if len(knowledge_group) == 1:
                # 只有一个知识点，直接添加无需合并
                merged_knowledge.append(knowledge_group[0])
            else:
                # 多个知识点，使用LLM智能合并
                merged = self.merge_knowledge_with_llm(knowledge_group, knowledge_type)
                merged_knowledge.append(merged)
        
        return merged_knowledge
    
    def merge_knowledge_with_llm(self, knowledge_group, knowledge_type):
        """
        使用LLM合并同类型的知识组
        
        功能：
            将多个相似知识点合并为一个更完整、准确的知识点
        
        参数：
            knowledge_group: 同类型的知识列表
            knowledge_type: 知识类型
        
        返回：
            合并后的知识字典
        """
        # 准备知识内容列表，用于构建提示词
        knowledge_contents = []
        all_keywords = set()   # 收集所有关键词
        all_sources = []       # 收集所有来源
        
        # 遍历知识组中的每个知识点
        for i, knowledge in enumerate(knowledge_group, 1):
            content = knowledge.get('content', '')
            confidence = knowledge.get('confidence', 0.5)
            keywords = knowledge.get('keywords', [])
            source = knowledge.get('source', '')
            category = knowledge.get('category', '')
            
            # 构建知识详情字符串
            knowledge_contents.append(f"{i}. 内容：{content}")
            knowledge_contents.append(f"   置信度：{confidence}")
            knowledge_contents.append(f"   分类：{category}")
            knowledge_contents.append(f"   来源：{source}")
            knowledge_contents.append(f"   关键词：{', '.join(keywords)}")
            knowledge_contents.append("")
            
            # 收集关键词和来源
            all_keywords.update(keywords)
            if source and source not in all_sources:
                all_sources.append(source)
        
        # 构建LLM合并提示词
        prompt = f"""
你是一个专业的知识整理专家。请将以下{knowledge_type}类型的知识点进行智能合并，生成一个更完整、准确的知识点。

### 合并要求：
1. 保留所有重要信息，避免信息丢失
2. 消除重复内容，整合相似表述
3. 提高内容的准确性和完整性
4. 保持逻辑清晰，结构合理
5. 合并后的置信度取所有知识点中的最高值

### 待合并的知识点：
{chr(10).join(knowledge_contents)}

### 请返回JSON格式：
{{
    "knowledge_type": "{knowledge_type}",
    "content": "合并后的知识内容",
    "confidence": 最高置信度值,
    "keywords": ["合并后的关键词列表"],
    "category": "合并后的分类",
    "sources": ["所有来源"],
    "frequency": {len(knowledge_group)}
}}

### 合并结果：
"""
        
        # 调用大模型进行知识合并
        response = get_completion(prompt, self.model)
        
        # 预处理响应
        response = preprocess_json_response(response)
        
        # 尝试解析JSON响应
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            # JSON解析失败时的容错处理
            print(f"知识合并JSON解析失败：{e}")
            print(f"AI返回内容：{response[:200]}...")
            
            # 使用简单的合并策略作为降级方案
            # 选择置信度最高的知识点作为代表
            best_knowledge = max(knowledge_group, key=lambda x: x.get('confidence', 0))
            return {
                "knowledge_type": knowledge_type,
                "content": best_knowledge['content'],
                "confidence": best_knowledge.get('confidence', 0.5),
                "frequency": len(knowledge_group),
                "keywords": list(all_keywords),
                "category": best_knowledge['category'],
                "sources": all_sources
            }


# =============================================================================
# 主函数入口
# =============================================================================

def main():
    """
    主函数：演示对话知识提取系统的完整使用流程
    """
    # 初始化对话知识提取器
    extractor = ConversationKnowledgeExtractor()
    
    print("=== 对话知识提取与沉淀示例（迪士尼主题乐园） ===\n")
    
    # -------------------------------------------------------------------------
    # 准备示例对话数据
    # -------------------------------------------------------------------------
    sample_conversations = [
        """
用户："我想去上海迪士尼乐园玩，门票多少钱？"
AI："上海迪士尼乐园的门票价格根据日期有所不同。平日成人票价为399元，周末和节假日为499元。儿童票（1.0-1.4米）平日为299元，周末为374元。1.0米以下儿童免费。"
用户："需要提前预订吗？"
AI："建议提前预订，特别是周末和节假日，可以通过官方网站或第三方平台预订。"
用户："从浦东机场怎么去？"
AI："从浦东机场到迪士尼乐园可以乘坐地铁2号线到广兰路站，然后换乘11号线到迪士尼站，全程约1小时。也可以打车，约40分钟。"
""",
        """
用户："迪士尼乐园今天开放吗？"
AI："上海迪士尼乐园通常每天开放，营业时间为上午8:00至晚上8:00。但建议您出发前查看官方网站或APP确认具体营业时间，因为可能会因特殊活动或维护调整。"
用户："现在人多不多？"
AI："人流量会因日期、季节和活动而变化。周末、节假日和寒暑假通常人较多。建议您查看官方APP的实时人流量信息，或关注社交媒体上的游客分享。"
用户："有什么必玩的项目？"
AI："上海迪士尼乐园的必玩项目包括：创极速光轮（明日世界）、七个小矮人矿山车（梦幻世界）、加勒比海盗：战争之潮（宝藏湾）、翱翔·飞越地平线（探险岛）等。"
""",
        """
用户："带小孩去迪士尼需要注意什么？"
AI："带小孩去迪士尼需要注意：1. 身高限制，部分项目有身高要求；2. 准备防晒、雨具等；3. 带些零食和水；4. 建议下载官方APP查看实时排队时间；5. 可以租用婴儿车；6. 注意安全，看管好孩子。"
用户："停车费怎么收？"
AI："上海迪士尼乐园停车场收费为100元/天，建议提前到达，因为停车场可能会满。也可以选择地铁出行，11号线迪士尼站就在乐园门口。"
用户："可以带食物进去吗？"
AI："可以携带密封包装的零食和水，但不能带玻璃瓶、酒精饮料等。园内有多个餐厅和小吃店，价格相对较高，建议合理安排。"
"""
    ]
    
    # -------------------------------------------------------------------------
    # 示例1：从单次对话中提取知识
    # -------------------------------------------------------------------------
    print("示例1: 从单次对话中提取知识")
    conversation = sample_conversations[0]
    print(f"对话内容:\n{conversation}")
    
    # 执行知识提取
    extracted = extractor.extract_knowledge_from_conversation(conversation)
    
    # 输出提取结果
    print(f"\n提取的知识点:")
    for i, knowledge in enumerate(extracted['extracted_knowledge'], 1):
        print(f"  {i}. 类型：{knowledge['knowledge_type']}")
        print(f"     内容：{knowledge['content']}")
        print(f"     置信度：{knowledge['confidence']}")
        print(f"     分类：{knowledge['category']}")
    
    print(f"\n对话摘要：{extracted['conversation_summary']}")
    print(f"用户意图：{extracted['user_intent']}")
    
    print("\n" + "=" * 60 + "\n")
    
    # -------------------------------------------------------------------------
    # 示例2：批量提取知识
    # -------------------------------------------------------------------------
    print("示例2: 批量提取知识")
    all_knowledge = extractor.batch_extract_knowledge(sample_conversations)
    print(f"总共提取了 {len(all_knowledge)} 个知识点")
    
    # 显示知识频率统计
    print(f"\n所有知识点:")
    for key, count in extractor.knowledge_frequency.most_common():
        print(f"  {key}: {count}次")
    
    print("\n" + "=" * 60 + "\n")
    
    # -------------------------------------------------------------------------
    # 示例3：合并相似知识
    # -------------------------------------------------------------------------
    print("示例3: 合并相似知识")
    merged_knowledge = extractor.merge_similar_knowledge(all_knowledge)
    print(f"合并后剩余 {len(merged_knowledge)} 个知识点")
    
    # 输出合并后的知识详情
    print(f"\n合并后的知识点:")
    for i, knowledge in enumerate(merged_knowledge, 1):
        print(f"  {i}. 类型：{knowledge.get('knowledge_type', '未知')}")
        print(f"     内容：{knowledge['content']}")
        print(f"     频率：{knowledge.get('frequency', 1)}次")
        print(f"     置信度：{knowledge.get('confidence', 0.5)}")
        print(f"     分类：{knowledge.get('category', '未知')}")
        print(f"     关键词：{knowledge.get('keywords', [])}")
        print(f"     来源：{knowledge.get('sources', [])}")
        print()
    
    print("\n" + "=" * 60 + "\n")


# =============================================================================
# 程序入口
# =============================================================================

if __name__ == "__main__":
    main()
