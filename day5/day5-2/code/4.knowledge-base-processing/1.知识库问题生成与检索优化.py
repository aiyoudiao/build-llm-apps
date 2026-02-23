"""
## 知识库问题生成与检索优化系统 - BM25版本

> 用户查询 → BM25检索 → 匹配知识库内容/问题 → 返回最相关知识
> 为知识库内容预先生成多样化问题，用户查询时既可以匹配原文，也可以匹配生成的问题，从而提高检索准确率。

### 功能说明

1. 使用大语言模型为知识库内容生成多样化问题
2. 构建BM25检索索引（原文索引 + 问题索引）
3. 支持两种检索方式并对比效果
4. 提升用户查询与知识库内容的匹配准确率

### 技术栈

- 百炼/通义千问 API：问题生成
- jieba：中文分词
- rank_bm25：BM25检索算法
- numpy/pandas：数据处理

步骤1: 准备知识库
   ↓
步骤2: 为每个知识切片AI生成问题
   ↓
步骤3: 构建两个BM25索引
   ├── 原文索引（content_bm25）
   └── 问题索引（question_bm25）
   ↓
步骤4: 用户查询时
   ├── 方法A: 直接匹配原文
   └── 方法B: 匹配生成的问题
   ↓
步骤5: 比较两种方法的检索效果

"""

# -----------------------------------------------------------------------------
# 导入依赖库
# -----------------------------------------------------------------------------
import os           # 环境变量操作
import json         # JSON数据解析与生成
import numpy as np  # 数值计算，用于排序和数组操作
from openai import OpenAI  # OpenAI兼容API客户端
import pandas as pd        # 数据处理（本代码中未直接使用）
from datetime import datetime  # 时间处理（本代码中未直接使用）
from rank_bm25 import BM25Okapi  # BM25检索算法实现
import jieba        # 中文分词工具
import re           # 正则表达式，用于文本清洗

# -----------------------------------------------------------------------------
# 配置API密钥与客户端
# -----------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')

# 初始化兼容OpenAI格式的百炼客户端
# base_url指向阿里云百炼服务的兼容接口
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# -----------------------------------------------------------------------------
# 工具函数定义
# -----------------------------------------------------------------------------

def preprocess_json_response(response):
    """
    预处理大模型返回的JSON响应
    
    功能：移除响应中可能包含的markdown代码块标记，确保JSON可被正确解析
    
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
        response = response[7:]  # 移除 ```json 前缀
    elif response.startswith('```'):
        response = response[3:]  # 移除 ``` 前缀
    
    # 移除结尾的markdown代码块标记
    if response.endswith('```'):
        response = response[:-3]  # 移除 ``` 后缀
    
    # 去除首尾空白字符
    return response.strip()


def get_completion(prompt, model="qwen-turbo-latest"):
    """
    调用大语言模型生成文本响应
    
    参数：
        prompt: 用户输入的提示词
        model: 使用的模型名称，默认使用通义千问turbo版本
    
    返回：
        模型生成的文本内容
    """
    # 构建符合OpenAI格式的消息列表
    messages = [{"role": "user", "content": prompt}]
    
    # 调用API生成响应
    response = client.chat.completions.create(
        model=model,           # 指定模型
        messages=messages,     # 对话消息
        temperature=0.7,       # 温度参数，控制生成随机性（0.7为平衡值）
    )
    
    # 提取并返回生成的文本内容
    return response.choices[0].message.content


def preprocess_text(text):
    """
    文本预处理与分词函数
    
    功能：
        1. 移除标点符号和特殊字符
        2. 使用jieba进行中文分词
        3. 过滤停用词和过短的词
    
    参数：
        text: 待处理的原始文本
    
    返回：
        分词后的词语列表
    """
    # 空文本直接返回空列表
    if not text:
        return []
    
    # 使用正则表达式移除所有非字母数字和空白的字符
    text = re.sub(r'[^\w\s]', '', text)
    
    # 使用jieba进行精确模式分词
    words = jieba.lcut(text)
    
    # 定义中文停用词表
    stop_words = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
        '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', 
        '你', '会', '着', '没有', '看', '好', '自己', '这'
    }
    
    # 过滤条件：词长大于1且不在停用词表中
    words = [word for word in words if len(word) > 1 and word not in stop_words]
    
    return words


# =============================================================================
# 核心类：知识库优化器
# =============================================================================

class KnowledgeBaseOptimizer:
    """
    知识库问题生成与检索优化器
    
    核心功能：
        1. 为知识库内容生成多样化问题
        2. 构建BM25检索索引（原文索引 + 问题索引）
        3. 支持多种检索方式
        4. 评估不同检索方法的效果
    
    属性：
        model: 使用的大模型名称
        knowledge_base: 知识库内容列表
        content_bm25: 原文内容的BM25索引对象
        question_bm25: 生成问题的BM25索引对象
        content_documents: 原文分词后的文档列表
        question_documents: 问题分词后的文档列表
        content_metadata: 原文内容的元数据
        question_metadata: 问题内容的元数据
    """
    
    def __init__(self, model="qwen-turbo-latest"):
        """
        初始化优化器
        
        参数：
            model: 使用的大模型名称，默认使用通义千问turbo版本
        """
        self.model = model
        self.knowledge_base = []
        self.content_bm25 = None
        self.question_bm25 = None
        self.content_documents = []
        self.question_documents = []
        self.content_metadata = []
        self.question_metadata = []
    
    def generate_questions_for_chunk(self, knowledge_chunk, num_questions=5):
        """
        为单个知识切片生成多样化问题
        
        功能：
            使用大模型分析知识内容，生成多个不同角度和难度的问题
        
        参数：
            knowledge_chunk: 知识内容文本
            num_questions: 生成问题的数量，默认5个
        
        返回：
            问题列表，每个问题包含问题内容、类型和难度
        """
        # 定义问题生成的指令模板
        instruction = """
你是一个专业的问答系统专家。给定的知识内容能回答哪些多样化的问题，这些问题可以：
1. 使用不同的问法（直接问、间接问、对比问等）
2. 避免重复和相似的问题
3. 确保问题不超出知识内容范围

请返回JSON格式：
{
    "questions": [
        {
            "question": "问题内容",
            "question_type": "问题类型（直接问/间接问/对比问/条件问等）",
            "difficulty": "难度等级（简单/中等/困难）"
        }
    ]
}
"""
        
        # 构建完整的提示词，包含指令、知识内容和生成数量
        prompt = f"""
### 指令 ###
{instruction}

### 知识内容 ###
{knowledge_chunk}

### 生成问题数量 ###
{num_questions}

### 生成结果 ###
"""
        
        # 调用大模型生成问题
        response = get_completion(prompt, self.model)
        
        # 预处理响应，移除markdown格式
        response = preprocess_json_response(response)
        
        # 尝试解析JSON响应
        try:
            result = json.loads(response)
            return result.get('questions', [])
        except json.JSONDecodeError as e:
            # JSON解析失败时的容错处理
            print(f"JSON解析失败：{e}")
            print(f"AI返回内容：{response[:50]}...")
            # 返回默认问题作为降级方案
            return [{
                "question": f"关于{knowledge_chunk[:50]}...的问题", 
                "question_type": "直接问", 
                "keywords": [], 
                "difficulty": "中等"
            }]
    
    def build_knowledge_index(self, knowledge_base):
        """
        构建知识库的BM25索引
        
        功能：
            1. 遍历知识库中的每个切片
            2. 对原文内容建立索引
            3. 对生成的问题建立索引（如果存在）
        
        参数：
            knowledge_base: 知识库列表，每个元素包含content和generated_questions
        """
        print("正在构建知识库索引...")
        
        # 保存知识库引用
        self.knowledge_base = knowledge_base
        
        # 初始化存储容器
        content_documents = []    # 原文分词文档
        question_documents = []   # 问题分词文档
        content_metadata = []     # 原文元数据
        question_metadata = []    # 问题元数据
        
        # 遍历知识库中的每个切片
        for i, chunk in enumerate(knowledge_base):
            # 获取知识切片的内容
            text = chunk.get('content', '')
            
            # 跳过空内容
            if not text.strip():
                continue
            
            # -------------------------
            # 构建原文索引
            # -------------------------
            content_words = preprocess_text(text)
            if content_words:
                content_documents.append(content_words)
                content_metadata.append({
                    "id": chunk.get('id', f"chunk_{i}"),  # 文档ID
                    "content": text,                       # 原始内容
                    "category": chunk.get('category', ''), # 分类标签
                    "chunk": chunk,                        # 完整切片数据
                    "type": "content"                      # 文档类型标记
                })
            
            # -------------------------
            # 构建问题索引
            # -------------------------
            if 'generated_questions' in chunk and chunk['generated_questions']:
                for j, question_data in enumerate(chunk['generated_questions']):
                    question = question_data.get('question', '')
                    
                    # 跳过空问题
                    if question.strip():
                        # 拼接原文和问题，保持上下文关联
                        combined_text = f"内容：{text} 问题：{question}"
                        question_words = preprocess_text(combined_text)
                        
                        if question_words:
                            question_documents.append(question_words)
                            question_metadata.append({
                                "id": f"{chunk.get('id', f'chunk_{i}')}_q{j}",
                                "content": question,
                                "combined_content": combined_text,
                                "category": chunk.get('category', ''),
                                "chunk": chunk,
                                "type": "question",
                                "question_data": question_data
                            })
        
        # -------------------------
        # 创建BM25索引对象
        # -------------------------
        if content_documents:
            self.content_bm25 = BM25Okapi(content_documents)
            self.content_documents = content_documents
            self.content_metadata = content_metadata
            print(f"原文索引构建完成，共索引 {len(content_documents)} 个知识切片")
        
        if question_documents:
            self.question_bm25 = BM25Okapi(question_documents)
            self.question_documents = question_documents
            self.question_metadata = question_metadata
            print(f"问题索引构建完成，共索引 {len(question_documents)} 个问题")
        
        # 检查是否有有效内容
        if not content_documents and not question_documents:
            print("没有有效的内容可以索引")
    
    def search_similar_chunks(self, query, k=3, search_type="content"):
        """
        使用BM25算法搜索相似内容
        
        参数：
            query: 用户查询文本
            k: 返回结果数量，默认3个
            search_type: 检索类型，"content"为原文检索，"question"为问题检索
        
        返回：
            搜索结果列表，包含元数据、分数和相似度
        """
        # 根据检索类型选择对应的索引和元数据
        if search_type == "content":
            if not self.content_bm25:
                return []
            bm25 = self.content_bm25
            metadata_store = self.content_metadata
        elif search_type == "question":
            if not self.question_bm25:
                return []
            bm25 = self.question_bm25
            metadata_store = self.question_metadata
        else:
            return []
        
        try:
            # 预处理查询文本
            query_words = preprocess_text(query)
            if not query_words:
                return []
            
            # 计算BM25相关性分数
            scores = bm25.get_scores(query_words)
            
            # 获取分数最高的k个索引
            top_indices = np.argsort(scores)[::-1][:k]
            
            # 构建结果列表
            results = []
            for idx in top_indices:
                # 只返回有相关性的结果（分数大于0）
                if scores[idx] > 0:
                    metadata = metadata_store[idx]
                    # 将BM25分数归一化到0-1范围
                    similarity = min(1.0, scores[idx] / 10.0)
                    results.append({
                        "metadata": metadata,
                        "score": scores[idx],
                        "similarity": similarity
                    })
            
            return results
            
        except Exception as e:
            # 异常处理
            print(f"搜索失败：{e}")
            return []
    
    def calculate_similarity(self, query, knowledge_chunk):
        """
        计算查询与知识切片的BM25相似度
        
        参数：
            query: 查询文本
            knowledge_chunk: 知识切片文本
        
        返回：
            归一化后的相似度分数（0-1范围）
        """
        try:
            # 分别对查询和知识切片进行分词
            query_words = preprocess_text(query)
            chunk_words = preprocess_text(knowledge_chunk)
            
            # 空文本返回0分
            if not query_words or not chunk_words:
                return 0.0
            
            # 创建临时BM25索引（仅包含当前知识切片）
            temp_bm25 = BM25Okapi([chunk_words])
            
            # 计算相关性分数
            scores = temp_bm25.get_scores(query_words)
            
            # 返回最高分数并归一化
            max_score = max(scores) if scores else 0.0
            return min(1.0, max_score / 10.0)
            
        except Exception as e:
            print(f"相似度计算失败：{e}")
            return 0.0
    
    def calculate_question_similarity(self, user_query, generated_questions):
        """
        计算用户查询与生成问题集合的最大相似度
        
        参数：
            user_query: 用户查询文本
            generated_questions: 生成的问题列表
        
        返回：
            最高相似度分数
        """
        similarities = []
        
        # 遍历所有生成问题，计算相似度
        for question_data in generated_questions:
            question = question_data['question']
            similarity = self.calculate_similarity(user_query, question)
            similarities.append(similarity)
        
        # 返回最高相似度
        return max(similarities) if similarities else 0.0
    
    def evaluate_retrieval_methods(self, knowledge_base, test_queries):
        """
        评估两种检索方法的准确度
        
        功能：
            对比原文检索和问题检索在测试集上的表现
        
        参数：
            knowledge_base: 知识库数据
            test_queries: 测试查询列表，每个包含query和correct_chunk
        
        返回：
            评估结果字典，包含准确率、分数等详细信息
        """
        # 首先构建知识库索引
        self.build_knowledge_index(knowledge_base)
        
        # 初始化结果存储结构
        results = {
            'content_similarity': [],    # 原文检索是否正确
            'question_similarity': [],   # 问题检索是否正确
            'improvement': [],           # 问题检索是否有改进
            'content_scores': [],        # 原文检索分数
            'question_scores': [],       # 问题检索分数
            'query_details': []          # 每个查询的详细信息
        }
        
        # 遍历每个测试查询
        for i, query_info in enumerate(test_queries):
            user_query = query_info['query']
            correct_chunk = query_info['correct_chunk']
            
            # -------------------------
            # 方法1：BM25原文检索
            # -------------------------
            content_results = self.search_similar_chunks(
                user_query, k=1, search_type="content"
            )
            content_correct = False
            content_score = 0.0
            content_chunk_id = None
            
            if content_results:
                best_match = content_results[0]['metadata']['chunk']
                # 判断是否检索到正确答案
                content_correct = best_match['content'] == correct_chunk
                content_score = content_results[0]['similarity']
                content_chunk_id = best_match['id']
            
            # -------------------------
            # 方法2：BM25问题检索
            # -------------------------
            question_results = self.search_similar_chunks(
                user_query, k=1, search_type="question"
            )
            question_correct = False
            question_score = 0.0
            question_chunk_id = None
            
            if question_results:
                best_match = question_results[0]['metadata']['chunk']
                question_correct = best_match['content'] == correct_chunk
                question_score = question_results[0]['similarity']
                question_chunk_id = best_match['id']
            
            # 记录评估结果
            results['content_similarity'].append(content_correct)
            results['question_similarity'].append(question_correct)
            results['improvement'].append(question_correct and not content_correct)
            results['content_scores'].append(content_score)
            results['question_scores'].append(question_score)
            
            # 记录查询详情
            results['query_details'].append({
                'query': user_query,
                'content_score': content_score,
                'question_score': question_score,
                'content_correct': content_correct,
                'question_correct': question_correct,
                'score_diff': question_score - content_score,
                'content_chunk_id': content_chunk_id,
                'question_chunk_id': question_chunk_id
            })
        
        return results
    
    def generate_diverse_questions(self, knowledge_chunk, num_questions=8):
        """
        生成更丰富多样化的问题
        
        与generate_questions_for_chunk相比，此方法生成更多维度的问题信息
        
        参数：
            knowledge_chunk: 知识内容文本
            num_questions: 生成问题数量，默认8个
        
        返回：
            问题列表，包含类型、难度、角度、可回答性和答案
        """
        # 定义更详细的问题生成指令
        instruction = """
你是一个专业的问答系统专家。请为给定的知识内容生成高度多样化的问题，确保：
1. 问题类型多样化：直接问、间接问、对比问、条件问、假设问、推理问等
2. 表达方式多样化：使用不同的句式、词汇、语气
3. 难度层次多样化：简单、中等、困难的问题都要有
4. 角度多样化：从不同角度和维度提问
5. 确保问题不超出知识内容范围

请返回JSON格式：
{
    "questions": [
        {
            "question": "问题内容",
            "question_type": "问题类型",
            "difficulty": "难度等级",
            "perspective": "提问角度",
            "is_answerable": "给出的知识能否回答该问题",
            "answer": "基于该知识的回答"
        }
    ]
}
"""
        
        # 构建完整提示词
        prompt = f"""
### 指令 ###
{instruction}

### 知识内容 ###
{knowledge_chunk}

### 生成问题数量 ###
{num_questions}

### 生成结果 ###
"""
        
        # 调用大模型生成
        response = get_completion(prompt, self.model)
        
        # 预处理响应
        response = preprocess_json_response(response)
        
        # 解析JSON
        try:
            result = json.loads(response)
            return result.get('questions', [])
        except json.JSONDecodeError as e:
            print(f"多样化问题生成JSON解析失败：{e}")
            print(f"AI返回内容：{response[:200]}...")
            return []


# =============================================================================
# 主函数入口
# =============================================================================

def main():
    """
    主函数：演示知识库优化系统的完整使用流程
    """
    # 初始化知识库优化器
    optimizer = KnowledgeBaseOptimizer()
    
    print("=== 知识库问题生成与检索优化示例（BM25版本）- 迪士尼主题乐园 ===\n")
    
    # -------------------------------------------------------------------------
    # 准备示例知识库
    # -------------------------------------------------------------------------
    knowledge_base = [
        {
            "id": "kb_001",
            "content": "上海迪士尼乐园位于上海市浦东新区，是中国大陆首座迪士尼主题乐园，"
                      "于2016年6月16日开园。乐园占地面积390公顷，包含七大主题园区："
                      "米奇大街、奇想花园、探险岛、宝藏湾、明日世界、梦幻世界和迪士尼小镇。",
            "category": "基本信息"
        },
        {
            "id": "kb_002", 
            "content": "上海迪士尼乐园的门票价格根据季节和日期有所不同。平日成人票价为399元，"
                      "周末和节假日为499元。儿童票（1.0-1.4米）平日为299元，周末和节假日为374元。"
                      "1.0米以下儿童免费入园。",
            "category": "价格信息"
        },
        {
            "id": "kb_003",
            "content": "上海迪士尼乐园的营业时间通常为上午8:00至晚上8:00，但具体时间会根据"
                      "季节和特殊活动进行调整。建议游客在出发前查看官方网站或APP获取最新的"
                      "营业时间信息。",
            "category": "营业信息"
        },
        {
            "id": "kb_004",
            "content": "从上海市区到上海迪士尼乐园有多种交通方式：1. 地铁11号线迪士尼站下车；"
                      "2. 乘坐迪士尼专线巴士；3. 打车约40-60分钟；4. 自驾车可停在乐园停车场，"
                      "停车费为100元/天。",
            "category": "交通信息"
        },
        {
            "id": "kb_005",
            "content": "上海迪士尼乐园的特色项目包括：创极速光轮（明日世界）、七个小矮人矿山车"
                      "（梦幻世界）、加勒比海盗：战争之潮（宝藏湾）、翱翔·飞越地平线（探险岛）等。"
                      "这些项目都有不同的身高和年龄限制。",
            "category": "游乐项目"
        },
        {
            "id": "kb_006",
            "content": "上海迪士尼乐园提供多种餐饮选择，包括米奇大街的皇家宴会厅、奇想花园的"
                      "漫月轩、宝藏湾的巴波萨烧烤等。园内餐厅价格相对较高，人均消费约150-300元。"
                      "建议游客可以携带密封包装的零食和水入园。",
            "category": "餐饮信息"
        },
        {
            "id": "kb_007",
            "content": "上海迪士尼乐园的购物体验非常丰富，每个主题园区都有特色商店。米奇大街的"
                      "M大街购物廊是最大的综合商店，销售各种迪士尼周边商品。建议游客在离园前"
                      "购买纪念品，避免携带不便。",
            "category": "购物信息"
        },
        {
            "id": "kb_008",
            "content": "上海迪士尼乐园提供多种服务设施，包括婴儿车租赁（50元/天）、轮椅租赁"
                      "（免费）、储物柜（60元/天）、充电宝租赁等。园内设有多个医疗点和失物"
                      "招领处，为游客提供便利服务。",
            "category": "服务设施"
        }
    ]
    
    # -------------------------------------------------------------------------
    # 示例1：为知识切片生成问题
    # -------------------------------------------------------------------------
    print("示例1: 为知识切片生成多样化问题")
    test_chunk = knowledge_base[0]['content']
    print(f"知识内容：{test_chunk}")
    
    questions = optimizer.generate_questions_for_chunk(test_chunk, num_questions=5)
    print(f"\n生成的5个问题:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q['question']} (类型：{q['question_type']}, 难度：{q['difficulty']})")
    
    print("\n" + "=" * 60 + "\n")
    
    # -------------------------------------------------------------------------
    # 示例2：生成更多样化的问题
    # -------------------------------------------------------------------------
    print("示例2: 生成更多样化的问题（8个）")
    diverse_questions = optimizer.generate_diverse_questions(test_chunk, num_questions=8)
    print(f"\n生成的8个多样化问题:")
    for i, q in enumerate(diverse_questions, 1):
        print(f"  {i}. {q['question']}")
        print(f"     类型：{q['question_type']}, 难度：{q['difficulty']}, "
              f"角度：{q['perspective']}, 能否回答：{q['is_answerable']}, "
              f"回答的答案：{q['answer']}")
    
    print("\n" + "=" * 60 + "\n")
    
    # -------------------------------------------------------------------------
    # 示例3：评估检索方法
    # -------------------------------------------------------------------------
    print("示例3: 评估两种检索方法的准确度")
    
    # 设计测试查询，覆盖不同知识类别
    test_queries = [
        {
            "query": "如果我想体验最刺激的过山车，应该去哪个区域？",
            "correct_chunk": knowledge_base[4]['content']
        },
        {
            "query": "什么时间去人比较少？",
            "correct_chunk": knowledge_base[2]['content']
        },
        {
            "query": "可以带食物进去吗？",
            "correct_chunk": knowledge_base[5]['content']
        }
    ]
    
    # 为知识库中的每个切片生成问题
    print('正在为知识库生成问题...')
    for chunk in knowledge_base:
        chunk['generated_questions'] = optimizer.generate_questions_for_chunk(chunk['content'])
    print('为知识库生成问题完毕')
    
    # 执行检索方法评估
    results = optimizer.evaluate_retrieval_methods(knowledge_base, test_queries)
    
    # 输出评估统计
    print(f"测试查询数量：{len(test_queries)}")
    print(f"BM25原文检索准确率：{sum(results['content_similarity'])/len(results['content_similarity'])*100:.1f}%")
    print(f"BM25问题检索准确率：{sum(results['question_similarity'])/len(results['question_similarity'])*100:.1f}%")
    print(f"问题检索改进的查询数量：{sum(results['improvement'])}")
    
    # -------------------------------------------------------------------------
    # 详细分析输出
    # -------------------------------------------------------------------------
    print(f"\n=== 详细分析 ===")
    
    # 按相似度分数差异排序
    sorted_details = sorted(results['query_details'], key=lambda x: x['score_diff'], reverse=True)
    
    # 输出问题检索表现更好的查询
    print(f"\n问题检索方法表现更好的查询（按分数差异排序）:")
    for i, detail in enumerate(sorted_details[:5], 1):
        if detail['score_diff'] > 0:
            print(f"  {i}. 查询：{detail['query']}")
            print(f"     原文检索分数：{detail['content_score']:.3f}")
            print(f"     问题检索分数：{detail['question_score']:.3f}")
            print(f"     分数差异：+{detail['score_diff']:.3f}")
            print(f"     原文检索：{'正确' if detail['content_correct'] else '错误'}")
            print(f"     问题检索：{'正确' if detail['question_correct'] else '错误'}")
    
    # 输出原文检索表现更好的查询
    print(f"\n原文检索方法表现更好的查询:")
    for i, detail in enumerate(sorted_details[-5:], 1):
        if detail['score_diff'] < 0:
            print(f"  {i}. 查询：{detail['query']}")
            print(f"     原文检索分数：{detail['content_score']:.3f}")
            print(f"     问题检索分数：{detail['question_score']:.3f}")
            print(f"     分数差异：{detail['score_diff']:.3f}")
            print(f"     原文检索：{'正确' if detail['content_correct'] else '错误'}")
            print(f"     问题检索：{'正确' if detail['question_correct'] else '错误'}")


# =============================================================================
# 程序入口
# =============================================================================

if __name__ == "__main__":
    main()
