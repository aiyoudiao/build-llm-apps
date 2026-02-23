"""
## 知识库版本管理与性能比较系统

知识库版本1 ──┐
            ├──→ 版本比较 ──→ 差异分析
知识库版本2 ──┘
              
知识库版本 ──→ 向量索引(FAISS) ──→ 性能评估 ──→ 准确率/响应时间
              
知识库版本 ──→ 回归测试 ──→ 通过率报告

> 对知识库进行版本化管理，使用向量检索技术评估不同版本的性能差异，支持持续迭代优化。


### 功能说明

1. 支持多版本知识库的创建与管理
2. 使用FAISS向量索引进行语义检索
3. 比较不同版本间的知识差异（增/删/改）
4. 评估版本检索性能（准确率/响应时间）
5. 生成回归测试报告

### 技术栈

- 通义千问 API：文本生成与嵌入
- dashscope SDK：阿里云百炼服务原生接口
- FAISS：Facebook向量相似度检索库
- numpy/pandas：数值计算与数据处理

### 工作流程

#### 版本创建流程

步骤1: 输入知识库内容
   ↓
步骤2: 对每个切片获取文本embedding（1024维向量）
   ↓
步骤3: 使用FAISS IndexFlatL2构建向量索引
   ↓
步骤4: 使用IndexIDMap建立ID映射
   ↓
步骤5: 存储版本信息（元数据+索引+统计）

#### 版本比较流程

步骤1: 获取两个版本的知识库
   ↓
步骤2: 创建ID映射字典
   ↓
步骤3: 对比ID集合
   ├── 新增ID = 版本2 ID - 版本1 ID
   ├── 删除ID = 版本1 ID - 版本2 ID
   └── 共同ID = 版本1 ID ∩ 版本2 ID
   ↓
步骤4: 对共同ID对比内容
   ├── 内容不同 → 修改
   └── 内容相同 → 未变
   ↓
步骤5: 输出差异报告

#### 性能评估流程

步骤1: 输入测试查询集
   ↓
步骤2: 对每个查询
   ├── 获取查询embedding
   ├── 使用FAISS检索top-k切片
   ├── 计算响应时间
   └── 评估检索质量（是否包含期望答案）
   ↓
步骤3: 计算整体指标
   ├── 准确率 = 正确数/总查询数
   └── 平均响应时间
   ↓
步骤4: 输出性能报告

"""

# -----------------------------------------------------------------------------
# 导入依赖库
# -----------------------------------------------------------------------------
import dashscope        # 阿里云百炼SDK，用于调用通义千问模型
import os               # 环境变量操作，用于获取API密钥
import json             # JSON数据解析与生成
import re               # 正则表达式（本代码中未直接使用）
from datetime import datetime, timedelta  # 时间处理，用于版本时间戳
from collections import defaultdict, Counter  # 字典与计数器（本代码中未直接使用）
import pandas as pd     # 数据处理（本代码中未直接使用）
import numpy as np      # 数值计算，用于向量操作
import faiss            # Facebook向量相似度检索库
from openai import OpenAI  # OpenAI兼容客户端，用于embedding调用

# -----------------------------------------------------------------------------
# 配置API密钥与客户端
# -----------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

# 初始化兼容OpenAI格式的百炼客户端
# 用于调用文本嵌入模型（embedding）
client = OpenAI(
    api_key=dashscope.api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# -----------------------------------------------------------------------------
# 全局配置
# -----------------------------------------------------------------------------
# 文本嵌入模型名称
TEXT_EMBEDDING_MODEL = "text-embedding-v4"
# 文本嵌入向量维度（1024维）
TEXT_EMBEDDING_DIM = 1024

# -----------------------------------------------------------------------------
# 工具函数定义
# -----------------------------------------------------------------------------

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


def get_text_embedding(text):
    """
    获取文本的向量嵌入（Embedding）
    
    功能：
        将文本转换为1024维的向量表示，用于语义相似度计算
    
    参数：
        text: 待嵌入的文本字符串
    
    返回：
        1024维的浮点数向量列表
    """
    # 调用OpenAI兼容接口获取文本嵌入
    response = client.embeddings.create(
        model=TEXT_EMBEDDING_MODEL,  # 指定嵌入模型
        input=text,                   # 输入文本
        dimensions=TEXT_EMBEDDING_DIM # 向量维度
    )
    
    # 返回第一个结果的嵌入向量
    return response.data[0].embedding


# =============================================================================
# 核心类：知识库版本管理器
# =============================================================================

class KnowledgeBaseVersionManager:
    """
    知识库版本管理器
    
    核心功能：
        1. 创建和管理多个知识库版本
        2. 为每个版本构建FAISS向量索引
        3. 比较版本间的知识差异
        4. 评估版本检索性能
        5. 生成回归测试报告
    
    属性：
        model: 使用的大模型名称
        versions: 存储所有版本信息的字典
    """
    
    def __init__(self, model="qwen-turbo-latest"):
        """
        初始化版本管理器
        
        参数：
            model: 使用的大模型名称，默认使用通义千问turbo版本
        """
        self.model = model
        self.versions = {}  # 存储所有版本，键为版本名，值为版本信息
    
    def create_version(self, knowledge_base, version_name, description=""):
        """
        创建知识库版本
        
        功能：
            1. 为知识库构建向量索引
            2. 计算版本统计信息
            3. 存储版本信息
        
        参数：
            knowledge_base: 知识库列表，每个元素包含id和content
            version_name: 版本名称（如v1.0）
            description: 版本描述
        
        返回：
            版本信息字典
        """
        # 构建向量索引，返回元数据存储和FAISS索引对象
        metadata_store, text_index = self.build_vector_index(knowledge_base)
        
        # 构建版本信息字典
        version_info = {
            "version_name": version_name,               # 版本名称
            "description": description,                 # 版本描述
            "created_date": datetime.now().isoformat(), # 创建时间
            "knowledge_base": knowledge_base,           # 知识库原文
            "metadata_store": metadata_store,           # 向量元数据
            "text_index": text_index,                   # FAISS索引对象
            "statistics": self.calculate_version_statistics(knowledge_base)  # 统计信息
        }
        
        # 存储版本信息
        self.versions[version_name] = version_info
        
        return version_info
    
    def build_vector_index(self, knowledge_base):
        """
        构建向量索引
        
        功能：
            1. 遍历知识库中的每个切片
            2. 获取每个切片的文本嵌入向量
            3. 使用FAISS构建向量检索索引
        
        参数：
            knowledge_base: 知识库列表
        
        返回：
            metadata_store: 元数据列表，包含id、content、chunk_id
            text_index_map: FAISS索引对象
        """
        metadata_store = []  # 存储向量元数据
        text_vectors = []    # 存储文本向量
        
        # 遍历知识库中的每个切片
        for i, chunk in enumerate(knowledge_base):
            content = chunk.get('content', '')
            
            # 跳过空内容
            if not content.strip():
                continue
            
            # 构建元数据
            metadata = {
                "id": i,                              # 内部ID
                "content": content,                   # 内容文本
                "chunk_id": chunk.get('id', f'chunk_{i}')  # 原始ID
            }
            
            # 获取文本的embedding向量
            vector = get_text_embedding(content)
            text_vectors.append(vector)
            metadata_store.append(metadata)
        
        # 创建FAISS索引
        # IndexFlatL2：基于L2欧氏距离的暴力搜索索引
        text_index = faiss.IndexFlatL2(TEXT_EMBEDDING_DIM)
        
        # IndexIDMap：为向量建立自定义ID映射，支持按ID检索
        text_index_map = faiss.IndexIDMap(text_index)
        
        # 如果有向量数据，添加到索引中
        if text_vectors:
            # 提取所有元数据的ID
            text_ids = [m["id"] for m in metadata_store]
            
            # 将向量和ID添加到索引中
            text_index_map.add_with_ids(
                np.array(text_vectors).astype('float32'),  # 向量数组
                np.array(text_ids)                          # ID数组
            )
        
        return metadata_store, text_index_map
    
    def calculate_version_statistics(self, knowledge_base):
        """
        计算版本统计信息
        
        参数：
            knowledge_base: 知识库列表
        
        返回：
            包含切片数量、总长度、平均长度的统计字典
        """
        # 计算知识切片总数
        total_chunks = len(knowledge_base)
        
        # 计算所有内容字符数总和
        total_content_length = sum(len(chunk.get('content', '')) for chunk in knowledge_base)
        
        # 计算平均切片长度
        average_chunk_length = total_content_length / total_chunks if total_chunks > 0 else 0
        
        return {
            "total_chunks": total_chunks,           # 切片总数
            "total_content_length": total_content_length,  # 总字符数
            "average_chunk_length": average_chunk_length   # 平均长度
        }
    
    def compare_versions(self, version1_name, version2_name):
        """
        比较两个版本的差异
        
        功能：
            1. 检测新增的知识切片
            2. 检测删除的知识切片
            3. 检测修改的知识切片
            4. 比较统计信息变化
        
        参数：
            version1_name: 版本1名称
            version2_name: 版本2名称
        
        返回：
            包含差异检测和统计比较的结果字典
        """
        # 检查版本是否存在
        if version1_name not in self.versions or version2_name not in self.versions:
            return {"error": "版本不存在"}
        
        # 获取两个版本的信息
        v1 = self.versions[version1_name]
        v2 = self.versions[version2_name]
        
        # 获取知识库内容
        kb1 = v1['knowledge_base']
        kb2 = v2['knowledge_base']
        
        # 构建比较结果
        comparison = {
            "version1": version1_name,                      # 版本1名称
            "version2": version2_name,                      # 版本2名称
            "comparison_date": datetime.now().isoformat(),  # 比较时间
            "changes": self.detect_changes(kb1, kb2),       # 差异检测
            "statistics_comparison": self.compare_statistics(  # 统计比较
                v1['statistics'], v2['statistics']
            )
        }
        
        return comparison
    
    def detect_changes(self, kb1, kb2):
        """
        检测知识库变化
        
        功能：
            通过对比ID集合和内容，识别增删改操作
        
        参数：
            kb1: 版本1的知识库
            kb2: 版本2的知识库
        
        返回：
            包含新增、删除、修改、未变切片的差异字典
        """
        # 初始化变化记录
        changes = {
            "added_chunks": [],      # 新增切片
            "removed_chunks": [],    # 删除切片
            "modified_chunks": [],   # 修改切片
            "unchanged_chunks": []   # 未变切片
        }
        
        # 创建ID到切片的映射字典，便于快速查找
        kb1_dict = {chunk.get('id'): chunk for chunk in kb1}
        kb2_dict = {chunk.get('id'): chunk for chunk in kb2}
        
        # 获取两个版本的ID集合
        kb1_ids = set(kb1_dict.keys())
        kb2_ids = set(kb2_dict.keys())
        
        # 计算集合差异
        added_ids = kb2_ids - kb1_ids       # 版本2新增的ID
        removed_ids = kb1_ids - kb2_ids     # 版本1删除的ID
        common_ids = kb1_ids & kb2_ids      # 两个版本共有的ID
        
        # 记录新增的知识切片
        for chunk_id in added_ids:
            changes["added_chunks"].append({
                "id": chunk_id,
                "content": kb2_dict[chunk_id].get('content', '')
            })
        
        # 记录删除的知识切片
        for chunk_id in removed_ids:
            changes["removed_chunks"].append({
                "id": chunk_id,
                "content": kb1_dict[chunk_id].get('content', '')
            })
        
        # 检测修改的知识切片
        for chunk_id in common_ids:
            chunk1 = kb1_dict[chunk_id]
            chunk2 = kb2_dict[chunk_id]
            
            # 对比内容是否相同
            if chunk1.get('content') != chunk2.get('content'):
                changes["modified_chunks"].append({
                    "id": chunk_id,
                    "old_content": chunk1.get('content', ''),
                    "new_content": chunk2.get('content', '')
                })
            else:
                changes["unchanged_chunks"].append(chunk_id)
        
        return changes
    
    def compare_statistics(self, stats1, stats2):
        """
        比较统计信息
        
        功能：
            计算两个版本统计数据的差异和变化百分比
        
        参数：
            stats1: 版本1的统计信息
            stats2: 版本2的统计信息
        
        返回：
            包含版本间差异的比较字典
        """
        comparison = {}
        
        # 遍历统计信息的每个键
        for key in stats1.keys():
            if key in stats2:
                # 数值类型的统计
                if isinstance(stats1[key], (int, float)):
                    comparison[key] = {
                        "version1": stats1[key],
                        "version2": stats2[key],
                        "difference": stats2[key] - stats1[key],
                        "percentage_change": (
                            (stats2[key] - stats1[key]) / stats1[key] * 100
                        ) if stats1[key] != 0 else 0
                    }
                # 字典类型的统计
                elif isinstance(stats1[key], dict):
                    comparison[key] = self.compare_dict_statistics(stats1[key], stats2[key])
        
        return comparison
    
    def compare_dict_statistics(self, dict1, dict2):
        """
        比较字典类型的统计信息
        
        参数：
            dict1: 版本1的字典统计
            dict2: 版本2的字典统计
        
        返回：
            包含差异的比较字典
        """
        comparison = {}
        # 获取所有键的并集
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        # 遍历所有键
        for key in all_keys:
            val1 = dict1.get(key, 0)
            val2 = dict2.get(key, 0)
            comparison[key] = {
                "version1": val1,
                "version2": val2,
                "difference": val2 - val1
            }
        
        return comparison
    
    def evaluate_version_performance(self, version_name, test_queries):
        """
        评估版本性能
        
        功能：
            1. 对每个测试查询执行检索
            2. 记录响应时间
            3. 评估检索质量
            4. 计算整体指标
        
        参数：
            version_name: 版本名称
            test_queries: 测试查询列表，包含query和expected_answer
        
        返回：
            包含查询结果和整体指标的性能报告
        """
        # 检查版本是否存在
        if version_name not in self.versions:
            return {"error": "版本不存在"}
        
        # 初始化性能指标
        performance_metrics = {
            "version_name": version_name,           # 版本名称
            "evaluation_date": datetime.now().isoformat(),  # 评估时间
            "query_results": [],                    # 每个查询的结果
            "overall_metrics": {}                   # 整体指标
        }
        
        total_queries = len(test_queries)  # 总查询数
        correct_answers = 0                 # 正确回答数
        response_times = []                 # 响应时间列表
        
        # 遍历每个测试查询
        for query_info in test_queries:
            query = query_info['query']
            expected_answer = query_info.get('expected_answer', '')
            
            # 记录开始时间
            start_time = datetime.now()
            
            # 执行检索
            retrieved_chunks = self.retrieve_relevant_chunks(query, version_name)
            
            # 记录结束时间
            end_time = datetime.now()
            
            # 计算响应时间（秒）
            response_time = (end_time - start_time).total_seconds()
            response_times.append(response_time)
            
            # 评估检索质量
            is_correct = self.evaluate_retrieval_quality(
                query, retrieved_chunks, expected_answer
            )
            if is_correct:
                correct_answers += 1
            
            # 记录查询结果
            performance_metrics["query_results"].append({
                "query": query,
                "retrieved_chunks": len(retrieved_chunks),
                "response_time": response_time,
                "is_correct": is_correct
            })
        
        # 计算整体指标
        accuracy = correct_answers / total_queries if total_queries > 0 else 0
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        performance_metrics["overall_metrics"] = {
            "accuracy": accuracy,               # 准确率
            "avg_response_time": avg_response_time,  # 平均响应时间
            "total_queries": total_queries,     # 总查询数
            "correct_answers": correct_answers  # 正确回答数
        }
        
        return performance_metrics
    
    def retrieve_relevant_chunks(self, query, version_name, k=3):
        """
        使用embedding和FAISS检索相关知识切片
        
        功能：
            1. 获取查询的向量嵌入
            2. 使用FAISS检索最相似的k个切片
            3. 返回带相似度分数的结果
        
        参数：
            query: 查询文本
            version_name: 版本名称
            k: 返回结果数量，默认3个
        
        返回：
            相关知识切片列表，包含id、content、similarity_score
        """
        # 检查版本是否存在
        if version_name not in self.versions:
            return []
        
        # 获取版本信息
        version_info = self.versions[version_name]
        metadata_store = version_info['metadata_store']
        text_index = version_info['text_index']
        
        # 获取查询的embedding向量
        query_vector = np.array([get_text_embedding(query)]).astype('float32')
        
        # 使用FAISS进行检索
        # distances: 距离数组，indices: 索引数组
        distances, indices = text_index.search(query_vector, k)
        
        relevant_chunks = []
        
        # 遍历检索结果
        for i, doc_id in enumerate(indices[0]):
            # FAISS返回-1表示没有找到匹配
            if doc_id != -1:
                # 通过ID在元数据中查找
                match = next(
                    (item for item in metadata_store if item["id"] == doc_id), 
                    None
                )
                if match:
                    # 构造返回的知识切片格式
                    chunk = {
                        "id": match["chunk_id"],
                        "content": match["content"],
                        # 将L2距离转换为相似度分数（距离越小相似度越高）
                        "similarity_score": 1.0 / (1.0 + distances[0][i])
                    }
                    relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def evaluate_retrieval_quality(self, query, retrieved_chunks, expected_answer):
        """
        评估检索质量
        
        功能：
            检查检索结果中是否包含期望答案的关键词
        
        参数：
            query: 查询文本
            retrieved_chunks: 检索到的知识切片列表
            expected_answer: 期望答案
        
        返回：
            布尔值，表示检索是否正确
        """
        # 空结果直接返回False
        if not retrieved_chunks:
            return False
        
        # 遍历检索结果
        for chunk in retrieved_chunks:
            content = chunk.get('content', '').lower()
            # 检查期望答案是否在内容中（不区分大小写）
            if expected_answer.lower() in content:
                return True
        
        return False
    
    def compare_version_performance(self, version1_name, version2_name, test_queries):
        """
        比较两个版本的性能
        
        功能：
            1. 分别评估两个版本的性能
            2. 对比准确率和响应时间
            3. 生成推荐建议
        
        参数：
            version1_name: 版本1名称
            version2_name: 版本2名称
            test_queries: 测试查询列表
        
        返回：
            包含性能对比和推荐建议的比较报告
        """
        # 分别评估两个版本
        perf1 = self.evaluate_version_performance(version1_name, test_queries)
        perf2 = self.evaluate_version_performance(version2_name, test_queries)
        
        # 检查是否有错误
        if "error" in perf1 or "error" in perf2:
            return {"error": "版本评估失败"}
        
        # 构建比较结果
        comparison = {
            "version1": version1_name,
            "version2": version2_name,
            "comparison_date": datetime.now().isoformat(),
            "performance_comparison": {
                "accuracy": {
                    "version1": perf1["overall_metrics"]["accuracy"],
                    "version2": perf2["overall_metrics"]["accuracy"],
                    "improvement": perf2["overall_metrics"]["accuracy"] - perf1["overall_metrics"]["accuracy"]
                },
                "response_time": {
                    "version1": perf1["overall_metrics"]["avg_response_time"],
                    "version2": perf2["overall_metrics"]["avg_response_time"],
                    "improvement": perf1["overall_metrics"]["avg_response_time"] - perf2["overall_metrics"]["avg_response_time"]
                }
            },
            "recommendation": self.generate_performance_recommendation(perf1, perf2)
        }
        
        return comparison
    
    def generate_performance_recommendation(self, perf1, perf2):
        """
        生成性能建议
        
        功能：
            根据准确率和响应时间的对比，给出版本选择建议
        
        参数：
            perf1: 版本1的性能报告
            perf2: 版本2的性能报告
        
        返回：
            推荐建议字符串
        """
        # 提取关键指标
        acc1 = perf1["overall_metrics"]["accuracy"]
        acc2 = perf2["overall_metrics"]["accuracy"]
        time1 = perf1["overall_metrics"]["avg_response_time"]
        time2 = perf2["overall_metrics"]["avg_response_time"]
        
        # 根据指标对比生成建议
        if acc2 > acc1 and time2 <= time1:
            # 版本2准确率更高且响应时间不增加
            return f"推荐使用版本2，准确率提升{(acc2-acc1)*100:.1f}%，响应时间{'提升' if time2 < time1 else '相当'}"
        elif acc2 > acc1 and time2 > time1:
            # 版本2准确率更高但响应时间增加
            return f"版本2准确率更高但响应时间较长，需要权衡"
        elif acc2 < acc1 and time2 < time1:
            # 版本2响应更快但准确率降低
            return f"版本2响应更快但准确率较低，需要权衡"
        else:
            # 版本1性能更优
            return f"推荐使用版本1，性能更优"
    
    def generate_regression_test(self, version_name, test_queries):
        """
        生成回归测试报告
        
        功能：
            1. 对每个测试用例执行检索
            2. 记录通过/失败状态
            3. 计算测试通过率
        
        参数：
            version_name: 版本名称
            test_queries: 测试查询列表
        
        返回：
            包含测试结果和通过率的回归测试报告
        """
        # 检查版本是否存在
        if version_name not in self.versions:
            return {"error": "版本不存在"}
        
        # 初始化回归测试结果
        regression_results = {
            "version_name": version_name,           # 版本名称
            "test_date": datetime.now().isoformat(), # 测试时间
            "test_results": [],                     # 每个测试用例的结果
            "pass_rate": 0                          # 通过率
        }
        
        passed_tests = 0        # 通过的测试数
        total_tests = len(test_queries)  # 总测试数
        
        # 遍历每个测试查询
        for query_info in test_queries:
            query = query_info['query']
            expected_answer = query_info.get('expected_answer', '')
            
            # 执行检索
            retrieved_chunks = self.retrieve_relevant_chunks(query, version_name)
            
            # 评估是否通过
            is_passed = self.evaluate_retrieval_quality(
                query, retrieved_chunks, expected_answer
            )
            
            if is_passed:
                passed_tests += 1
            
            # 记录测试结果
            regression_results["test_results"].append({
                "query": query,
                "expected": expected_answer,
                "retrieved": len(retrieved_chunks),
                "passed": is_passed
            })
        
        # 计算通过率
        regression_results["pass_rate"] = passed_tests / total_tests if total_tests > 0 else 0
        
        return regression_results


# =============================================================================
# 主函数入口
# =============================================================================

def main():
    """
    主函数：演示知识库版本管理系统的完整使用流程
    """
    # 初始化版本管理器
    version_manager = KnowledgeBaseVersionManager()
    
    print("=== 知识库版本管理与性能比较示例（迪士尼主题乐园） ===\n")
    
    # -------------------------------------------------------------------------
    # 准备版本1（基础版本）- 3条知识
    # -------------------------------------------------------------------------
    knowledge_base_v1 = [
        {
            "id": "kb_001",
            "content": "上海迪士尼乐园位于上海市浦东新区，是中国大陆首座迪士尼主题乐园，"
                      "于2016年6月16日开园。"
        },
        {
            "id": "kb_002",
            "content": "上海迪士尼乐园的门票价格：平日成人票价为399元，"
                      "周末和节假日为499元。"
        },
        {
            "id": "kb_003",
            "content": "上海迪士尼乐园营业时间为上午8:00至晚上8:00。"
        }
    ]
    
    # -------------------------------------------------------------------------
    # 准备版本2（增强版本）- 5条知识，内容更丰富
    # -------------------------------------------------------------------------
    knowledge_base_v2 = [
        {
            "id": "kb_001",
            "content": "上海迪士尼乐园位于上海市浦东新区，是中国大陆首座迪士尼主题乐园，"
                      "于2016年6月16日开园。乐园占地面积390公顷，包含七大主题园区。"
        },
        {
            "id": "kb_002",
            "content": "上海迪士尼乐园的门票价格：平日成人票价为399元，周末和节假日为499元。"
                      "儿童票（1.0-1.4米）平日为299元，周末为374元。1.0米以下儿童免费。"
        },
        {
            "id": "kb_003",
            "content": "上海迪士尼乐园营业时间为上午8:00至晚上8:00，全年无休。"
                      "建议出发前查看官方网站确认具体时间。"
        },
        {
            "id": "kb_004",
            "content": "从上海市区到迪士尼乐园可以乘坐地铁11号线到迪士尼站，"
                      "或乘坐迪士尼专线巴士。"
        },
        {
            "id": "kb_005",
            "content": "上海迪士尼乐园的特色项目包括：创极速光轮、七个小矮人矿山车、"
                      "加勒比海盗等。"
        }
    ]
    
    # -------------------------------------------------------------------------
    # 功能1：创建知识库版本
    # -------------------------------------------------------------------------
    print("功能1: 创建知识库版本")
    v1_info = version_manager.create_version(knowledge_base_v1, "v1.0", "基础版本")
    v2_info = version_manager.create_version(knowledge_base_v2, "v2.0", "增强版本")
    
    print(f"版本1信息:")
    print(f"  版本名：{v1_info['version_name']}")
    print(f"  描述：{v1_info['description']}")
    print(f"  知识切片数量：{v1_info['statistics']['total_chunks']}")
    print(f"  平均切片长度：{v1_info['statistics']['average_chunk_length']:.0f}字符")
    
    print(f"\n版本2信息:")
    print(f"  版本名：{v2_info['version_name']}")
    print(f"  描述：{v2_info['description']}")
    print(f"  知识切片数量：{v2_info['statistics']['total_chunks']}")
    print(f"  平均切片长度：{v2_info['statistics']['average_chunk_length']:.0f}字符")
    
    print("\n" + "=" * 60 + "\n")
    
    # -------------------------------------------------------------------------
    # 功能2：版本差异比较
    # -------------------------------------------------------------------------
    print("功能2: 版本差异比较")
    comparison = version_manager.compare_versions("v1.0", "v2.0")
    
    print(f"版本比较结果:")
    changes = comparison['changes']
    print(f"  新增知识切片：{len(changes['added_chunks'])}个")
    print(f"  删除知识切片：{len(changes['removed_chunks'])}个")
    print(f"  修改知识切片：{len(changes['modified_chunks'])}个")
    
    print(f"\n新增的知识切片:")
    for i, chunk in enumerate(changes['added_chunks'], 1):
        print(f"  {i}. ID: {chunk['id']}")
        print(f"     内容：{chunk['content']}")
    
    print(f"\n修改的知识切片:")
    for i, chunk in enumerate(changes['modified_chunks'], 1):
        print(f"  {i}. ID: {chunk['id']}")
        print(f"     旧内容：{chunk['old_content']}")
        print(f"     新内容：{chunk['new_content']}")
    
    print("\n" + "=" * 60 + "\n")
    
    # -------------------------------------------------------------------------
    # 功能3：版本性能评估
    # -------------------------------------------------------------------------
    print("功能3: 版本性能评估")
    
    # 准备测试查询集
    test_queries = [
        {"query": "上海迪士尼乐园在哪里？", "expected_answer": "浦东新区"},
        {"query": "门票多少钱？", "expected_answer": "价格"},
        {"query": "营业时间是什么？", "expected_answer": "8:00"},
        {"query": "怎么去迪士尼？", "expected_answer": "地铁"},
        {"query": "有什么好玩的项目？", "expected_answer": "项目"}
    ]
    
    # 评估两个版本的性能
    perf_v1 = version_manager.evaluate_version_performance("v1.0", test_queries)
    perf_v2 = version_manager.evaluate_version_performance("v2.0", test_queries)
    
    print(f"版本1性能:")
    print(f"  准确率：{perf_v1['overall_metrics']['accuracy']*100:.1f}%")
    print(f"  平均响应时间：{perf_v1['overall_metrics']['avg_response_time']*1000:.1f}ms")
    
    print(f"\n版本2性能:")
    print(f"  准确率：{perf_v2['overall_metrics']['accuracy']*100:.1f}%")
    print(f"  平均响应时间：{perf_v2['overall_metrics']['avg_response_time']*1000:.1f}ms")
    
    print("\n" + "=" * 60 + "\n")
    
    # -------------------------------------------------------------------------
    # 功能4：性能比较与建议
    # -------------------------------------------------------------------------
    print("功能4: 性能比较与建议")
    perf_comparison = version_manager.compare_version_performance("v1.0", "v2.0", test_queries)
    
    print(f"性能比较结果:")
    comp = perf_comparison['performance_comparison']
    print(f"  准确率提升：{comp['accuracy']['improvement']*100:.1f}%")
    print(f"  响应时间变化：{comp['response_time']['improvement']*1000:.1f}ms")
    print(f"  建议：{perf_comparison['recommendation']}")
    
    print("\n" + "=" * 60 + "\n")
    
    # -------------------------------------------------------------------------
    # 功能5：回归测试
    # -------------------------------------------------------------------------
    print("功能5: 回归测试")
    regression_v2 = version_manager.generate_regression_test("v2.0", test_queries)
    
    print(f"回归测试结果:")
    print(f"  测试通过率：{regression_v2['pass_rate']*100:.1f}%")
    print(f"  测试用例数量：{len(regression_v2['test_results'])}")
    
    print(f"\n详细测试结果:")
    for i, result in enumerate(regression_v2['test_results'], 1):
        status = "通过" if result['passed'] else "失败"
        print(f"  {i}. {result['query']} [{status}]")


# =============================================================================
# 程序入口
# =============================================================================

if __name__ == "__main__":
    main()
