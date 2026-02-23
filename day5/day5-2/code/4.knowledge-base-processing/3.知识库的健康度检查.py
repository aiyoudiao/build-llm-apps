"""
## 知识库健康度检查系统

知识库 + 测试查询 → LLM分析 → 健康度报告
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
    缺少知识    过期知识    冲突知识
        ↓           ↓           ↓
    覆盖率      新鲜度      一致性
                    ↓
              综合健康评分

> 使用LLM作为检查器，从三个维度评估知识库质量，生成可操作的健康度报告和改进建议。

### 功能说明

1. 检查知识库的完整性（缺少的知识）
2. 检查知识库的时效性（过期的知识）
3. 检查知识库的一致性（冲突的知识）
4. 计算综合健康度评分
5. 生成改进建议报告

### 技术栈

- 通义千问 API：知识分析与检查
- dashscope SDK：阿里云百炼服务原生接口
- datetime：时间处理


步骤1: 输入知识库和测试查询
   ↓
步骤2: LLM检查缺少的知识
   ├── 分析查询能否在知识库中找到答案
   ├── 识别知识空白
   └── 输出覆盖率评分
   ↓
步骤3: LLM检查过期的知识
   ├── 检查时间、价格、政策等信息
   ├── 识别需要更新的内容
   └── 输出新鲜度评分
   ↓
步骤4: LLM检查冲突的知识
   ├── 对比不同切片的信息
   ├── 识别矛盾冲突
   └── 输出一致性评分
   ↓
步骤5: 计算综合健康评分
   ├── 覆盖率 × 40%
   ├── 新鲜度 × 30%
   └── 一致性 × 30%
   ↓
步骤6: 生成健康度报告和改进建议

"""

# -----------------------------------------------------------------------------
# 导入依赖库
# -----------------------------------------------------------------------------
import dashscope        # 阿里云百炼SDK，用于调用通义千问模型
import os               # 环境变量操作，用于获取API密钥
import json             # JSON数据解析与生成
import re               # 正则表达式（本代码中未直接使用）
from datetime import datetime  # 时间处理，用于记录检查时间和判断知识时效

# -----------------------------------------------------------------------------
# 配置API密钥
# -----------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

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


# =============================================================================
# 核心类：知识库健康度检查器
# =============================================================================

class KnowledgeBaseHealthChecker:
    """
    知识库健康度检查器
    
    核心功能：
        1. 检查缺少的知识（完整性维度）
        2. 检查过期的知识（时效性维度）
        3. 检查冲突的知识（一致性维度）
        4. 计算综合健康度评分
        5. 生成改进建议报告
    
    属性：
        model: 使用的大模型名称
        health_report: 存储健康度报告结果
    """
    
    def __init__(self, model="qwen-turbo-latest"):
        """
        初始化健康度检查器
        
        参数：
            model: 使用的大模型名称，默认使用通义千问turbo版本
        """
        self.model = model
        self.health_report = {}
    
    def check_missing_knowledge(self, knowledge_base, test_queries):
        """
        使用LLM检查缺少的知识（完整性检查）
        
        功能：
            分析测试查询能否在知识库中找到答案，识别知识盲区
        
        参数：
            knowledge_base: 知识库列表，包含id和content
            test_queries: 测试查询列表，包含query和expected_answer
        
        返回：
            包含缺少知识列表、覆盖率评分和完整性分析的字典
        """
        # 定义知识完整性检查的指令模板
        instruction = """
你是一个知识库完整性检查专家。请分析给定的测试查询和知识库内容，判断知识库中是否缺少相关的知识。

检查标准：
1. 查询是否能在知识库中找到相关答案
2. 知识是否完整、准确
3. 是否覆盖了用户的主要需求
4. 是否存在知识空白

请返回JSON格式：
{
    "missing_knowledge": [
        {
            "query": "测试查询",
            "missing_aspect": "缺少的知识方面",
            "importance": "重要性（高/中/低）",
            "suggested_content": "建议的知识内容",
            "category": "知识分类"
        }
    ],
    "coverage_score": "覆盖率评分(0-1)",
    "completeness_analysis": "完整性分析"
}
"""
        
        # 构建知识库内容摘要，将所有知识切片格式化为字符串
        knowledge_summary = []
        for chunk in knowledge_base:
            knowledge_summary.append(f"ID: {chunk.get('id', 'unknown')} - {chunk.get('content', '')}")
        
        # 将知识库摘要拼接为完整文本
        knowledge_text = "\n".join(knowledge_summary)
        
        # 构建测试查询列表，格式化为字符串
        queries_text = []
        for query_info in test_queries:
            query = query_info['query']
            expected = query_info.get('expected_answer', '')
            queries_text.append(f"查询：{query} | 期望答案：{expected}")
        
        # 将测试查询拼接为完整文本
        queries_text = "\n".join(queries_text)
        
        # 构建完整的提示词，包含指令、知识库内容和测试查询
        prompt = f"""
### 指令 ###
{instruction}

### 知识库内容 ###
{knowledge_text}

### 测试查询 ###
{queries_text}

### 分析结果 ###
"""
        
        try:
            # 调用大模型进行分析
            response = get_completion(prompt, self.model)
            
            # 预处理响应，移除markdown代码块格式
            if response.startswith('```json'):
                response = response[7:]  # 移除 ```json 前缀
            elif response.startswith('```'):
                response = response[3:]  # 移除 ``` 前缀
            if response.endswith('```'):
                response = response[:-3]  # 移除 ``` 后缀
            
            # 解析JSON响应
            result = json.loads(response.strip())
            return result
            
        except Exception as e:
            # 异常处理，返回None
            print(f"LLM检查缺少知识失败：{e}")
            return None
    
    def check_outdated_knowledge(self, knowledge_base):
        """
        使用LLM检查过期的知识（时效性检查）
        
        功能：
            分析知识内容中的时间、价格、政策等信息是否过期
        
        参数：
            knowledge_base: 知识库列表，包含id、content和last_updated
        
        返回：
            包含过期知识列表、新鲜度评分和更新建议的字典
        """
        # 定义知识时效性检查的指令模板
        instruction = """
你是一个知识时效性检查专家。请分析给定的知识内容，判断是否存在过期或需要更新的信息。

检查标准：
1. 时间相关信息是否过期（年份、日期、时间范围）
2. 价格信息是否最新（价格、费用、票价等）
3. 政策规则是否更新（政策、规定、规则等）
4. 活动信息是否有效（活动、节日、特殊安排等）
5. 联系方式是否准确（电话、地址、网址等）
6. 技术信息是否过时（版本、技术标准等）

请返回JSON格式：
{
    "outdated_knowledge": [
        {
            "chunk_id": "知识切片ID",
            "content": "知识内容",
            "outdated_aspect": "过期方面",
            "severity": "严重程度（高/中/低）",
            "suggested_update": "建议更新内容",
            "last_verified": "最后验证时间"
        }
    ],
    "freshness_score": "新鲜度评分(0-1)",
    "update_recommendations": "更新建议"
}
"""
        
        # 构建知识库内容，包含ID、更新时间和内容
        knowledge_text = []
        for chunk in knowledge_base:
            content = chunk.get('content', '')
            chunk_id = chunk.get('id', 'unknown')
            last_updated = chunk.get('last_updated', 'unknown')
            knowledge_text.append(f"ID: {chunk_id} | 更新时间：{last_updated} | 内容：{content}")
        
        # 将知识库内容拼接为完整文本
        knowledge_text = "\n".join(knowledge_text)
        
        # 构建完整的提示词，包含指令、知识库内容和当前时间
        prompt = f"""
### 指令 ###
{instruction}

### 知识库内容 ###
{knowledge_text}

### 当前时间 ###
{datetime.now().strftime('%Y年%m月%d日')}

### 分析结果 ###
"""
        
        try:
            # 调用大模型进行分析
            response = get_completion(prompt, self.model)
            
            # 预处理响应，移除markdown代码块格式
            if response.startswith('```json'):
                response = response[7:]  # 移除 ```json 前缀
            elif response.startswith('```'):
                response = response[3:]  # 移除 ``` 前缀
            if response.endswith('```'):
                response = response[:-3]  # 移除 ``` 后缀
            
            # 解析JSON响应
            result = json.loads(response.strip())
            return result
            
        except Exception as e:
            # 异常处理，返回None
            print(f"LLM检查过期知识失败：{e}")
            return None
    
    def check_conflicting_knowledge(self, knowledge_base):
        """
        使用LLM检查冲突的知识（一致性检查）
        
        功能：
            分析知识库中是否存在矛盾或冲突的信息
        
        参数：
            knowledge_base: 知识库列表，包含id和content
        
        返回：
            包含冲突知识列表、一致性评分和冲突分析的字典
        """
        # 定义知识一致性检查的指令模板
        instruction = """
你是一个知识一致性检查专家。请分析给定的知识库，找出可能存在冲突或矛盾的信息。

检查标准：
1. 同一主题的不同说法（地点、名称、描述等）
2. 价格信息的差异（价格、费用、收费标准等）
3. 时间信息的不一致（营业时间、开放时间、活动时间等）
4. 规则政策的冲突（规定、政策、要求等）
5. 操作流程的差异（步骤、方法、流程等）
6. 联系方式的差异（地址、电话、网址等）

请返回JSON格式：
{
    "conflicting_knowledge": [
        {
            "conflict_type": "冲突类型",
            "chunk_ids": ["相关切片ID"],
            "conflicting_content": ["冲突内容"],
            "severity": "严重程度（高/中/低）",
            "resolution_suggestion": "解决建议"
        }
    ],
    "consistency_score": "一致性评分(0-1)",
    "conflict_analysis": "冲突分析"
}
"""
        
        # 构建知识库内容，包含ID和内容
        knowledge_text = []
        for chunk in knowledge_base:
            content = chunk.get('content', '')
            chunk_id = chunk.get('id', 'unknown')
            knowledge_text.append(f"ID: {chunk_id} | 内容：{content}")
        
        # 将知识库内容拼接为完整文本
        knowledge_text = "\n".join(knowledge_text)
        
        # 构建完整的提示词，包含指令和知识库内容
        prompt = f"""
### 指令 ###
{instruction}

### 知识库内容 ###
{knowledge_text}

### 分析结果 ###
"""
        
        try:
            # 调用大模型进行分析
            response = get_completion(prompt, self.model)
            
            # 预处理响应，移除markdown代码块格式
            if response.startswith('```json'):
                response = response[7:]  # 移除 ```json 前缀
            elif response.startswith('```'):
                response = response[3:]  # 移除 ``` 前缀
            if response.endswith('```'):
                response = response[:-3]  # 移除 ``` 后缀
            
            # 解析JSON响应
            result = json.loads(response.strip())
            return result
            
        except Exception as e:
            # 异常处理，返回None
            print(f"LLM检查冲突知识失败：{e}")
            return None
    
    def calculate_overall_health_score(self, missing_result, outdated_result, conflicting_result):
        """
        计算整体健康度评分
        
        功能：
            将三个维度的评分加权计算，得到综合健康度分数
        
        参数：
            missing_result: 缺少知识检查结果
            outdated_result: 过期知识检查结果
            conflicting_result: 冲突知识检查结果
        
        返回：
            综合健康度评分（0-1范围）
        """
        # 从各检查结果中提取维度评分，默认值为0
        coverage_score = missing_result.get('coverage_score', 0) if missing_result else 0
        freshness_score = outdated_result.get('freshness_score', 0) if outdated_result else 0
        consistency_score = conflicting_result.get('consistency_score', 0) if conflicting_result else 0
        
        # 加权计算综合评分
        # 覆盖率权重40%：知识覆盖是基础，权重最高
        # 新鲜度权重30%：信息时效性重要
        # 一致性权重30%：信息一致性重要
        overall_score = (
            coverage_score * 0.4 +      # 覆盖率权重40%
            freshness_score * 0.3 +     # 新鲜度权重30%
            consistency_score * 0.3     # 一致性权重30%
        )
        
        return overall_score
    
    def generate_health_report(self, knowledge_base, test_queries):
        """
        生成完整的健康度报告
        
        功能：
            执行三项检查，计算综合评分，生成完整报告
        
        参数：
            knowledge_base: 知识库列表
            test_queries: 测试查询列表
        
        返回：
            包含整体评分、健康等级、详细分析和建议的完整报告
        """
        print("正在检查知识库健康度...")
        
        # 步骤1：检查缺少的知识（完整性）
        print("1. 检查缺少的知识...")
        missing_result = self.check_missing_knowledge(knowledge_base, test_queries)
        
        # 步骤2：检查过期的知识（时效性）
        print("2. 检查过期的知识...")
        outdated_result = self.check_outdated_knowledge(knowledge_base)
        
        # 步骤3：检查冲突的知识（一致性）
        print("3. 检查冲突的知识...")
        conflicting_result = self.check_conflicting_knowledge(knowledge_base)
        
        # 步骤4：计算整体健康度评分
        overall_score = self.calculate_overall_health_score(missing_result, outdated_result, conflicting_result)
        
        # 步骤5：生成完整报告
        report = {
            "overall_health_score": overall_score,                      # 综合健康度评分
            "health_level": self.get_health_level(overall_score),       # 健康等级
            "missing_knowledge": missing_result,                        # 缺少知识详情
            "outdated_knowledge": outdated_result,                      # 过期知识详情
            "conflicting_knowledge": conflicting_result,                # 冲突知识详情
            "recommendations": self.generate_recommendations(           # 改进建议
                missing_result, outdated_result, conflicting_result
            ),
            "check_date": datetime.now().isoformat()                    # 检查时间
        }
        
        return report
    
    def get_health_level(self, score):
        """
        根据评分确定健康等级
        
        参数：
            score: 健康度评分（0-1范围）
        
        返回：
            健康等级字符串（优秀/良好/一般/需要改进）
        """
        if score >= 0.8:
            return "优秀"
        elif score >= 0.6:
            return "良好"
        elif score >= 0.4:
            return "一般"
        else:
            return "需要改进"
    
    def generate_recommendations(self, missing_result, outdated_result, conflicting_result):
        """
        生成改进建议
        
        功能：
            根据三项检查结果，生成具体的改进建议
        
        参数：
            missing_result: 缺少知识检查结果
            outdated_result: 过期知识检查结果
            conflicting_result: 冲突知识检查结果
        
        返回：
            改进建议列表
        """
        recommendations = []
        
        # 基于缺少知识的建议
        missing_count = len(missing_result.get('missing_knowledge', [])) if missing_result else 0
        if missing_count > 0:
            recommendations.append(f"补充{missing_count}个缺少的知识点，提高覆盖率")
        
        # 基于过期知识的建议
        outdated_count = len(outdated_result.get('outdated_knowledge', [])) if outdated_result else 0
        if outdated_count > 0:
            recommendations.append(f"更新{outdated_count}个过期知识点，确保信息时效性")
        
        # 基于冲突知识的建议
        conflicting_count = len(conflicting_result.get('conflicting_knowledge', [])) if conflicting_result else 0
        if conflicting_count > 0:
            recommendations.append(f"解决{conflicting_count}个知识冲突，提高一致性")
        
        # 如果没有问题，给出维护建议
        if not recommendations:
            recommendations.append("知识库状态良好，建议定期维护")
        
        return recommendations


# =============================================================================
# 主函数入口
# =============================================================================

def main():
    """
    主函数：演示知识库健康度检查系统的完整使用流程
    """
    # 初始化知识库健康度检查器
    checker = KnowledgeBaseHealthChecker()
    
    print("=== 知识库健康度检查示例（迪士尼主题乐园） ===\n")
    
    # -------------------------------------------------------------------------
    # 准备示例知识库（包含一些故意的问题用于演示）
    # -------------------------------------------------------------------------
    knowledge_base = [
        {
            "id": "kb_001",
            "content": "上海迪士尼乐园位于上海市浦东新区，是中国大陆首座迪士尼主题乐园，"
                      "于2016年6月16日开园。乐园占地面积390公顷，包含七大主题园区。",
            "last_updated": "2024-01-15"
        },
        {
            "id": "kb_002",
            "content": "上海迪士尼乐园的门票价格：平日成人票价为399元，周末和节假日为499元。"
                      "儿童票平日为299元，周末为374元。",
            "last_updated": "2023-12-01"  # 故意设置为较旧的时间，用于检测过期知识
        },
        {
            "id": "kb_003",
            "content": "上海迪士尼乐园门票价格：成人票平日350元，周末450元。"
                      "儿童票平日250元，周末350元。",
            "last_updated": "2024-02-01"  # 故意设置与kb_002冲突的价格
        },
        {
            "id": "kb_004",
            "content": "上海迪士尼乐园营业时间为上午8:00至晚上8:00，全年无休。",
            "last_updated": "2024-01-20"
        },
        {
            "id": "kb_005",
            "content": "从上海市区到迪士尼乐园可以乘坐地铁11号线到迪士尼站，"
                      "或乘坐迪士尼专线巴士。",
            "last_updated": "2024-01-10"
        }
    ]
    
    # -------------------------------------------------------------------------
    # 准备测试查询（包含一些知识库中没有的信息，用于检测知识缺失）
    # -------------------------------------------------------------------------
    test_queries = [
        {
            "query": "上海迪士尼乐园在哪里？",
            "expected_answer": "浦东新区"
        },
        {
            "query": "门票多少钱？",
            "expected_answer": "价格信息"
        },
        {
            "query": "营业时间是什么？",
            "expected_answer": "8:00-20:00"
        },
        {
            "query": "怎么去迪士尼？",
            "expected_answer": "地铁11号线"
        },
        {
            "query": "有什么特别活动？",  # 知识库中没有相关信息，用于检测知识缺失
            "expected_answer": "活动信息"
        },
        {
            "query": "停车费是多少？",  # 知识库中没有相关信息，用于检测知识缺失
            "expected_answer": "停车费信息"
        }
    ]
    
    # -------------------------------------------------------------------------
    # 生成健康度报告
    # -------------------------------------------------------------------------
    health_report = checker.generate_health_report(knowledge_base, test_queries)
    
    # -------------------------------------------------------------------------
    # 显示报告摘要
    # -------------------------------------------------------------------------
    print("=== 知识库健康度报告 ===\n")
    
    print(f"整体健康度评分：{health_report['overall_health_score']:.2f}")
    print(f"健康等级：{health_report['health_level']}")
    print(f"检查时间：{health_report['check_date']}")
    
    print("\n" + "=" * 60 + "\n")
    
    # -------------------------------------------------------------------------
    # 详细分析输出
    # -------------------------------------------------------------------------
    print("=== 详细分析 ===\n")
    
    # 1. 缺少的知识分析
    print("1. 缺少的知识分析:")
    missing = health_report['missing_knowledge']
    if missing:
        print(f"   覆盖率：{missing.get('coverage_score', 0)*100:.1f}%")
        print(f"   缺少知识点数量：{len(missing.get('missing_knowledge', []))}")
        for i, item in enumerate(missing.get('missing_knowledge', [])[:3], 1):
            print(f"   {i}. 查询：{item.get('query', '未知')}")
            print(f"      缺少方面：{item.get('missing_aspect', '未知')}")
            print(f"      重要性：{item.get('importance', '未知')}")
    
    print("\n" + "-" * 40 + "\n")
    
    # 2. 过期的知识分析
    print("2. 过期的知识分析:")
    outdated = health_report['outdated_knowledge']
    if outdated:
        print(f"   新鲜度评分：{outdated.get('freshness_score', 0):.2f}")
        print(f"   过期知识点数量：{len(outdated.get('outdated_knowledge', []))}")
        for i, item in enumerate(outdated.get('outdated_knowledge', [])[:3], 1):
            print(f"   {i}. 切片ID：{item.get('chunk_id', '未知')}")
            print(f"      过期方面：{item.get('outdated_aspect', '未知')}")
            print(f"      严重程度：{item.get('severity', '未知')}")
    
    print("\n" + "-" * 40 + "\n")
    
    # 3. 冲突的知识分析
    print("3. 冲突的知识分析:")
    conflicting = health_report['conflicting_knowledge']
    if conflicting:
        print(f"   一致性评分：{conflicting.get('consistency_score', 0):.2f}")
        print(f"   冲突数量：{len(conflicting.get('conflicting_knowledge', []))}")
        for i, item in enumerate(conflicting.get('conflicting_knowledge', [])[:3], 1):
            print(f"   {i}. 冲突类型：{item.get('conflict_type', '未知')}")
            print(f"      相关切片：{item.get('chunk_ids', [])}")
            print(f"      严重程度：{item.get('severity', '未知')}")
    
    print("\n" + "=" * 60 + "\n")
    
    # -------------------------------------------------------------------------
    # 输出改进建议
    # -------------------------------------------------------------------------
    print("=== 改进建议 ===\n")
    for i, recommendation in enumerate(health_report['recommendations'], 1):
        print(f"{i}. {recommendation}")


# =============================================================================
# 程序入口
# =============================================================================

if __name__ == "__main__":
    main()
