from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import re
from typing import List, Optional, Any

"""
LLM 语义切片策略 (LLM Semantic Chunking)
说明: 利用大语言模型（LLM）的深度理解能力，智能地将文本按照语义完整性进行切分。
      这种方法通常能产生比基于规则（如标点符号）更高质量的切片，但成本较高且速度较慢。
"""

# 加载环境变量，主要用于获取 API Key
load_dotenv()

def advanced_semantic_chunking_with_llm(text: str, max_chunk_size: int = 512) -> List[str]:
    """
    使用 LLM 进行高级语义切片。
    
    参数:
        text (str): 需要切片的原始文本内容。
        max_chunk_size (int): 每个切片的目标最大字符长度（默认为 512）。
    
    返回:
        List[str]: 切分后的文本块列表。如果调用失败，返回空列表。
    """
    # 检查环境变量中是否存在 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告: 未找到 DASHSCOPE_API_KEY 环境变量，无法调用 LLM。")
        return []
    
    # 初始化 OpenAI 客户端 (使用阿里云 DashScope 的兼容接口)
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # 构建 Prompt (提示词)
    # 明确要求模型按照语义完整性切分，并返回 JSON 格式
    prompt = f"""
请将以下文本按照语义完整性进行切片，每个切片不超过{max_chunk_size}字符。
要求：
1. 保持语义完整性，确保每个切片是一个独立的语义单元。
2. 在自然的分割点（如段落结束、话题转换处）切分。
3. 返回 JSON 格式的切片列表，不要包含 Markdown 标记，格式如下：
{{
  "chunks": [
    "第一个切片内容",
    "第二个切片内容",
    ...
  ]
}}

文本内容：
{text}
"""
    
    try:
        print("正在调用 LLM 进行语义切片...")
        # 调用 Chat Completion 接口
        response = client.chat.completions.create(
            model="qwen-turbo-latest", # 使用通义千问模型
            messages=[
                {"role": "system", "content": "你是一个专业的文本切片助手。请严格按照 JSON 格式返回结果，不要添加任何额外的解释或 Markdown 标记。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1 # 降低随机性，保证输出格式稳定
        )
        
        # 获取模型返回的内容
        result = response.choices[0].message.content
        print(f"LLM 返回结果预览: {result[:200]}...")
        
        # --- 结果清洗与解析 ---
        cleaned_result = result.strip()
        
        # 移除可能存在的 Markdown 代码块标记 (```json ... ```)
        if cleaned_result.startswith('```'):
            cleaned_result = re.sub(r'^```(?:json)?\s*', '', cleaned_result)
        if cleaned_result.endswith('```'):
            cleaned_result = re.sub(r'\s*```$', '', cleaned_result)
        
        # 尝试解析 JSON
        try:
            chunks_data = json.loads(cleaned_result)
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试从文本中提取 JSON 对象部分
            print("JSON 直接解析失败，尝试提取 JSON 片段...")
            json_match = re.search(r'\{.*\}', cleaned_result, re.DOTALL)
            if json_match:
                chunks_data = json.loads(json_match.group())
            else:
                raise ValueError("无法从模型输出中提取有效的 JSON 数据")

        # --- 处理不同的返回结构 ---
        # 优先查找 "chunks" 字段
        if isinstance(chunks_data, dict):
            if "chunks" in chunks_data and isinstance(chunks_data["chunks"], list):
                return chunks_data["chunks"]
            elif "slice" in chunks_data:
                # 兼容可能返回 "slice" 字段的情况
                if isinstance(chunks_data["slice"], list):
                    return chunks_data["slice"]
                else:
                    return [str(chunks_data["slice"])]
            else:
                print(f"警告: JSON 结构不符合预期，可用键: {list(chunks_data.keys())}")
                return []
        elif isinstance(chunks_data, list):
            # 如果直接返回了列表
            return chunks_data
        else:
            print(f"警告: 意外的返回数据类型: {type(chunks_data)}")
            return []
        
    except Exception as e:
        print(f"LLM 切片过程发生错误: {str(e)}")
        # 出错时返回空列表，避免程序崩溃
        return []

def test_chunking_methods():
    """
    测试 LLM 切片方法的主函数。
    """
    # 示例文本
    text = """
迪士尼乐园提供多种门票类型以满足不同游客需求。一日票是最基础的门票类型，可在购买时选定日期使用，价格根据季节浮动。两日票需要连续两天使用，总价比购买两天单日票优惠约9折。特定日票包含部分节庆活动时段，需注意门票标注的有效期限。

购票渠道以官方渠道为主，包括上海迪士尼官网、官方App、微信公众号及小程序。第三方平台如飞猪、携程等合作代理商也可购票，但需认准官方授权标识。所有电子票需绑定身份证件，港澳台居民可用通行证，外籍游客用护照，儿童票需提供出生证明或户口本复印件。

生日福利需在官方渠道登记，可获赠生日徽章和甜品券。半年内有效结婚证持有者可购买特别套票，含皇家宴会厅双人餐。军人优惠现役及退役军人凭证件享8折，需至少提前3天登记审批。
"""

    print("\n=== LLM 高级语义切片测试 ===")
    
    # 调用切片函数
    chunks = advanced_semantic_chunking_with_llm(text, max_chunk_size=300)
    
    if chunks:
        print(f"LLM 高级语义切片生成 {len(chunks)} 个切片:")
        for i, chunk in enumerate(chunks):
            print(f"LLM 语义块 {i+1} (长度: {len(chunk)}):")
            print(f"{chunk}")
            print("-" * 30)
    else:
        print("未能生成切片。")

if __name__ == "__main__":
    test_chunking_methods()
