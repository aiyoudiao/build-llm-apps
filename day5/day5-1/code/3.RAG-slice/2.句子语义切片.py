import re
from typing import List

"""
语义切片策略 (Semantic Chunking)
说明: 基于自然语言的句子边界进行切分，确保每个切片包含完整的句子，保持语义的完整性。
      通过正则表达式识别常见的句子结束符（如句号、问号、感叹号等）。
"""

def semantic_chunking(text: str, max_chunk_size: int = 512) -> List[str]:
    """
    基于语义的切片 - 按句子分割：
    1. 使用正则表达式识别句子结束符，将文本分割成句子列表。
    2. 逐个累加句子，直到接近最大切片长度。
    3. 这种方法避免了在句子中间强行切断，比固定长度切片更符合人类阅读习惯。
    
    参数:
        text (str): 需要切片的原始文本内容。
        max_chunk_size (int): 每个切片的最大字符长度（默认为 512）。
    
    返回:
        List[str]: 切分后的文本块列表。
    """
    # 使用正则表达式分割句子
    # 正则解释:
    # ([.!?。！？\n]+)
    # [...] 匹配字符集中的任意一个字符（中英文句号、问号、感叹号、换行符）
    # + 表示匹配一次或多次
    # () 将匹配到的标点符号作为一个分组保留下来，这样 split 不会丢弃标点
    sentences_and_punctuations = re.split(r'([.!?。！？\n]+)', text)
    
    # 重新组合句子和标点
    sentences = []
    current_sentence = ""
    
    for item in sentences_and_punctuations:
        # 累加当前部分
        current_sentence += item
        # 如果当前部分包含结束符，说明一个完整的句子结束了
        # 这里通过检查 item 是否只包含标点符号来判断（简单判断）
        if re.match(r'^[.!?。！？\n]+$', item):
             sentences.append(current_sentence)
             current_sentence = ""
    
    # 如果最后还有剩余内容（可能没有标点结尾），加入列表
    if current_sentence:
        sentences.append(current_sentence)

    # 开始构建切片
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # 去除首尾空白（可选，视需求而定，这里保留原始格式稍微处理一下）
        # sentence = sentence.strip() 
        # 如果句子本身就是空的（比如连续的换行符导致的），跳过
        if not sentence.strip():
            continue
            
        # 检查：如果当前切片加上新句子超过最大长度，且当前切片不为空
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            # 保存当前切片
            chunks.append(current_chunk.strip())
            # 开始新的切片，以当前句子开头
            current_chunk = sentence
        else:
            # 否则，将句子追加到当前切片
            # 如果当前切片不为空，添加空格分隔（对于中文其实不需要空格，但为了通用性）
            # 这里简单直接拼接，因为标点符号已经包含了分隔信息
            current_chunk += sentence
    
    # 添加最后一个切片
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def print_chunk_analysis(chunks: List[str], method_name: str) -> None:
    """
    打印切片结果的详细统计分析。
    
    参数:
        chunks (List[str]): 切片列表。
        method_name (str): 切片方法的名称，用于显示标题。
    """
    print(f"\n{'='*60}")
    print(f" {method_name}")
    print(f"{'='*60}")
    
    if not chunks:
        print(" 未生成任何切片")
        return
    
    # 计算统计指标
    total_length = sum(len(chunk) for chunk in chunks)
    avg_length = total_length / len(chunks)
    min_length = min(len(chunk) for chunk in chunks)
    max_length = max(len(chunk) for chunk in chunks)
    
    print(f" 统计信息:")
    print(f"   - 切片数量: {len(chunks)}")
    print(f"   - 平均长度: {avg_length:.1f} 字符")
    print(f"   - 最短长度: {min_length} 字符")
    print(f"   - 最长长度: {max_length} 字符")
    print(f"   - 长度方差: {max_length - min_length} 字符")
    
    print(f"\n 切片内容:")
    for i, chunk in enumerate(chunks, 1):
        print(f"   块 {i} ({len(chunk)} 字符):")
        print(f"   {chunk}")
        print()

# --- 测试数据 ---
# 示例文本
text = """
迪士尼乐园提供多种门票类型以满足不同游客需求。一日票是最基础的门票类型，可在购买时选定日期使用，价格根据季节浮动。两日票需要连续两天使用，总价比购买两天单日票优惠约9折。特定日票包含部分节庆活动时段，需注意门票标注的有效期限。

购票渠道以官方渠道为主，包括上海迪士尼官网、官方App、微信公众号及小程序。第三方平台如飞猪、携程等合作代理商也可购票，但需认准官方授权标识。所有电子票需绑定身份证件，港澳台居民可用通行证，外籍游客用护照，儿童票需提供出生证明或户口本复印件。

生日福利需在官方渠道登记，可获赠生日徽章和甜品券。半年内有效结婚证持有者可购买特别套票，含皇家宴会厅双人餐。军人优惠现役及退役军人凭证件享8折，需至少提前3天登记审批。
"""

if __name__ == "__main__":
    print(" 语义切片策略测试")
    print(f" 测试文本长度: {len(text)} 字符")
    
    # 执行语义切片
    chunks = semantic_chunking(text, max_chunk_size=300)
    
    # 打印分析结果
    print_chunk_analysis(chunks, "语义切片")
