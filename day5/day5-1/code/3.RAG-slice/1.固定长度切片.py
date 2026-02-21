from typing import List

"""
固定长度切片策略 (Fixed Length Chunking)
说明: 将文本按照固定的字符长度进行切分，同时尝试在切分点附近寻找句子边界，以减少语义截断。
      为了保持上下文连贯性，切片之间通常会设置一定的重叠区域。
"""

def improved_fixed_length_chunking(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    改进的固定长度切片：
    1. 按照固定长度切分。
    2. 尝试在切分点前回溯，寻找最近的句子结束符，避免切断句子。
    3. 支持切片重叠，保证上下文衔接。
    
    参数:
        text (str): 需要切片的原始文本内容。
        chunk_size (int): 每个切片的目标最大长度（默认为 512）。
        overlap (int): 切片之间的重叠长度（默认为 50）。
    
    返回:
        List[str]: 切分后的文本块列表。
    """
    # 存储最终生成的切片列表
    chunks = []
    
    # 切片的起始位置
    start = 0
    
    # 循环直到处理完所有文本
    while start < len(text):
        # 计算当前切片的理论结束位置
        end = start + chunk_size
        
        # 如果计算出的结束位置还在文本范围内，尝试优化切分点
        if end < len(text):
            # 在切分点前回溯最多 100 个字符，寻找句子结束符
            # range(start, stop, step): 从 end 开始，向前回溯
            # max(start, end - 100): 确保回溯不会超过当前切片的起始位置，且最多回溯 100 字符
            for i in range(end, max(start, end - 100), -1):
                # 检查字符是否为常见的句子结束符（中英文）
                if text[i] in '.!?。！？':
                    # 如果找到结束符，将切分点设置在结束符之后（包含结束符）
                    end = i + 1
                    break
        
        # 截取当前切片内容
        chunk = text[start:end]
        
        # 只有当切片内容不为空时才添加
        if len(chunk.strip()) > 0:
            chunks.append(chunk.strip())
        
        # 更新下一个切片的起始位置
        # 下一个起始位置 = 当前结束位置 - 重叠长度
        # 这样可以保证相邻切片之间有 overlap 长度的内容重叠
        start = end - overlap
    
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
    print(f"   - 长度方差: {max_length - min_length} 字符") # 这里的方差实际上是极差
    
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
    print(" 固定长度切片策略测试")
    print(f" 测试文本长度: {len(text)} 字符")
    
    # 执行改进的固定长度切片
    chunks = improved_fixed_length_chunking(text, chunk_size=300, overlap=50)
    
    # 打印分析结果
    print_chunk_analysis(chunks, "固定长度切片")
