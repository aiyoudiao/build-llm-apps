from typing import List

"""
滑动窗口切片策略 (Sliding Window Chunking)
说明: 使用固定长度的窗口在文本上滑动，生成具有重叠区域的切片。
      这种方法可以保证上下文的连续性，避免关键信息被切断。
"""

def sliding_window_chunking(text: str, window_size: int = 128, step_size: int = 64) -> List[str]:
    """
    执行滑动窗口切片：以固定步长移动固定大小的窗口来截取文本。
    
    参数:
        text (str): 需要切片的原始文本内容。
        window_size (int): 窗口大小，即每个切片的最大长度（默认为 128）。
        step_size (int): 步长，即每次窗口移动的距离（默认为 64）。
                         (window_size - step_size) 即为重叠区域的大小。
    
    返回:
        List[str]: 切分后的文本块列表。
    """
    # 存储最终生成的切片列表
    chunks = []
    
    # 遍历文本，步长为 step_size
    # range(start, stop, step) 生成从 0 开始，每次增加 step_size 的索引序列
    for i in range(0, len(text), step_size):
        # 截取当前窗口范围内的文本
        # text[i : i + window_size] 获取从索引 i 开始，长度为 window_size 的子串
        # 如果 i + window_size 超过文本长度，Python 会自动处理，只截取到文本末尾
        chunk = text[i:i + window_size]
        
        # 过滤掉空白切片，确保切片包含有效内容
        if len(chunk.strip()) > 0:
            chunks.append(chunk.strip())
    
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
    print(" 滑动窗口切片策略测试")
    print(f" 测试文本长度: {len(text)} 字符")
    
    # 执行滑动窗口切片
    # 窗口大小 128，步长 64，意味着有 50% 的重叠
    chunks = sliding_window_chunking(text, window_size=128, step_size=64)
    
    # 打印分析结果
    print_chunk_analysis(chunks, "滑动窗口切片")
