import time
from typing import List, Dict, Optional

"""
层次切片策略 (Hierarchical Chunking)
说明: 基于 Markdown 文档的结构层次（如标题、段落）进行智能切片，保持语义完整性。
"""

def hierarchical_chunking(text: str, target_size: int = 512, preserve_hierarchy: bool = True) -> List[str]:
    """
    执行层次切片：基于文档的 Markdown 结构层次进行切分。
    
    参数:
        text (str): 需要切片的原始文本内容。
        target_size (int): 每个切片的目标字符长度（默认为 512）。
        preserve_hierarchy (bool): 是否尝试保留标题层级结构（默认为 True）。
        
    返回:
        List[str]: 切分后的文本块列表。
    """
    # 存储最终生成的切片列表
    chunks = []
    
    # 定义 Markdown 的层次标记映射
    # key 为层级名称，value 为对应的起始标记列表
    hierarchy_markers: Dict[str, List[str]] = {
        'title1': ['# ', '标题1：', '一、', '1. '],  # 一级标题标记
        'title2': ['## ', '标题2：', '二、', '2. '], # 二级标题标记
        'title3': ['### ', '标题3：', '三、', '3. '], # 三级标题标记
        'paragraph': ['\n\n', '\n']               # 段落标记
    }
    
    # 将输入文本按行分割，便于逐行处理
    lines = text.split('\n')
    
    # 初始化当前正在构建的切片内容
    current_chunk = ""
    # 初始化当前切片的层级路径（暂时未使用，可用于扩展元数据）
    current_hierarchy = []
    
    # 遍历文本的每一行进行处理
    for line in lines:
        # 去除行首尾的空白字符
        line = line.strip()
        
        # 如果是空行，则跳过不处理
        if not line:
            continue
        
        # --- 步骤 1: 检测当前行的层次级别 ---
        line_level = None
        # 遍历定义的层级标记
        for level, markers in hierarchy_markers.items():
            for marker in markers:
                # 检查当前行是否以某个标记开头
                if line.startswith(marker):
                    line_level = level
                    break
            if line_level:
                break
        
        # 如果没有检测到特定的层次标记，默认为普通段落
        if not line_level:
            line_level = 'paragraph'
        
        # --- 步骤 2: 判断是否需要开始新的切片 ---
        should_start_new_chunk = False
        
        # 规则 1: 如果遇到更高级别的标题（如一级或二级标题），强制开始新切片
        # 这样可以保证大的章节内容独立
        if preserve_hierarchy and line_level in ['title1', 'title2']:
            should_start_new_chunk = True
        
        # 规则 2: 如果当前切片加上新的一行超过了目标大小，且当前切片不为空
        if len(current_chunk) + len(line) > target_size and current_chunk.strip():
            should_start_new_chunk = True
        
        # 规则 3: 如果当前行是段落，且当前切片长度已经接近目标大小（80%），提前截断
        # 避免切片过长
        if line_level == 'paragraph' and len(current_chunk) > target_size * 0.8:
            should_start_new_chunk = True
        
        # --- 步骤 3: 执行切片分割 ---
        if should_start_new_chunk and current_chunk.strip():
            # 将当前积累的文本作为一个完整的切片保存
            chunks.append(current_chunk.strip())
            # 重置当前切片缓冲区
            current_chunk = ""
            current_hierarchy = []
        
        # --- 步骤 4: 将当前行添加到当前切片 ---
        if current_chunk:
            # 如果当前切片已有内容，添加换行符连接
            current_chunk += "\n" + line
        else:
            # 如果是切片的第一行，直接赋值
            current_chunk = line
        
        # 更新层级信息（非段落内容记录层级）
        if line_level != 'paragraph':
            current_hierarchy.append(line_level)
    
    # --- 步骤 5: 处理循环结束后剩余的文本 ---
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
    print(f"   - 长度方差: {max_length - min_length} 字符") # 这里的方差实际上是极差
    
    print(f"\n 切片内容:")
    for i, chunk in enumerate(chunks, 1):
        print(f"   块 {i} ({len(chunk)} 字符):")
        print(f"   {chunk}")
        print()

# --- 测试数据 ---
# 包含丰富层次结构的 Markdown 文本
text = """
# 迪士尼乐园门票指南

## 一、门票类型介绍

### 1. 基础门票类型
迪士尼乐园提供多种门票类型以满足不同游客需求。一日票是最基础的门票类型，可在购买时选定日期使用，价格根据季节浮动。两日票需要连续两天使用，总价比购买两天单日票优惠约9折。特定日票包含部分节庆活动时段，需注意门票标注的有效期限。

### 2. 特殊门票类型
年票适合经常游玩的游客，提供更多优惠和特权。VIP门票包含快速通道服务，可减少排队时间。团体票适用于10人以上团队，享受团体折扣。

## 二、购票渠道与流程

### 1. 官方购票渠道
购票渠道以官方渠道为主，包括上海迪士尼官网、官方App、微信公众号及小程序。这些渠道提供最可靠的服务和最新的票务信息。

### 2. 第三方平台
第三方平台如飞猪、携程等合作代理商也可购票，但需认准官方授权标识。建议优先选择官方渠道以确保购票安全。

### 3. 证件要求
所有电子票需绑定身份证件，港澳台居民可用通行证，外籍游客用护照，儿童票需提供出生证明或户口本复印件。

## 三、入园须知

### 1. 入园时间
乐园通常在上午8:00开园，晚上8:00闭园，具体时间可能因季节和特殊活动调整。建议提前30分钟到达园区。

### 2. 安全检查
入园前需要进行安全检查，禁止携带危险物品、玻璃制品等。建议轻装简行，提高入园效率。

### 3. 园区服务
园区内提供寄存服务、轮椅租赁、婴儿车租赁等服务，可在游客服务中心咨询详情。

生日福利需在官方渠道登记，可获赠生日徽章和甜品券。半年内有效结婚证持有者可购买特别套票，含皇家宴会厅双人餐。军人优惠现役及退役军人凭证件享8折，需至少提前3天登记审批。
"""

if __name__ == "__main__":
    print(" 层次切片策略测试")
    print(f" 测试文本长度: {len(text)} 字符")
    
    # 执行层次切片，设置目标大小为 300 字符
    chunks = hierarchical_chunking(text, target_size=300, preserve_hierarchy=True)
    
    # 打印分析结果
    print_chunk_analysis(chunks, "层次切片")
