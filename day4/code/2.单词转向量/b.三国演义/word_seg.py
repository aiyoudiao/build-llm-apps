# 西游记单词分词

import jieba
import sys
import os

# 将上级目录添加到 sys.path 中，以便导入 utils 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import files_processing

# 定义源文件目录
source_folder = './source'
# 定义分词结果输出目录
segment_folder = './segment'

def segment_lines(file_list, segment_out_dir, stop_words=[]):
    """
    对文件列表中的文件进行分词，并将结果保存到指定目录
    :param file_list: 待处理的文件路径列表
    :param segment_out_dir: 分词结果输出目录
    :param stop_words: 停用词列表，默认为空
    """
    # 如果输出目录不存在，则创建该目录
    if not os.path.exists(segment_out_dir):
        os.makedirs(segment_out_dir)

    for i, file in enumerate(file_list):
        # 构造输出文件名，格式为 segment_0.txt, segment_1.txt ...
        segment_out_name = os.path.join(segment_out_dir, 'segment_{}.txt'.format(i))
        
        # 如果分词文件已存在，则跳过处理
        if os.path.exists(segment_out_name):
            continue

        # 流式读写文件
        with open(file, 'rb') as f:
            document = f.read()
            # 使用 jieba 进行分词
            document_cut = jieba.cut(document)
            sentence_segment = []
            # 过滤停用词
            for word in document_cut:
                if word not in stop_words:
                    sentence_segment.append(word)
            # 将分词结果用空格连接
            result = ' '.join(sentence_segment)
            # 编码为 utf-8
            result = result.encode('utf-8')
            # 写入输出文件
            with open (segment_out_name, 'wb') as f2:
                f2.write(result)

# 获取源文件夹下的所有 .txt 文件列表
# files_processing.get_files_list 是 utils 包中的辅助函数
file_list = files_processing.get_files_list(source_folder, postfix='*.txt')

# 执行分词操作，输出到 segment 文件夹下
segment_lines(file_list, segment_folder)
