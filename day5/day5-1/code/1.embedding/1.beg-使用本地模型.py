import os
import torch
from modelscope import snapshot_download

# BGE-M3 模型的 modelscope 地址
model_name = "BAAI/bge-m3"
# 模型下载保存的目录
cache_dir = "../../../../../models"
# 模型的本地路径
model_dir = os.path.join(cache_dir, model_name)

# 判断目录下是否包含该模型，如果不包含则下载
if not os.path.exists(model_dir):
    print(f"开始下载模型 {model_name} 到 {cache_dir}")
    # 使用 modelscope 下载模型
    snapshot_download(model_name, cache_dir=cache_dir)
    print(f"{model_name} 下载完成")
else:
    print(f"模型 {model_name} 已存在于 {model_dir}")
    print("请勿重复下载")

# 选择设备：优先使用 CUDA，其次是 MPS (Mac)，最后是 CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"使用设备 {device}")

# 加载 BGE-M3 模型
from FlagEmbedding import BGEM3FlagModel

# use_fp16=True 表示使用半精度浮点数，可以节省显存并加速推理
print("正在加载模型...")
model = BGEM3FlagModel(model_dir, use_fp16=True, device=device)
print("模型加载完成")

# 定义待编码的句子列表
sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

# 生成句子的 dense embedding (稠密向量)
# BGEM3FlagModel.encode 方法返回一个字典，包含 'dense_vecs', 'lexical_weights', 'colbert_vecs' 等
print("正在生成向量...")
embeddings_1 = model.encode(sentences_1, batch_size=12, max_length=8192)
embeddings_2 = model.encode(sentences_2)

# 获取稠密向量
dense_vec_1 = embeddings_1['dense_vecs']
dense_vec_2 = embeddings_2['dense_vecs']

print(f"句子1的向量形状: {dense_vec_1.shape}")
print(f"句子2的向量形状: {dense_vec_2.shape}")

# 计算相似度
# 使用 @ 运算符进行矩阵乘法计算点积，因为向量已经归一化，所以点积等于余弦相似度
similarity = dense_vec_1 @ dense_vec_2.T
print("相似度矩阵:")
print(similarity)

# 打印具体的相似度分数
print(f"'{sentences_1[0]}' 和 '{sentences_2[0]}' 的相似度: {similarity[0][0]:.4f}")
print(f"'{sentences_1[0]}' 和 '{sentences_2[1]}' 的相似度: {similarity[0][1]:.4f}")
print(f"'{sentences_1[1]}' 和 '{sentences_2[0]}' 的相似度: {similarity[1][0]:.4f}")
print(f"'{sentences_1[1]}' 和 '{sentences_2[1]}' 的相似度: {similarity[1][1]:.4f}")
