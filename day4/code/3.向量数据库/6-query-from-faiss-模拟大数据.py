from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
import faiss
import pickle
import time

# 定义向量维度
dimension = 1024
top_k = 5

# 索引文件路径
save_dir = "./faiss_store_1b"
index_file = os.path.join(save_dir, "faiss_index_ivf_pq.bin")
metadata_file = os.path.join(save_dir, "metadata_sample.pkl")

# 检查文件是否存在
if not os.path.exists(index_file):
    print(f"索引文件 {index_file} 不存在，请先运行 5-save-to-faiss-1b.py 生成索引")
    exit()

print(f"正在加载索引文件 {index_file} ...")
# 在 10 亿级场景下，索引文件可能很大（几十GB），建议使用内存映射（mmap）方式加载
# faiss.IO_FLAG_MMAP 使得操作系统按需加载页面，极大降低内存占用并加速启动
start_time = time.time()
index = faiss.read_index(index_file, faiss.IO_FLAG_MMAP)
print(f"索引加载完成，耗时 {time.time() - start_time:.2f} 秒")
print(f"索引中包含 {index.ntotal} 个向量")

# 加载部分元数据用于演示（真实场景下这部分是从数据库查询）
if os.path.exists(metadata_file):
    with open(metadata_file, "rb") as f:
        metadata_sample = pickle.load(f)
    print(f"元数据示例加载成功，包含 {len(metadata_sample)} 条记录")
else:
    print("元数据文件不存在")
    metadata_sample = {}

# 模拟一个查询向量 (float32)
# 真实场景下，你需要调用 embedding API 生成查询向量
query_vector = np.random.random((1, dimension)).astype('float32')
print("\n正在执行查询...")

start_time = time.time()
# IVF-PQ 索引的查询非常快，即使是 10 亿数据也能在毫秒级完成
# nprobe 参数决定了我们要搜索多少个倒排桶（Cluster）
# 默认 nprobe=1，增加 nprobe 可以提高准确率，但会稍微降低速度
index.nprobe = 10  
print(f"设置 nprobe = {index.nprobe} (搜索桶数量)")

distances, retrieved_ids = index.search(query_vector, top_k)
print(f"查询完成，耗时 {(time.time() - start_time) * 1000:.2f} 毫秒")

print('\n --- 查询结果 ---\n')
for i in range(top_k):
    doc_id = retrieved_ids[0][i]
    distance = distances[0][i]

    if doc_id == -1:
        print(f"排名 {i+1}: 未找到结果")
        continue

    # 尝试从元数据中获取信息
    # 注意：我们的元数据示例只存了前100个ID，如果你查到了后面的ID，可能就没有元数据了
    meta_info = metadata_sample.get(doc_id, "Metadata not found (in sample)")

    print(f"排名 {i+1} (相似度得分/距离：{distance:.4f})")
    print(f"ID: {doc_id}")
    print(f"元数据：{meta_info}")
    print("-" * 30)
