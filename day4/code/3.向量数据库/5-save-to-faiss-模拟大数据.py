from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
import faiss
import pickle
import time

# 定义向量维度，text-embedding-v4 模型支持 1024 维
dimension = 1024

# 模拟 10 亿数据的场景（因为本机跑不动 10 亿，这里我们生成 10 万条随机向量演示流程）
# 真实场景下，你需要分批次读取数据，或者使用 faiss 的 add_with_ids 方法分批添加
total_vectors = 100000  # 演示用 10 万条，实际场景可能是 10^9
train_size = 10000      # 用于训练索引的样本数量，通常需要 10万~100万

print(f"正在生成 {total_vectors} 条随机向量用于模拟...")
# 生成随机向量作为示例数据 (float32)
data = np.random.random((total_vectors, dimension)).astype('float32')
# 生成对应的 ID
ids = np.arange(total_vectors)

# 定义索引类型：IVF + PQ (倒排索引 + 乘积量化)
# IVF1024: 将数据空间划分为 1024 个聚类中心（倒排桶）。对于 10 亿数据，建议设为 65536 或更大
# PQ64: 将 1024 维向量压缩为 64 字节（每个子向量 1024/64 = 16 维）。这能极大降低内存占用
# "IDMap": 让我们能自定义 ID（比如对应数据库的主键）
index_string = f"IVF1024,PQ64" 

print(f"创建索引结构: {index_string}")
# 使用 index_factory 工厂方法创建索引
index = faiss.index_factory(dimension, index_string)

# 包装一层 IndexIDMap 以支持自定义 ID (add_with_ids)
# 注意：对于某些复杂的组合索引，faiss 可能已经默认支持 id，或者需要特殊处理
# 这里我们用 IndexIDMap2，它比 IndexIDMap 更适合某些情况，但在 IVF+PQ 下通常不需要额外包装，
# 因为 IVF 索引本身不支持直接 add_with_ids，需要先 train。
# 但为了支持自定义 ID，我们通常使用 IndexIDMap 包裹一个 Flat 索引，或者在 IVF 索引中直接使用 add_with_ids (如果支持)。
# 对于 IVF 索引，faiss 默认的 add 方法会自动生成从 0 开始的 ID。
# 如果要用自定义 ID，可以使用 index.add_with_ids (前提是 index 类型支持)。
# 标准的 IVF+PQ 索引支持 add_with_ids。

print("正在训练索引 (Train)...")
start_time = time.time()
# 必须先训练！从数据中取出一部分（train_size）来训练聚类中心和量化码书
# 在 10 亿数据场景下，这一步通常需要几十分钟到几小时，且需要较大的内存
index.train(data[:train_size])
print(f"训练完成，耗时 {time.time() - start_time:.2f} 秒")

print("正在添加数据 (Add)...")
start_time = time.time()
# 添加所有数据
# 在真实 10 亿场景下，这里应该是一个循环，每次读取 10 万条数据，调用 index.add_with_ids
batch_size = 10000
for i in range(0, total_vectors, batch_size):
    batch_data = data[i : i + batch_size]
    batch_ids = ids[i : i + batch_size]
    index.add_with_ids(batch_data, batch_ids)
    print(f"已添加 {i + batch_size} / {total_vectors} 条数据")

print(f"添加完成，耗时 {time.time() - start_time:.2f} 秒")
print(f"索引中当前包含 {index.ntotal} 个向量")

# 确保保存目录存在
save_dir = "./faiss_store_1b"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 保存索引到文件
index_file = os.path.join(save_dir, "faiss_index_ivf_pq.bin")
print(f"正在保存索引到 {index_file} ...")
start_time = time.time()
faiss.write_index(index, index_file)
print(f"保存完成，耗时 {time.time() - start_time:.2f} 秒")

# 模拟保存一些元数据（真实场景下元数据可能存在 MySQL/MongoDB/HBase 中，而不是 pickle）
# 这里只存前 100 条作为示例
metadata_sample = {i: f"Document metadata for ID {i}" for i in range(100)}
metadata_file = os.path.join(save_dir, "metadata_sample.pkl")
with open(metadata_file, "wb") as f:
    pickle.dump(metadata_sample, f)
print(f"元数据示例已保存到 {metadata_file}")
