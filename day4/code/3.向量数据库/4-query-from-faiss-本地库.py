from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
import faiss
from openai import OpenAI
import pickle

# 从环境变量中获取 API Key
api_key = os.getenv('DASHSCOPE_API_KEY')

# 定义向量维度
dimension = 1024

try:
    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    )
except Exception as e:
    print("OpenAI 客户端初始化失败，检查一下环境变量 DASHSCOPE_API_KEY 是否存在")
    print(e)
    exit()

# 确保保存目录存在
save_dir = "./faiss_store"
if not os.path.exists(save_dir):
    print(f"保存目录 {save_dir} 不存在，请先运行 3-save-to-faiss.py 生成索引")
    exit()

# 加载 FAISS 索引
index_file = os.path.join(save_dir, "faiss_index.bin")
if not os.path.exists(index_file):
    print(f"索引文件 {index_file} 不存在，请先运行 3-save-to-faiss.py 生成索引")
    exit()

index = faiss.read_index(index_file)
print(f"索引加载成功，包含 {index.ntotal} 个向量")


# 加载元数据
metadata_file = os.path.join(save_dir, "metadata.pkl")
if not os.path.exists(metadata_file):
    print(f"元数据文件 {metadata_file} 不存在，请先运行 3-save-to-faiss.py 生成索引")
    exit()

with open(metadata_file, "rb") as f:
    metadata_store = pickle.load(f)
print(f"元数据加载成功，包含 {len(metadata_store)} 条记录")

# 定义查询函数
def search(query_text, top_k=3):
    print(f"\n正在为查询文本「 {query_text} 」生成向量...")
    try:
        # 为查询文本生成向量
        query_completion = client.embeddings.create(
            model='text-embedding-v4',
            input=query_text,
            dimensions=dimension,
            encoding_format='float'
        )
        query_vector = np.array([query_completion.data[0].embedding]).astype('float32')

        # 在索引中搜索最相似的 top_k 个向量
        distances, retrieved_ids = index.search(query_vector, top_k)

        print('\n --- 查询结果 ---\n')
        for i in range(top_k):
            doc_id = retrieved_ids[0][i]
            if doc_id == -1:
                print(f"排名 {i+1}: 未找到更多结果")
                continue

            retrieved_doc = metadata_store[doc_id]
            print(f"排名 {i+1} (相似度得分/距离：{distances[0][i]:.4f})")
            print(f"ID: {doc_id}")
            print(f"文本：{retrieved_doc['text']}")
            print(f"元数据：{retrieved_doc['metadata']}")
            print("-" * 30)

    except Exception as e:
        print(f"查询出错: {e}")

# 执行查询示例
if __name__ == "__main__":
    query_text = "我想知道买东西要交多少税"
    search(query_text)
