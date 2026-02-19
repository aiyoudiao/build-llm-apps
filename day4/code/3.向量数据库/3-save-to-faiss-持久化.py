from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
import faiss
from openai import OpenAI
import json
import pickle

# 从环境变量中获取 API Key
api_key = os.getenv('DASHSCOPE_API_KEY') 

# 定义向量维度，text-embedding-v4 模型支持 1024 维
dimension = 1024

try:
    # 初始化 OpenAI 客户端，配置阿里云 DashScope 的 base_url
    client = OpenAI(
        api_key=api_key,
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    )
except Exception as e:
    print("OpenAI 客户端初始化失败，检查一下环境变量 DASHSCOPE_API_KEY 是否存在")
    print(e)
    exit()

# 准备文档数据
documents = [
    {
        "id": "doc1",
        "text": "对于从美国直邮的电子产品，如发现质量问题，请在收货后30天内联系客服申请退货，需保留原包装。",
        "metadata": {"source": "us_shipping_policy.pdf", "category": "退货政策", "author": "CustomerService"}
    },
    {
        "id": "doc2",
        "text": "购买海外版iPhone的用户，虽然享有全球联保，但部分地区可能不提供免费维修服务，退货需自行承担国际运费。",
        "metadata": {"source": "warranty_terms.docx", "category": "售后服务", "author": "LegalDept"}
    },
    {
        "id": "doc3",
        "text": "关于欧洲时尚品牌的服饰退货，若非质量问题（如尺码不合），请在14天内寄回，商品吊牌必须完好无损。",
        "metadata": {"source": "eu_fashion_return.html", "category": "退货政策", "author": "E-commerceTeam"}
    },
    {
        "id": "doc4",
        "text": "由于黑色星期五促销期间订单量激增，所有跨境物流可能会延迟5-7个工作日，此时申请取消订单可能需要更长处理时间。",
        "metadata": {"source": "holiday_notice.txt", "category": "物流公告", "author": "OpsDept"}
    },
    {
        "id": "doc5",
        "text": "所有跨境电商商品在入境时需缴纳进口税，若产生关税，将在订单结算时预收或由物流公司代缴。",
        "metadata": {"source": "tax_policy.pdf", "category": "购买须知", "author": "FinanceDept"}
    },
    {
        "id": "doc6",
        "text": "根据海关规定，个人年度跨境消费限额为26000元，单笔订单超过5000元可能无法清关，请分拆下单。",
        "metadata": {"source": "customs_regulations.html", "category": "购买须知", "author": "ComplianceDept"}
    }
]

# 用于存储文档元数据的列表
metadata_store = []
# 用于存储生成的向量列表
vectors_list = []
# 用于存储向量对应的 ID 列表
vector_ids = []

print("正在为文档生成向量...")
for i, doc in enumerate(documents):
    try:
        # 调用 embedding 接口生成向量
        completion = client.embeddings.create(
            model='text-embedding-v4',
            input=doc["text"],
            dimensions=dimension,
            encoding_format='float',
        )
        vector = completion.data[0].embedding
        vectors_list.append(vector)
        vector_ids.append(i)
        metadata_store.append(doc)
        print(f"文档 {i+1}/{len(documents)} 已处理")
    except Exception as e:
        print(f"文档 {i+1}/{len(documents)} 处理失败: {e}")
        continue

# 转换为 numpy 数组
vectors_np = np.array(vectors_list).astype('float32')
vector_ids_np = np.array(vector_ids)

# 创建 FAISS 索引
index_flat_l2 = faiss.IndexFlatL2(dimension)
index = faiss.IndexIDMap(index_flat_l2)
index.add_with_ids(vectors_np, vector_ids_np)

print(f"\nFAISS 索引已创建，共包含 {index.ntotal} 个向量。")

# 确保保存目录存在
save_dir = "./faiss_store"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 保存 FAISS 索引到文件
index_file = os.path.join(save_dir, "faiss_index.bin")
faiss.write_index(index, index_file)
print(f"索引已保存到 {index_file}")

# 保存元数据到文件（使用 pickle）
metadata_file = os.path.join(save_dir, "metadata.pkl")
with open(metadata_file, "wb") as f:
    pickle.dump(metadata_store, f)
print(f"元数据已保存到 {metadata_file}")
