from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
import faiss
from openai import OpenAI

# 从环境变量中获取 API Key
api_key = os.getenv('DASHSCOPE_API_KEY') 
# api_key = os.environ.get('DASHSCOPE_API_KEY') # 另一种获取方式

# 定义向量维度，text-embedding-v4 模型支持 1024 维
dimension = 1024

try:
    # 初始化 OpenAI 客户端，配置阿里云 DashScope 的 base_url
    client = OpenAI(
        api_key=api_key,
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', # DashScope 兼容 OpenAI 接口的地址
    )
except Exception as e:
    print("OpenAI 客户端初始化失败，检查一下环境变量 DASHSCOPE_API_KEY 是否存在")
    print(e)
    exit()

# 准备示例文档数据，包含文本内容和元数据
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

# 用于存储文档元数据的列表，索引与向量索引一一对应
metadata_store = []
# 用于存储生成的向量列表
vectors_list = []
# 用于存储向量对应的 ID 列表
vector_ids = []

print("正在为文档生成向量")
# 遍历文档列表，为每个文档生成嵌入向量
for i, doc in enumerate(documents):
    try:
        # 调用 embedding 接口生成向量
        completion = client.embeddings.create(
            model='text-embedding-v4',
            input=doc["text"], # 输入文档文本
            dimensions=dimension, # 指定向量维度
            encoding_format='float', # 返回浮点数格式
        )
        # 获取生成的向量
        vector = completion.data[0].embedding
        vectors_list.append(vector)

        # 记录 ID 和元数据
        vector_ids.append(i)
        metadata_store.append(doc)

        print(f"文档 {i+1}/{len(documents)} 已处理")
    except Exception as e:
        print(f"文档 {i+1}/{len(documents)} 处理失败")
        print(e)
        continue

# 将向量列表转换为 numpy 数组，FAISS 需要 float32 类型
vectors_np = np.array(vectors_list).astype('float32')
# 将 ID 列表转换为 numpy 数组
vector_ids_np = np.array(vector_ids)


top = 3 # 设定检索结果返回前 3 个

# 创建 FAISS 索引，IndexFlatL2 使用欧氏距离进行暴力搜索
index_flat_l2 = faiss.IndexFlatL2(dimension)

# 使用 IndexIDMap 包装索引，使其支持自定义 ID
index = faiss.IndexIDMap(index_flat_l2)

# 将向量和对应的 ID 添加到索引中
index.add_with_ids(vectors_np, vector_ids_np)

print(f"\nFAISS 索引已成功创建，共包含 {index.ntotal} 个向量。")

# 定义查询文本
query_text = '我想了解一下海外商品的退货政策'
print(f"\n正在为查询文本「 {query_text} 」生成向量")

try:
    # 为查询文本生成向量
    query_completion = client.embeddings.create(
        model = 'text-embedding-v4',
        input=query_text,
        dimensions=1024,
        encoding_format='float'
    )
    # 将查询向量转换为 numpy 数组
    query_vector = np.array([query_completion.data[0].embedding]).astype('float32')

    # 在索引中搜索最相似的 top 个向量，返回距离和对应的 ID
    distances, retrieved_ids = index.search(query_vector, top)

    print('\n --- 查询结果 ---\n')

    # 遍历搜索结果
    for i in range(top):
        doc_id = retrieved_ids[0][i]
        # print(f"doc_id {doc_id}")

        # FAISS 如果没找到足够的结果会返回 -1
        if doc_id == -1:
            print(f"\n排名 {i+1}: 未找到更多结果")
            continue

        # 根据 ID 从 metadata_store 中获取对应的文档信息
        retrieved_doc = metadata_store[doc_id]

        print(f"\n--- 排名 {i+1} (相似度得分/距离：{distances[0][i]:.4f}) ---")
        print(f"ID: {doc_id}")
        print(f"原始文本：{retrieved_doc['text']}")
        # 打印文档的元数据
        print(f"元数据：{retrieved_doc['metadata']}")
except Exception as e:
    print(f"查询操作发生错误：{e}")
