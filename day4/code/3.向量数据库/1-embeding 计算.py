from dotenv import load_dotenv
load_dotenv()

import json
import os

from openai import OpenAI

api_key = os.getenv('DASHSCOPE_API_KEY') # 从 .env 文件中获取API密钥
# api_key = os.environ.get('DASHSCOPE_API_KEY') # 从环境变量中获取API密钥

client = OpenAI(
    api_key=api_key,
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', # https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions 换成 dashscope 服务的base_url
)

completion = client.embeddings.create(
    model='text-embedding-v4',
    input="我想知道商品的退货政策", # 内容
    dimensions=1024, # 纬度
    encoding_format='float', # 浮点数格式
)

print(completion.model_dump_json()) # 向量结果

