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

input = '现在是北京时间哪一年几月几号几点？'
completion = client.chat.completions.create(
    model='qwen-turbo',
    messages=[
        {'role': 'system', 'content': '你是一个乐于助人的智能小助手'},
        {'role': 'user', 'content': input}
    ],
    extra_body={
        'result_format': 'message',
        "enable_search": True, # 开启联网搜索
        "search_domain": "websearch", # 网页搜索
    }
)

print(completion.choices[0].message.content)
