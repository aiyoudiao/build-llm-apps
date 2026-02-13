from dotenv import load_dotenv
load_dotenv()

import json
import os
import dashscope
from dashscope.api_entities.dashscope_response import Role
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY') # 从 .env 文件中获取API密钥
# dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY') # 从环境变量中获取API密钥

def get_response(messages):
    response = dashscope.Generation.call(
        model='deepseek-v3',
        messages=messages,
        result_format='message'
    )
    return response

review = "这个商品真的很差，我非常不喜欢它。"
messages=[
    {'role': 'system', 'content': '你是一个情感分析大师，你的任务是分析用户的正负向情感，请回复用户的情感是 正向 还是 负向'},
    {'role': 'user', 'content': review}
]

response = get_response(messages)
print(response.output.choices[0].message.content)
