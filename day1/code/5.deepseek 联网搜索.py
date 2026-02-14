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
        model='deepseek-r1',
        messages=messages,
        result_format='message',
        enable_search=True, # 开启联网搜索
        search_domain="websearch", # 网页搜索
    )
    return response

# input = '你是哪家公司的大语言模型？'
input = '现在是北京时间哪一年几月几号几点？'
messages=[
    {'role': 'system', 'content': '你是一个乐于助人的智能小助手'},
    {'role': 'user', 'content': input}
]

response = get_response(messages)
print(response.output.choices[0].message.content)
