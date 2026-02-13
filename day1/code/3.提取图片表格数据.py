from dotenv import load_dotenv
load_dotenv()

import json
import os
import dashscope
from dashscope.api_entities.dashscope_response import Role
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY') # 从 .env 文件中获取API密钥
# dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY') # 从环境变量中获取API密钥

def get_response(messages):
    # response = dashscope.Generation.call(
    #     model='qwen-max', 得换成专业的模型，因为 qwen-max 不支持图片识别
    #     messages=messages
    # )

    response = dashscope.MultiModalConversation.call(
        model='qwen3-vl-plus', # 支持图片识别的模型，如果使用 qwen-vl-plus 效果会差点
        messages=messages
    )
    return response

content = [
    {'image': 'https://aiwucai.oss-cn-huhehaote.aliyuncs.com/pdf_table.jpg'},
    {'text': '这是一个表格图片，帮我提取里面的内容，仅输出JSON格式'}
]

messages =[{
    'role': 'user',
    'content': content
}]

response = get_response(messages)

print(response)
print(response.output.choices[0].message.content[0]['text'])
