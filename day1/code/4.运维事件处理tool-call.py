from dotenv import load_dotenv
load_dotenv()

import json
import os
import random
import dashscope
from dashscope.api_entities.dashscope_response import Role
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY') # 从 .env 文件中获取API密钥
# dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY') # 从环境变量中获取API密钥

def get_mock_status():
    connections = random.randint(10, 100)
    cpu_usage = round(random.uniform(0, 100), 1)
    memory_usage = round(random.uniform(10, 100), 1)
    status_info = {
        '连接数': connections,
        'CPU占用率': cpu_usage,
        '内存占用率': memory_usage
    }
    return json.dumps(status_info, ensure_ascii=False)

def get_response (messages):
    response = dashscope.Generation.call(
        model='qwen-turbo', # qwen-turbo、qwen-max、deepseek-v3 都支持 tool calls
        messages=messages,
        tools=tools,
        result_format='message'
    )
    return response

current_locals = locals()

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_mock_status',
            'description': '调用监控系统接口，获取模拟的系统状态，包括连接数、CPU占用率和内存占用率',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': []
            }
        }
    }
]

input = """告警：系统内存占用率超过90%
时间：2026-02-14 12:00:00
"""
messages=[
    {'role': 'system', 'content': '你是一个运维事件处理助手，你的任务是根据告警信息和系统状态，判断是否需要触发运维事件。'},
    {'role': 'user', 'content': input}
]

i = 0

while True:
    i += 1
    response = get_response(messages)
    print('response = ', response)
    message = response.output.choices[0].message
    print('message = ', message)
    messages.append(message)
    print('messages = ', messages)

    if response.output.choices[0].finish_reason == 'stop':
        break
    if i > 3:
        break

    if message.tool_calls:
        for tool_call in message.tool_calls:
            print('tool_call = ', tool_call)
            function_name = tool_call['function']['name']
            print('function_name = ', function_name)
            function_to_call = current_locals.get(function_name)
            if function_to_call:
                function_args = json.loads(tool_call['function']['arguments'])
                print('function_args = ', function_args)
                function_response = function_to_call(**function_args)
                print('function_response = ', function_response)
                tool_info = {
                    'name': function_name,
                    'role': 'tool',
                    'tool_call_id': tool_call['id'],
                    'content': function_response
                }
                print('tool_info = ', tool_info)
                messages.append(tool_info)
                print('messages = ', messages)

