from dotenv import load_dotenv
load_dotenv()

import json
import os
import dashscope
from dashscope.api_entities.dashscope_response import Role
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY') # 从 .env 文件中获取API密钥
# dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY') # 从环境变量中获取API密钥

def get_mock_weather(city: str, unit = '摄氏度') -> str:
    temperature = 0
    if '上海' in city or 'shanghai' in city:
        temperature = 10
    if '北京' in city or 'beijing' in city:
        temperature = -5
    if '福州' in city or 'fuzhou' in city:
        temperature = 15

    weather_info = {
        'city': city,
        'temperature': temperature,
        'unit': unit,
        'forecast': '晴' if temperature > 0 else '阴' if temperature <= 0 else '雨雪'
    }

    return json.dumps(weather_info)

def get_response(messages, functions):
    try:
        response = dashscope.Generation.call(
            # model='deepseek-v3', !NOTE 这里使用 qwen-max 模型，因为 deepseek-v3 模型不支持 function call，它支持的是另一种 tool calls
            model='qwen-max',
            messages=messages,
            functions=functions,
            result_format='message'
        )
        return response
    except Exception as e:
        print(f"调用模型失败: {str(e)}")
        return None

def run_conversation(functions):
    query = "请问上海的天气怎么样？"
    messages=[{'role': 'user', 'content': query}]

    response = get_response(messages, functions)
    if not response or not response.output:
        print("获取模型响应失败")
        return None

    print('response = ', response)

    message = response.output.choices[0].message
    messages.append(message)
    print('message = ', message)

    if hasattr(message, 'function_call') and message.function_call:
        function_call = message.function_call
        tool_name = function_call['name']
        print('tool_name = ', tool_name)
        arguments = json.loads(function_call['arguments'])
        print('arguments = ', arguments)
        tool_response = get_mock_weather(
            city=arguments['city'],
            unit=arguments['unit']
        )
        print('tool_response = ', tool_response)
        tool_info = { 'role': 'function', 'name': tool_name, 'content': tool_response}
        print('tool_info = ', tool_info)
        messages.append(tool_info)
        print('messages = ', messages)

        response = get_response(messages, functions)
        print('response = ', response)
        if not response or not response.output:
            print("获取第二轮模型响应失败")
            return None

        message = response.output.choices[0].message
        print('message = ', message)
        return message
    return message

functions = [
    {
        'name': 'get_mock_weather',
        'description': '获取指定城市的当前天气',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string',
                    'description': '城市名称，例如：上海、北京、福州等',
                },
                'unit': {
                    'type': 'string',
                    'description': '温度单位，摄氏度或华氏度',
                },
            },
            'required': ['city'],
        },
    }
]

if __name__ == '__main__':
    result = run_conversation(functions)
    if result:
        print('最终回复:', result.content)
    else:
        print('没有回复')
