"""
==============================================================================
模块名称：智能天气查询助手 (AI Weather Agent)
功能描述：
    本脚本实现了一个基于大语言模型 (LLM) 的智能天气查询代理。
    它利用阿里云 DashScope (通义千问) 的 Function Calling 能力，
    自动识别用户意图，调用高德地图 API 获取实时天气数据，
    并最终由大模型生成自然流畅的天气报告。

核心流程：
    1. [初始化] 加载环境变量中的 API Keys (DashScope & 高德)。
    2. [定义工具] 向大模型注册 "get_current_weather" 工具及其参数规范。
    3. [用户提问] 用户输入自然语言问题 (如："北京天气怎么样？")。
    4. [意图识别] 大模型分析意图，决定调用工具并提取参数 (城市/代码)。
    5. [执行工具] Python 代码拦截请求，调用高德 API 获取原始 JSON 数据。
    6. [结果合成] 将 API 返回的数据作为上下文再次喂给大模型。
    7. [最终回复] 大模型根据真实数据生成自然语言回答并输出。

前置要求：
    - Python 环境已安装: pip install requests dashscope
    - 环境变量配置:
        * DASHSCOPE_API_KEY: 阿里云百炼/DashScope API Key
        * AMAP_MAPS_API_KEY: 高德开放平台 Web 服务 API Key
==============================================================================
"""

import os
import json
import requests
from http import HTTPStatus
import dashscope
from dotenv import load_dotenv

load_dotenv()
# ==============================================================================
# 配置与常量
# ==============================================================================

# 设置 DashScope API Key (从环境变量读取，保障安全)
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
if not dashscope.api_key:
    raise ValueError("未找到环境变量 DASHSCOPE_API_KEY，请配置后重试。")

# 高德天气 API 的工具定义 (JSON Schema)
# 这部分告诉大模型：如果你需要查天气，请调用这个函数，并遵循以下参数格式
WEATHER_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "获取指定城市的当前天气信息。当用户询问天气时使用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称，例如：'北京', '上海', '广州'",
                },
                "adcode": {
                    "type": "string",
                    "description": "城市行政区划代码 (可选)，例如：'110000'。如果不知道代码，仅传 location 即可。",
                }
            },
            "required": ["location"], # 城市名是必填项
        },
    },
}

# ==============================================================================
# 工具函数实现
# ==============================================================================

def get_current_weather(location: str, adcode: str = None):
    """
    实际执行天气查询的函数。
    调用高德地图 REST API 获取天气数据。
    
    Args:
        location (str): 城市名称。
        adcode (str, optional): 城市代码，优先级高于 location。
        
    Returns:
        dict: 包含天气信息的字典，若失败则返回错误信息。
    """
    gaode_api_key = os.getenv("AMAP_MAPS_API_KEY")
    if not gaode_api_key:
        return {"error": "服务端未配置高德 API Key (AMAP_MAPS_API_KEY)"}

    base_url = "https://restapi.amap.com/v3/weather/weatherInfo"
    
    # 构建请求参数
    # 策略：如果有 adcode 优先用 adcode，否则用城市名让高德自行匹配
    query_city = adcode if adcode else location
    
    params = {
        "key": gaode_api_key,
        "city": query_city,
        "extensions": "base",  # base: 实况天气; all: 预报+实况
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status() # 检查 HTTP 错误
        data = response.json()
        
        # 简单校验高德返回的业务状态码 (status: 1 表示成功)
        if data.get("status") == "1":
            return data
        else:
            return {"error": f"高德API业务失败: {data.get('info', 'Unknown error')}"}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"网络请求失败: {str(e)}"}

# ==============================================================================
# 主逻辑控制
# ==============================================================================

def run_weather_query(user_question: str = "北京现在天气怎么样？"):
    """
    执行完整的天气查询工作流：
    1. 发送问题给大模型
    2. 处理工具调用
    3. 获取真实数据
    4. 让大模型生成最终回复
    """
    
    # 1. 初始化对话历史
    messages = [
        {"role": "system", "content": "你是一个智能天气助手。当用户询问天气时，请调用工具获取真实数据，然后基于数据用自然、友好的语气回答用户。不要直接输出 JSON 数据。"},
        {"role": "user", "content": user_question}
    ]
    
    print(f"🤖 用户提问：{user_question}")
    print("-" * 30)

    # 2. 第一次调用大模型 (意图识别 & 工具调用决策)
    response = dashscope.Generation.call(
        model="qwen-turbo-latest",  # 使用最新的 Qwen 模型
        messages=messages,
        tools=[WEATHER_TOOL_DEFINITION], # 注入工具定义
        tool_choice="auto",              # 让模型自动决定是否调用
        result_format="message"          # 确保返回格式兼容 message 结构
    )

    # 检查第一次调用是否成功
    if response.status_code != HTTPStatus.OK:
        print(f"❌ 大模型请求失败: {response.code} - {response.message}")
        return

    # 获取模型的第一轮响应消息
    first_response_msg = response.output.choices[0].message
    
    # 3. 判断模型是否触发了工具调用
    if hasattr(first_response_msg, 'tool_calls') and first_response_msg.tool_calls:
        tool_call = first_response_msg.tool_calls[0]
        # Handle both object and dictionary structures for tool calls
        if isinstance(tool_call, dict):
            func_name = tool_call['function']['name']
            func_args = json.loads(tool_call['function']['arguments'])
            tool_call_id = tool_call.get('id')
        else:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            tool_call_id = tool_call.id
        
        print(f"🔧 模型决定调用工具：{func_name}")
        print(f"📥 提取参数：{func_args}")

        # 执行对应的本地函数
        if func_name == "get_current_weather":
            location = func_args.get("location", "北京")
            adcode = func_args.get("adcode", None)
            
            # --- 执行真实的外部 API 调用 ---
            weather_data = get_current_weather(location, adcode)
            print(f"🌐 高德 API 返回原始数据：{weather_data}")
            
            # 4. 将工具执行结果回传给大模型 (关键步骤：形成闭环)
            # 构造一个 role 为 'tool' 的消息，携带执行结果
            tool_response_message = {
                "role": "tool",
                "content": json.dumps(weather_data, ensure_ascii=False),
                "tool_call_id": tool_call_id # 关联具体的工具调用 ID
            }
            
            # 更新对话历史：加入模型的调用请求 + 我们的执行结果
            messages.append(first_response_msg)
            messages.append(tool_response_message)
            
            # 5. 第二次调用大模型 (基于真实数据生成自然语言回复)
            print("-" * 30)
            print("⏳ 正在生成最终回复...")
            
            final_response = dashscope.Generation.call(
                model="qwen-turbo-latest",
                messages=messages, # 包含完整上下文的对话
                tools=[WEATHER_TOOL_DEFINITION]
            )
            
            if final_response.status_code == HTTPStatus.OK:
                final_answer = final_response.output.choices[0].message.content
                print("\n✅ 助手回答：")
                print(final_answer)
            else:
                print(f"❌ 生成最终回复失败：{final_response.code}")
                
    else:
        # 如果模型没有调用工具（比如它认为不需要，或者只是闲聊），直接输出内容
        print("\n✅ 助手回答：")
        print(first_response_msg.content)

if __name__ == "__main__":
    # 可以在这里修改测试问题
    test_question = "上海今天的天气如何？"
    run_weather_query(test_question)
