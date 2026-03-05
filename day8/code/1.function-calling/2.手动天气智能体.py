"""
===============================================================================
模块名称：手动天气智能体 (Manual Weather Agent)
===============================================================================

【功能概述】
本脚本演示了如何在不使用高级 Agent 框架（如 qwen-agent）的情况下，
**手动实现**大模型的 Function Calling（函数调用）全流程。
它通过两次独立的 API 调用，完成了“用户提问 -> 模型生成工具指令 -> 
本地执行工具 -> 结果回传模型 -> 生成最终回答”的闭环。

【核心流程】
Step 1: 定义工具 Schema -> 告诉大模型有哪些能力可用。
Step 2: 第一次 API 调用 -> 发送用户问题，让模型判断是否需要调用工具。
Step 3: 解析与执行   -> 若模型返回工具调用指令，本地解析参数并请求高德 API。
Step 4: 结果回传     -> 将 API 返回的天气数据封装为 "tool" 角色消息。
Step 5: 第二次 API 调用 -> 将“问题 + 指令 + 数据”再次发给模型，生成自然语言回答。

【技术要点】
- 显式状态管理：手动维护 messages 列表和 tool_calls 状态。
- 动态参数映射：利用 Python inspect 模块动态匹配函数参数，增强鲁棒性。
- 标准协议遵循：严格遵循 DashScope/OpenAI 风格的 Message 格式。

【前置准备】
1. 安装依赖：pip install dashscope requests
2. 配置密钥：
   - 替换代码中的 DASHSCOPE_API_KEY
   - 替换代码中的 GAODE_API_KEY (高德地图)
===============================================================================
"""
import os
import requests
from http import HTTPStatus
import dashscope
import json
from inspect import signature
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# 全局配置区域
# -----------------------------------------------------------------------------

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
GAODE_API_KEY = os.getenv("AMAP_MAPS_API_KEY")

# 定义工具描述 Schema (JSON Format)
# 这是告诉大模型“我有什么能力”的关键说明书
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. 北京",
                },
                "adcode": {
                    "type": "string",
                    "description": "The city code, e.g. 110000 (北京)",
                }
            },
            "required": ["location"],
        },
    },
}

# -----------------------------------------------------------------------------
# 工具实现区域
# -----------------------------------------------------------------------------

def get_weather_from_gaode(location: str, adcode: str = None):
    """
    具体工具实现：调用高德地图 API 获取实时天气。
    
    :param location: 城市名称
    :param adcode: 城市行政编码（可选，优先级高于城市名）
    :return: 包含天气信息的字典或错误信息
    """
    base_url = "https://restapi.amap.com/v3/weather/weatherInfo"
    
    params = {
        "key": GAODE_API_KEY,
        # 优先使用行政编码，若无则使用城市名
        "city": adcode if adcode else location,
        "extensions": "base",  # base=实时，all=预报+实时
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Exception: {str(e)}"}

# -----------------------------------------------------------------------------
# 核心逻辑区域：手动 Agent 循环
# -----------------------------------------------------------------------------

def run_weather_query():
    """
    执行完整的天气查询流程。
    模拟 Agent 的思考与行动过程：
    1. 询问模型 -> 2. 解析工具调用 -> 3. 执行工具 -> 4. 反馈结果 -> 5. 生成回答
    """
    
    # 初始化对话历史
    messages = [
        {"role": "system", "content": "你是一个智能助手，可以查询天气信息。如果用户询问天气，请调用工具获取真实数据后回答。"},
        {"role": "user", "content": "北京现在天气怎么样？"}
    ]
    
    print("🔄 [阶段 1] 正在调用大模型识别意图...")
    
    # === 第一次调用：意图识别与工具规划 ===
    response = dashscope.Generation.call(
        model="qwen-turbo",
        messages=messages,
        tools=[weather_tool],      # 传入工具定义
        tool_choice="auto",        # 让模型自动决定是否调用
        result_format='message'    # 确保返回格式兼容 message 结构
    )
    
    if response.status_code != HTTPStatus.OK:
        print(f"❌ 第一次调用失败：{response.code} - {response.message}")
        return

    # 解析模型返回的消息
    assistant_message = response.output.choices[0].message
    
    # 检查模型是否决定调用工具
    # 注意：不同版本 SDK 返回结构可能略有差异，此处做兼容性检查
    tool_calls = getattr(assistant_message, 'tool_calls', None)

    if tool_calls:
        print("✅ 检测到工具调用指令，正在执行本地工具...")
        
        # 构建工具映射表，方便扩展更多工具
        tool_map = {
            "get_current_weather": get_weather_from_gaode,
        }
        
        # 将 assistant 的消息转为字典格式以便后续拼接
        assistant_dict = {
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": tool_calls # 保留工具调用记录
        }
        
        tool_response_messages = []
        
        # 遍历所有待调用的工具（支持并行调用扩展）
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            func_args = json.loads(tool_call["function"]["arguments"])
            call_id = tool_call["id"]
            
            print(f"   🛠️ 执行工具：{func_name}, 参数：{func_args}")
            
            if func_name in tool_map:
                target_func = tool_map[func_name]
                
                # 【优雅处理】动态参数匹配
                # 防止函数参数与模型生成参数不完全一致导致报错
                sig = signature(target_func)
                valid_args = {
                    k: v for k, v in func_args.items() 
                    if k in sig.parameters
                }
                
                # 执行实际的工具函数
                result = target_func(**valid_args)
                
                # 构建工具响应消息 (Role: Tool)
                # 这是让大模型“看到”执行结果的关键
                tool_response = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": func_name,
                    "content": json.dumps(result, ensure_ascii=False)
                }
                tool_response_messages.append(tool_response)
            else:
                print(f"   ⚠️ 未找到工具函数：{func_name}")

        # === 组装完整上下文 ===
        # 顺序：历史消息 + 模型指令 + 工具执行结果
        updated_messages = messages + [assistant_dict] + tool_response_messages
        
        print("🔄 [阶段 2] 正在将工具结果回传给大模型以生成最终回答...")
        
        # === 第二次调用：基于事实生成回答 ===
        response2 = dashscope.Generation.call(
            model="qwen-turbo",
            messages=updated_messages,
            tools=[weather_tool],
            tool_choice="auto",
            result_format='message'
        )
        
        if response2.status_code == HTTPStatus.OK:
            final_content = response2.output.choices[0].message.content
            print("\n" + "="*30)
            print("🤖 最终回复:")
            print(final_content)
            print("="*30 + "\n")
        else:
            print(f"❌ 第二次调用失败：{response2.code} - {response2.message}")
            
    else:
        # 模型认为不需要调用工具，直接回答
        print("💡 模型未调用工具，直接输出回答:")
        print(assistant_message.content)

# -----------------------------------------------------------------------------
# 程序入口
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 运行天气查询流程
    run_weather_query()
