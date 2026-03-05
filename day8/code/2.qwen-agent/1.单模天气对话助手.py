"""
===============================================================================
模块名称：智能天气查询助手 (Intelligent Weather Assistant)
===============================================================================

【功能概述】
本脚本构建了一个基于大语言模型 (LLM) 的智能 Agent，能够理解自然语言指令，
自动调用外部工具查询实时天气信息。它结合了阿里云通义千问 (Qwen) 的语义理解能力
和高德地图 (AMap) 的实时数据服务。

【核心组件】
1. LLM 引擎：使用 qwen-turbo 模型进行意图识别和回答生成。
2. 自定义工具 (WeatherTool)：封装高德地图 API，提供标准化的天气查询接口。
3. 交互界面：默认启动 Web GUI，支持浏览器聊天；同时也保留了 TUI (命令行) 模式。

【运行流程】
Step 1: 初始化配置 -> 加载环境变量中的 DashScope API Key。
Step 2: 注册工具   -> 定义并注册 `get_current_weather` 工具类。
Step 3: 启动服务   -> 初始化 Assistant Agent，绑定 LLM 配置与工具列表。
Step 4: 用户交互   -> 
        (A) Web 模式：启动本地服务器，用户在浏览器输入问题。
        (B) TUI 模式：用户在终端输入问题。
Step 5: 意图识别   -> LLM 分析用户输入，若发现天气查询意图，提取城市参数。
Step 6: 工具执行   -> 调用高德 API 获取 JSON 数据，解析为自然语言。
Step 7: 生成回复   -> LLM 结合工具返回的数据，生成最终友好的回答。

【前置准备】
1. 环境依赖：pip install qwen-agent dashscope requests pandas sqlalchemy
2. API 密钥：
   - 阿里云 DashScope API Key (需在环境变量设置 DASHSCOPE_API_KEY)
   - 高德地图 Web 服务 API Key (需在代码中配置)
3. 网络环境：确保能访问阿里云和高德地图 API 服务。

===============================================================================
"""

import os
import asyncio
import requests
from typing import Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
import pandas as pd
from sqlalchemy import create_engine
from qwen_agent.tools.base import BaseTool, register_tool
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# 全局配置区域
# -----------------------------------------------------------------------------

# 定义资源文件根目录，用于后续可能扩展的文件读取功能
ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

# 配置 DashScope (阿里云模型服务)
# 优先从环境变量获取 API Key，保障密钥安全；若无则设为空字符串（会导致后续调用失败）
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
dashscope.timeout = 30  # 设置网络请求超时时间为 30 秒，防止长时间阻塞


# 定义工具描述 schema
# 这部分信息会发送给 LLM，让它知道有哪些工具可用以及参数的格式
functions_desc = [
    {
        "name": "get_current_weather",
        "description": "获取指定位置的当前天气情况",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称，例如：北京",
                },
                "adcode": {
                    "type": "string",
                    "description": "城市编码，例如：110000（北京）",
                }
            },
            "required": ["location"],
        },
    }
]

# -----------------------------------------------------------------------------
# 自定义工具实现：天气查询
# -----------------------------------------------------------------------------

@register_tool('get_current_weather')
class WeatherTool(BaseTool):
    """
    天气查询工具类。
    继承自 BaseTool，通过 @register_tool 装饰器注册到 Agent 系统中。
    负责具体执行高德地图 API 的 HTTP 请求和数据解析。
    """
    description = '获取指定位置的当前天气情况'
    
    # 定义工具参数元数据，辅助 LLM 理解如何调用
    parameters = [{
        'name': 'location',
        'type': 'string',
        'description': '城市名称，例如：北京',
        'required': True
    }, {
        'name': 'adcode',
        'type': 'string',
        'description': '城市编码，例如：110000（北京）',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        """
        工具入口函数。
        :param params: JSON 格式的参数字符串，由 Agent 框架自动填入
        :return: 天气查询结果的文本描述
        """
        import json
        # 解析 Agent 传来的 JSON 参数
        args = json.loads(params)
        location = args['location']
        adcode = args.get('adcode', None)
        
        # 执行具体的天气查询逻辑
        return self.get_weather_from_gaode(location, adcode)
    
    def get_weather_from_gaode(self, location: str, adcode: str = None) -> str:
        """
        核心业务逻辑：调用高德地图 API 查询天气。
        :param location: 城市名称
        :param adcode: 城市行政编码（可选，优先级高于城市名）
        :return: 格式化后的天气信息字符串或错误信息
        """
        gaode_api_key = os.getenv('AMAP_MAPS_API_KEY', '')
        
        base_url = "https://restapi.amap.com/v3/weather/weatherInfo"
        
        # 构建 API 请求参数
        params = {
            "key": gaode_api_key,
            # 优先使用行政编码，若无则直接使用城市名称
            "city": adcode if adcode else location,
            "extensions": "base",  # "base" 返回实时天气，"all" 返回预报 + 实时
        }
        
        try:
            # 发送 HTTP GET 请求
            response = requests.get(base_url, params=params)
            
            # 检查 HTTP 状态码
            if response.status_code == 200:
                data = response.json()
                
                # 检查高德 API 业务状态码 ('1' 表示成功) 且包含生活指数数据
                if data.get('status') == '1' and data.get('lives'):
                    weather_info = data['lives'][0]
                    
                    # 格式化输出结果，使其对人类更友好
                    result = (
                        f"天气查询结果：\n"
                        f"城市：{weather_info.get('city')}\n"
                        f"天气：{weather_info.get('weather')}\n"
                        f"温度：{weather_info.get('temperature')}°C\n"
                        f"风向：{weather_info.get('winddirection')}\n"
                        f"风力：{weather_info.get('windpower')}\n"
                        f"湿度：{weather_info.get('humidity')}%\n"
                        f"发布时间：{weather_info.get('reporttime')}"
                    )
                    return result
                else:
                    # API 调用成功但业务逻辑失败（如城市不存在）
                    return f"获取天气信息失败：{data.get('info', '未知错误')}"
            else:
                # 网络层面或服务器错误
                return f"请求失败：HTTP状态码 {response.status_code}"
                
        except Exception as e:
            # 捕获所有未预期的异常（如网络断开、JSON 解析错误等）
            return f"获取天气信息出错：{str(e)}"

# -----------------------------------------------------------------------------
# 助手服务初始化
# -----------------------------------------------------------------------------

def init_agent_service():
    """
    初始化 AI 助手服务。
    配置大模型参数、系统人设以及绑定的工具列表。
    :return: 初始化完成的 Assistant 对象
    """
    # 大模型配置
    llm_cfg = {
        'model': 'qwen-max',  # 指定模型版本
        'timeout': 30,                     # 模型推理超时时间
        'retry_count': 3,                  # 失败重试次数
    }
    
    try:
        # 实例化 Assistant Agent
        bot = Assistant(
            llm=llm_cfg,
            name='天气助手',
            description='天气助手，查询天气',
            system_message="你是一名有用的助手",  # 设定 AI 的人设基调
            function_list=['get_current_weather'],  # 注册可用的工具名称
        )
        print("助手初始化成功！")
        return bot
    except Exception as e:
        # 初始化失败通常是因为 API Key 错误或模型名称无效
        print(f"助手初始化失败：{str(e)}")
        raise

# -----------------------------------------------------------------------------
# 交互模式实现
# -----------------------------------------------------------------------------

def app_tui():
    """
    终端交互模式 (Text User Interface)。
    提供命令行聊天窗口，支持多轮对话上下文记忆。
    """
    try:
        # 1. 初始化助手
        bot = init_agent_service()

        # 2. 维护对话历史列表，用于多轮对话上下文
        messages = []
        
        print("--- 已进入终端对话模式 (输入 Ctrl+C 退出) ---")
        
        while True:
            try:
                # 获取用户文本输入
                query = input('\nuser question: ')
                
                # 获取可选的文件 URL 输入（本例中天气工具主要依赖文本，此功能为扩展预留）
                file = input('file url (press enter if no file): ').strip()
                
                # 输入有效性校验
                if not query:
                    print('❌ 问题不能为空，请重试！')
                    continue
                    
                # 构建符合 Agent 协议的消息格式
                if not file:
                    messages.append({'role': 'user', 'content': query})
                else:
                    # 若包含文件，构造多模态消息结构
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})

                print("\n⏳ 正在处理您的请求...")
                
                # 流式运行助手并打印响应
                # bot.run 返回一个生成器，逐步输出思考过程和最终结果
                response = []
                for response in bot.run(messages):
                    # 这里简化处理，直接打印最后的响应内容
                    print('🤖 bot response:', response)
                
                # 将助手的回复加入历史记录，保持上下文连贯
                messages.extend(response)
                
            except KeyboardInterrupt:
                print("\n👋 用户中断退出。")
                break
            except Exception as e:
                print(f"❌ 处理请求时出错：{str(e)}")
                print("请重试或输入新的问题")
                
    except Exception as e:
        print(f"💥 启动终端模式失败：{str(e)}")


def app_gui():
    """
    图形界面模式 (Web GUI)。
    基于 qwen_agent.gui 启动本地 Web 服务器，提供类似 ChatGPT 的网页聊天界面。
    """
    try:
        print("🚀 正在启动 Web 界面...")
        
        # 1. 初始化助手
        bot = init_agent_service()
        
        # 2. 配置聊天界面的引导提示语 (Prompt Suggestions)
        # 帮助用户快速了解助手能做什么
        chatbot_config = {
            'prompt.suggestions': [
                '北京今天的天气怎么样？',
                '上海现在下雨吗？',
                '广州的温度是多少？'
            ]
        }
        
        print("✅ Web 界面准备就绪。")
        print("🌐 请在浏览器中打开显示的地址开始对话...")
        
        # 3. 启动 Web 服务 (阻塞运行)
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
        
    except Exception as e:
        print(f"💥 启动 Web 界面失败：{str(e)}")
        print("💡 提示：请检查网络连接、DashScope API Key 配置以及依赖库是否安装完整。")


# -----------------------------------------------------------------------------
# 程序入口
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # 选择运行模式
    # 当前默认启动图形界面模式，如需命令行模式可改为 app_tui()
    app_gui()          
    # app_tui()        
