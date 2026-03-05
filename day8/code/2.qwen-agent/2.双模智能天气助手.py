# -*- coding: utf-8 -*-
"""
==============================================================================
模块名称：双模天气智能助手 (Dual-Mode Weather Intelligent Assistant)

【功能概述】
本脚本构建了一个高级 AI Agent，能够理解自然语言并自动查询实时天气。
其核心特色在于采用了“双模工具架构”：
1. 传统模式：通过内置 Python 类直接调用高德地图 API。
2. 前沿模式：通过 MCP (Model Context Protocol) 协议连接高德官方地图服务节点。
这种架构既保证了代码的可控性，又展示了未来 Agent 连接外部服务的标准化方向。

【核心组件】
- LLM 引擎：阿里云通义千问 (qwen-turbo)，负责意图识别与回答生成。
- 工具层 A：WeatherTool (Python 原生实现)。
- 工具层 B：AMap MCP Server (Node.js 进程，通过标准协议通信)。
- 交互层：支持 Web 图形界面 (默认) 和 终端命令行界面。

【运行流程】
Step 1: 环境初始化 -> 加载 DashScope API Key，配置超时参数。
Step 2: 工具注册   -> 
        (A) 定义并注册本地 Python 天气工具。
        (B) 配置 MCP 服务器启动参数 (命令、参数、环境变量)。
Step 3: Agent 组装  -> 创建 Assistant 实例，注入 LLM 配置与混合工具列表。
Step 4: 服务启动   -> 
        - 若为 Web 模式：启动本地 HTTP 服务器，渲染聊天 UI。
        - 若为终端模式：进入输入循环，维护对话历史上下文。
Step 5: 意图处理   -> 
        - 用户提问 -> LLM 分析 -> 选择最佳工具 (Python 或 MCP) -> 执行请求。
        - 获取数据 -> LLM 整合信息 -> 输出自然语言回复。

==============================================================================
"""

import os
import json
import asyncio
import requests
from typing import Optional, List, Dict, Any


# 第三方库导入
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# 全局配置与常量
# ==============================================================================

# 定义资源文件根目录 (预留扩展)
ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

# 配置 DashScope 全局参数
# 注意：生产环境中建议通过 os.getenv('DASHSCOPE_API_KEY') 从环境变量读取
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '') 
if not dashscope.api_key:
    # 如果环境变量未设置，且代码中未在其他地方显式赋值，这里可能会失败
    print("⚠️ 警告: 未检测到 DASHSCOPE_API_KEY 环境变量，请确保已在其他地方配置。")

gaode_api_key = os.getenv("AMAP_MAPS_API_KEY")
if not gaode_api_key:
    print("⚠️ 警告: 未检测到 AMAP_MAPS_API_KEY 环境变量，请确保已在其他地方配置。")

dashscope.timeout = 30  # 设置请求超时时间为 30 秒，防止网络卡顿导致程序假死

# 定义工具描述 Schema (用于大模型理解工具功能)
FUNCTIONS_DESC = [
    {
        "name": "get_current_weather",
        "description": "获取指定位置的当前天气情况，包括温度、风向、湿度等。",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称，例如：北京、上海、广州",
                },
                "adcode": {
                    "type": "string",
                    "description": "城市行政区划代码，例如：110000（北京）。可选，若不提供则尝试通过城市名匹配。",
                }
            },
            "required": ["location"],
        },
    }
]

# ==============================================================================
# 自定义工具实现：天气查询
# ==============================================================================

@register_tool('get_current_weather')
class WeatherTool(BaseTool):
    """
    天气查询工具类。
    继承自 qwen_agent 的 BaseTool，通过 @register_tool 装饰器注册到系统中。
    负责调用高德地图 API 并格式化返回结果。
    """
    
    # 工具描述，会传递给大模型
    description = '获取指定位置的当前天气情况'
    
    # 参数定义，用于大模型提取参数
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
        Args:
            params (str): JSON 格式的参数字符串，由大模型生成。
            **kwargs: 其他上下文参数。
        Returns:
            str: 格式化后的天气信息或错误消息。
        """
        try:
            args = json.loads(params)
            location = args.get('location', '')
            adcode = args.get('adcode', None)
            
            if not location:
                return "错误：未提供城市名称。"
                
            return self.get_weather_from_gaode(location, adcode)
        except json.JSONDecodeError:
            return "错误：参数格式解析失败。"
        except Exception as e:
            return f"工具执行出错：{str(e)}"
    
    def get_weather_from_gaode(self, location: str, adcode: str = None) -> str:
        """
        内部方法：执行高德地图 API 请求。
        """
        GAODE_API_KEY = gaode_api_key 
        BASE_URL = "https://restapi.amap.com/v3/weather/weatherInfo"
        
        params = {
            "key": GAODE_API_KEY,
            "city": adcode if adcode else location,
            "extensions": "base",  # base: 实况; all: 预报+实况
        }
        
        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status() # 检查 HTTP 状态码
            data = response.json()
            
            # 检查高德业务状态码 (1: 成功, 0: 失败)
            if data.get('status') == '1' and data.get('lives'):
                weather_info = data['lives'][0]
                # 格式化输出，使其更适合大模型阅读和用户查看
                result = (
                    f"🌤️ 天气查询结果：\n"
                    f"📍 城市：{weather_info.get('city')}\n"
                    f"☁️ 天气：{weather_info.get('weather')}\n"
                    f"🌡️ 温度：{weather_info.get('temperature')}°C\n"
                    f"🍃 风向：{weather_info.get('winddirection')}\n"
                    f"💨 风力：{weather_info.get('windpower')}级\n"
                    f"💧 湿度：{weather_info.get('humidity')}%\n"
                    f"🕒 发布时间：{weather_info.get('reporttime')}"
                )
                return result
            else:
                return f"❌ 获取天气信息失败：{data.get('info', '未知错误')}"
                
        except requests.exceptions.RequestException as e:
            return f"🌐 网络请求失败：{str(e)}"
        except Exception as e:
            return f"⚠️ 发生未知错误：{str(e)}"

# ==============================================================================
# 核心服务初始化
# ==============================================================================

def init_agent_service():
    """
    初始化智能助手服务。
    配置 LLM 模型、加载本地工具及 MCP 远程工具。
    
    Returns:
        Assistant: 初始化完成的 Agent 实例。
    """
    # 大模型配置
    llm_cfg = {
        'model': 'qwen-turbo-2025-04-28',  # 指定模型版本
        'timeout': 30,
        'retry_count': 3,  # 失败重试次数
    }

    # 工具列表配置
    tools = [
        # 1. 引用本地注册的 Python 工具
        'get_current_weather',
        
        # 2. 配置 MCP (Model Context Protocol) 服务器
        # 允许 Agent 通过标准协议调用外部 Node.js 服务
        {
            "mcpServers": {
                "amap-maps": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "@amap/amap-maps-mcp-server"
                    ],
                    "env": {
                        "AMAP_MAPS_API_KEY": gaode_api_key
                    }
                }
            }
        }
    ]
    
    try:
        print("🚀 正在初始化智能助手...")
        bot = Assistant(
            llm=llm_cfg,
            name='天气助手',
            description='一个能查询实时天气的智能助手',
            system_message="你是一名专业、友好的天气助手。请使用提供的工具查询天气，并用自然、简洁的语言回答用户。如果查询失败，请礼貌地告知用户原因。",
            function_list=tools,
        )
        print("✅ 助手初始化成功！已加载本地工具及 MCP 服务配置。")
        return bot
    except Exception as e:
        print(f"❌ 助手初始化失败: {str(e)}")
        raise

# ==============================================================================
# 交互模式实现
# ==============================================================================

def app_tui():
    """
    终端交互模式 (Text User Interface)。
    提供命令行下的连续对话功能，支持文件输入（预留）。
    """
    try:
        bot = init_agent_service()
        messages = []  # 维护对话历史
        
        print("\n" + "="*40)
        print("🖥️  已进入终端对话模式 (输入 'exit' 退出)")
        print("="*40 + "\n")
        
        while True:
            try:
                # 获取用户文本输入
                query = input('👤 用户提问: ').strip()
                
                # 退出条件
                if query.lower() in ['exit', 'quit', '退出']:
                    print("👋 再见！")
                    break
                    
                if not query:
                    print('⚠️  提问不能为空，请重试。\n')
                    continue
                
                # 获取可选的文件输入 (当前版本主要演示文本，此功能为框架预留)
                file_url = input('📎 文件链接 (直接回车跳过): ').strip()
                
                # 构建消息体
                if not file_url:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file_url}]})

                print("\n🤖 助手正在思考并查询数据...")
                
                # 流式处理响应
                # bot.run 返回一个生成器，逐步产出响应片段
                full_response = []
                for response_chunk in bot.run(messages):
                    # response_chunk 通常是一个 list，包含最新的消息状态
                    if response_chunk:
                        last_msg = response_chunk[-1]
                        if last_msg.get('role') == 'assistant':
                            content = last_msg.get('content', '')
                            # 简单打印流式内容 (实际生产中可做更精细的流式渲染)
                            print(f"\r💬 回复: {content}", end='', flush=True)
                            full_response = response_chunk
                
                print("\n") # 换行
                # 将完整的助手响应加入历史记录，保持上下文
                messages.extend(full_response)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  用户中断对话。")
                break
            except Exception as e:
                print(f"\n❌ 处理请求时出错: {str(e)}")
                print("请检查网络连接或 API 配置，然后重试。\n")
                
    except Exception as e:
        print(f"❌ 启动终端模式失败: {str(e)}")


def app_gui():
    """
    图形界面模式 (Graphical User Interface)。
    基于 Streamlit 或 Gradio (取决于 qwen_agent 实现) 启动 Web 服务。
    """
    try:
        print("\n🚀 正在启动 Web 图形界面...")
        bot = init_agent_service()
        
        # 配置聊天界面的预设问题，引导用户
        chatbot_config = {
            'prompt.suggestions': [
                '北京今天的天气怎么样？',
                '上海现在下雨吗？',
                '广州明天的气温是多少？' # 注意：当前工具仅支持实况，预报需修改 extensions
            ]
        }
        
        print("✅ Web 服务准备就绪。")
        print("🌐 请在浏览器中打开显示的本地地址进行交互。")
        print("(按 Ctrl+C 停止服务)\n")
        
        # 启动 WebUI
        # run() 方法通常会阻塞当前线程直到服务停止
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
        
    except Exception as e:
        print(f"❌ 启动 Web 界面失败: {str(e)}")
        print("💡 建议检查：\n1. 网络连接是否正常\n2. API Key 是否有效\n3. 端口是否被占用")


# ==============================================================================
# 程序入口
# ==============================================================================

if __name__ == '__main__':
    # 运行模式选择
    # 当前默认启动图形界面 (GUI)
    # 如需使用终端模式，可注释掉 app_gui() 并取消 app_tui() 的注释
    
    try:
        # app_gui()          # 启动 Web 图形界面
        app_tui()        # [备选] 启动命令行终端界面
    except KeyboardInterrupt:
        print("\n\n🛑 服务已手动停止。")
    except Exception as e:
        print(f"\n💥 程序发生严重错误: {e}")
