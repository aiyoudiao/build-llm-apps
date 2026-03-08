#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高德地图智能助手 (Amap Intelligent Assistant)

【功能概述】
本模块是一个基于阿里云通义千问 (Qwen) 大模型、Qwen-Agent 框架以及高德地图官方 MCP 服务构建的智能应用。
它充当“用户”与“高德地图真实数据”之间的智能桥梁：
1. 理解用户的自然语言地理需求（如“规划从机场到酒店的路线”、“推荐附近的餐厅”）。
2. 自动调度高德地图官方 MCP 服务 (@amap/amap-maps-mcp-server) 获取实时、准确的地图数据。
3. 利用大模型的推理能力，将复杂的地图数据整合为易懂的行程建议或导航指引。

【系统架构】
- 交互层：提供 Web 图形界面 (GUI)、终端交互 (TUI) 和单次测试 (Test) 三种模式。
- 代理层 (Agent)：使用 qwen-max 模型作为大脑，负责意图识别、任务拆解和结果润色。
- 工具层 (MCP)：通过 Model Context Protocol 动态调用高德官方 Node.js 服务。
  * 特点：无需编写本地地图脚本，直接复用高德官方提供的标准 API 集合（搜索、路径规划、POI 查询等）。

【核心流程】
1. [初始化] 读取 DashScope API Key，配置 LLM 参数。
2. [注册工具] 定义 MCP 服务器配置，使用 `npx` 启动高德官方服务包，并注入高德 API Key。
3. [实例化] 创建 Assistant 对象，注入系统提示词（设定为专业地图向导）和工具列表。
4. [交互循环] 
   - 接收用户输入 (文本描述 或 文件/图片)。
   - Agent 分析意图 -> 决定调用高德工具 (如：AMAP_SEARCH, AMAP_ROUTE)。
   - MCP 服务向高德服务器请求真实数据。
   - Agent 结合数据生成自然语言回复 (如：“建议您乘坐地铁 2 号线，全程约 40 分钟...”)。

【依赖说明】
- 环境依赖：必须安装 Node.js (包含 npm/npx)，用于运行高德 MCP 服务。
- Python 依赖：`pip install dashscope qwen-agent`。
- 密钥配置：
  1. 环境变量 `DASHSCOPE_API_KEY` (阿里云大模型)。
  2. 代码中需填入有效的 `AMAP_MAPS_API_KEY` (高德开放平台)。

【运行模式】
- GUI 模式 (默认): 启动本地 Web 服务器，提供友好的聊天界面和丰富的旅游/导航预设场景。
- TUI 模式：在命令行中进行多轮对话，支持文件输入。
- Test 模式：执行单次查询用于调试连通性。
"""

import os
from typing import Optional, List, Dict, Any
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from dotenv import load_dotenv

load_dotenv()
# ==========================================
# 1. 全局配置与初始化
# ==========================================

# [安全提示] API Key 应从环境变量读取，避免硬编码在代码中提交到版本控制
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
dashscope.timeout = 30  # 设置 HTTP 请求超时时间
AMAP_MAPS_API_KEY = os.getenv('AMAP_MAPS_API_KEY', '')

def init_agent_service() -> Assistant:
    """
    初始化高德地图助手服务
    
    此函数负责构建 Agent 的核心配置，重点在于正确配置高德地图的 MCP 服务。
    
    Returns:
        Assistant: 配置完成并可立即使用的助手实例
        
    Raises:
        Exception: 当 API Key 缺失、Node.js 环境缺失或服务初始化失败时抛出异常
    """
    # [LLM 配置] 使用通义千问最大模型 (qwen-max) 以处理复杂的地理逻辑和多步规划
    llm_cfg = {
        'model': 'qwen-max',
        'timeout': 30,      # 单次调用超时
        'retry_count': 3,   # 网络波动时的自动重试次数
    }
    
    # [系统角色设定] 定义 AI 的行为准则
    # 强调其作为“专业地图向导”的身份，鼓励使用工具获取实时数据
    system_prompt = (
        '你扮演一个专业的地图与旅行助手，依托高德地图数据为用户提供服务。\n'
        '你的核心能力包括：地点搜索 (POI)、路径规划 (驾车/公交/步行/骑行)、周边推荐、天气查询等。\n'
        '重要原则：\n'
        '1. 必须优先调用“高德地图服务”工具获取真实、实时的地理数据，严禁编造地址或路线。\n'
        '2. 在规划路线时，应综合考虑时间、距离和交通方式，给出最优建议。\n'
        '3. 回答应结构清晰，包含具体的步骤、预计时间和关键地标。\n'
        '4. 如果用户意图模糊，应主动询问细节（如出发地、偏好交通方式）。'
    )
    
    # [MCP 工具配置] 定义如何启动高德官方服务
    # 关键点：使用 npx 动态运行 @amap/amap-maps-mcp-server
    # 注意：必须将 "你的KEY" 替换为真实的高德开放平台 API Key
    tools_config = [{
        "mcpServers": {
            "amap-maps": {
                "command": "npx",  # 使用 Node.js 的包执行工具
                "args": [
                    "-y",          # 自动确认安装，无需交互
                    "@amap/amap-maps-mcp-server" # 高德官方提供的 MCP 服务包
                ],
                "env": {
                    # 获取地址：https://lbs.amap.com/
                    "AMAP_MAPS_API_KEY": AMAP_MAPS_API_KEY
                }
            }
        }
    }]
    
    # [预检查] 简单的 Key 格式检查
    if tools_config[0]["mcpServers"]["amap-maps"]["env"]["AMAP_MAPS_API_KEY"] == "你的KEY":
        print("⚠️  警告：检测到高德 API Key 仍为占位符 '你的KEY'，请在代码中替换为真实 Key 否则服务将不可用！")

    try:
        # [实例化助手] 组装所有配置创建 Assistant 对象
        bot = Assistant(
            llm=llm_cfg,
            name='高德地图智能助手',
            description='基于高德真实数据的地图查询、路线规划与旅行推荐专家',
            system_message=system_prompt,
            function_list=tools_config,
        )
        print("✅ 助手服务初始化成功！已加载 qwen-max 模型及高德地图 MCP 服务。")
        return bot
    except Exception as e:
        error_msg = f"❌ 助手初始化失败: {str(e)}"
        print(error_msg)
        # 提供具体的排查建议
        if "api_key" in str(e).lower() or not dashscope.api_key:
            print("💡 提示：请检查是否已设置环境变量 DASHSCOPE_API_KEY")
        if "npx" in str(e) or "node" in str(e).lower():
            print("💡 提示：请确认系统已安装 Node.js 并且 npx 命令可用")
        if "你的KEY" in str(e) or "Invalid Key" in str(e):
            print("💡 提示：请检查代码中的 AMAP_MAPS_API_KEY 是否已替换为有效的高德 Key")
        raise

# ==========================================
# 2. 功能模式定义
# ==========================================

def test_mode(query: str = '帮我查找上海东方明珠的具体位置', file_path: Optional[str] = None):
    """
    测试模式：执行单次查询并打印结果
    
    适用于快速验证高德 Key 有效性、Node.js 环境及工具调用是否正常。
    
    Args:
        query: 用户查询语句
        file_path: 可选的本地文件路径（如包含位置信息的图片）
    """
    try:
        print(f"🚀 进入测试模式，正在处理请求：'{query}'...")
        bot = init_agent_service()
        messages = []

        # [消息构建] 根据是否有文件附件构建不同格式的消息体
        if not file_path:
            messages.append({'role': 'user', 'content': query})
        else:
            # 支持多模态输入：文本指令 + 文件/图片
            messages.append({'role': 'user', 'content': [{'text': query}, {'file': file_path}]})

        # [流式处理] 遍历生成器获取实时响应
        for response in bot.run(messages):
            if response:
                last_msg = response[-1]
                if last_msg.get('role') == 'assistant':
                    print(f"\n🤖 助手回答:\n{last_msg['content']}")
                    
    except Exception as e:
        print(f"💥 测试过程中发生错误: {str(e)}")

def app_tui():
    """
    终端交互模式 (TUI)
    
    提供命令行下的多轮对话体验。
    特点：轻量级，无需浏览器，适合在服务器后台快速查询路线或地点。
    """
    print("🖥️  正在启动终端交互模式... (输入 'quit' 退出)")
    try:
        bot = init_agent_service()
        conversation_history: List[Dict[str, Any]] = []

        while True:
            try:
                # [用户输入] 获取问题
                query = input('\n👤 用户提问: ').strip()
                
                # [退出条件]
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见！")
                    break
                
                if not query:
                    print("⚠️  输入不能为空，请重新输入。")
                    continue

                # [文件输入] 允许用户输入本地文件路径或 URL
                file_input = input('📎 附加文件/图片路径 (直接回车跳过): ').strip()
                
                # [构建消息]
                if not file_input:
                    conversation_history.append({'role': 'user', 'content': query})
                else:
                    conversation_history.append({'role': 'user', 'content': [{'text': query}, {'file': file_input}]})

                print("⏳ 正在查询地图数据并规划...")
                
                # [执行推理]
                final_response = []
                for response_snapshot in bot.run(conversation_history):
                    final_response = response_snapshot
                
                # [更新历史] 将助手的回复加入上下文，以支持多轮对话
                if final_response:
                    conversation_history.extend(final_response)
                    last_msg = final_response[-1]
                    if last_msg.get('role') == 'assistant':
                        print(f"\n🤖 助手:\n{last_msg['content']}")
                        
            except KeyboardInterrupt:
                print("\n⚠️  检测到中断信号，退出对话。")
                break
            except Exception as e:
                print(f"💥 对话处理出错: {str(e)}")
                print("💡 建议：检查网络连接、API Key 配置或 Node.js 环境。")
                
    except Exception as e:
        print(f"❌ 启动终端模式失败: {str(e)}")

def app_gui():
    """
    图形界面模式 (GUI)
    
    启动本地 Web 服务器，提供现代化的聊天界面。
    特点：
    - 预设丰富的旅游与导航场景建议，降低用户使用门槛。
    - 支持 Markdown 渲染，路线步骤和地点信息展示更美观。
    - 自动打开浏览器。
    """
    print("🌐 正在启动 Web 图形界面...")
    try:
        bot = init_agent_service()
        
        # [界面配置] 定义预设的建议问题，覆盖典型地图使用场景
        chatbot_config = {
            'prompt.suggestions': [
                '帮我规划上海一日游行程，主要想去外滩和迪士尼',
                '我在南京路步行街，帮我找一家评分高的本帮菜餐厅',
                '从浦东机场到外滩怎么走最方便？',
                '推荐上海三个适合拍照的网红景点',
                '帮我查找上海科技馆的具体地址和营业时间',
                '从徐家汇到外滩有哪些公交路线？',
                '现在在豫园，附近有什么好玩的地方推荐？',
                '帮我找一下静安寺附近的停车场',
                '上海野生动物园到迪士尼乐园怎么走？',
                '推荐陆家嘴附近的高档餐厅'
            ]
        }
        
        print("✅ Web 服务准备就绪。")
        print("👉 如果浏览器未自动打开，请访问: http://127.0.0.1:7860 (端口可能动态变化)")
        
        # [启动服务] 运行 WebUI
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
        
    except Exception as e:
        print(f"❌ 启动 Web 界面失败: {str(e)}")
        print("💡 常见原因：端口被占用、缺少 Node.js 环境、API Key 无效或未替换。")

# ==========================================
# 3. 程序入口
# ==========================================
if __name__ == '__main__':
    # [运行模式选择]
    # 取消注释下方对应行即可切换模式
    
    # test_mode()             # 模式 1: 单次测试 (适合调试 Key 和环境)
    # app_tui()               # 模式 2: 终端对话 (适合轻量使用)
    
    app_gui()                 # 模式 3: Web 图形界面 (默认推荐，体验最佳)
