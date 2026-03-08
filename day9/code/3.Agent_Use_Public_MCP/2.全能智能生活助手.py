#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全能智能生活助手 (All-in-One Intelligent Life Assistant)

【功能概述】
本模块是一个基于阿里云通义千问 (Qwen) 大模型构建的高级智能代理应用。
它创新性地集成了“本地运行”与“云端服务”两种模式的 MCP 工具，具备三大核心能力：
1. 🗺️ 地理服务：通过本地高德地图 MCP 服务，提供精准的地点查询、路线规划和周边推荐。
2. 🔍 实时搜索：通过远程必应 (Bing) MCP 服务，获取互联网最新新闻、资讯和数据。
3. 🌐 网页阅读：通过远程 Fetch MCP 服务，提取指定 URL 的网页内容并转换为 Markdown，支持长文总结。

【系统架构】
- 交互层：提供 Web 图形界面 (GUI)、终端交互 (TUI) 和单次测试 (Test) 三种模式。
- 代理层 (Agent)：使用 qwen-max 模型作为“大脑”，负责意图识别与工具路由。
  * 自动判断用户意图：是需要查地图？搜新闻？还是读网页？
- 工具层 (Hybrid MCP)：
  * 本地工具：高德地图 (依赖 Node.js 环境，数据隐私性高)。
  * 远程工具：必应搜索 & 网页抓取 (依赖 ModelScope 云端服务，无需本地部署)。

【核心流程】
1. [初始化] 读取 DashScope API Key，校验本地 Node.js 环境。
2. [注册工具] 
   - 启动本地 `npx` 进程运行高德服务。
   - 配置 SSE 连接指向 ModelScope 托管的搜索与抓取服务。
3. [意图路由] 
   - 用户提问 -> Agent 分析 -> 选择最佳工具 (或组合使用)。
   - 例：“查一下去外滩的路并看看今天天气新闻” -> 同时调用高德 + 必应。
4. [结果合成] Agent 整合多源数据，生成自然、结构化的最终回复。

【依赖说明】
- 环境依赖：必须安装 Node.js (用于运行高德本地服务)。
- Python 依赖：`pip install dashscope qwen-agent`。
- 密钥配置 (需准备两个平台的 Key)：
  1. DASHSCOPE_API_KEY (环境变量): 阿里云大模型调用权限。
  2. AMAP_MAPS_API_KEY (代码配置): 高德开放平台地图服务权限。
  3. MODEL_SCOPE_FETCH_API_KEY、MODEL_SCOPE_BING_API_KEY (代码配置): 魔搭社区远程 MCP 服务权限。

【运行模式】
- GUI 模式 (默认): 启动 Web 界面，提供跨领域的预设场景（导航、搜索、阅读）。
- TUI 模式：命令行多轮对话。
- Test 模式：单次功能验证。
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

# [安全提示] 从环境变量读取阿里云 API Key
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
dashscope.timeout = 30  # 设置 HTTP 请求超时时间

def init_agent_service() -> Assistant:
    """
    初始化全能智能助手服务
    
    此函数配置了混合架构的 MCP 工具链：
    1. 本地高德地图服务 (Node.js)
    2. 远程必应搜索服务 (ModelScope SSE)
    3. 远程网页抓取服务 (ModelScope SSE)
    
    Returns:
        Assistant: 配置完成并可立即使用的助手实例
        
    Raises:
        Exception: 当环境缺失或 Key 配置错误时抛出异常
    """
    # [LLM 配置] 使用 qwen-max 以处理复杂的多工具协同任务
    llm_cfg = {
        'model': 'qwen-max',
        'timeout': 30,
        'retry_count': 3,
    }
    
    # [系统角色设定] 定义 AI 的全能助手身份
    system_prompt = (
        '你是一个全能的智能生活助手，集成了地图导航、实时搜索和网页阅读三大能力。\n'
        '你的职责是根据用户意图，灵活调用以下工具：\n'
        '1. 【高德地图】：处理地点查询、路线规划、周边推荐等地理相关问题。\n'
        '2. 【必应搜索】：处理需要最新信息的查询，如新闻、天气、股市、突发事件等。\n'
        '3. 【网页抓取】：处理用户提供的 URL，提取正文内容并进行总结或格式转换。\n'
        '重要原则：\n'
        '- 优先使用工具获取真实数据，严禁编造信息。\n'
        '- 如果用户问题涉及多个领域（如“去某地的路线及当地新闻”），请并行或串行调用多个工具。\n'
        '- 回答应条理清晰，来源明确。'
    )
    
    # 1. 高德地图 Key (来自高德开放平台)
    AMAP_KEY = os.getenv('AMAP_MAPS_API_KEY', '')
    
    # 2. ModelScope Key (来自魔搭社区个人中心 Access Token)
    # 用于访问远程托管的 Bing 搜索和 Fetch 服务

    MODEL_SCOPE_FETCH_API_KEY = os.getenv('MODEL_SCOPE_FETCH_API_KEY', '')
    MODEL_SCOPE_BING_API_KEY = os.getenv('MODEL_SCOPE_BING_API_KEY', '')
    
    # [MCP 工具配置] 混合架构定义
    tools_config = [{
        "mcpServers": {
            # --- 工具 A: 高德地图 (本地运行) ---
            # 依赖 Node.js 环境，通过 npx 动态启动官方服务包
            "amap-maps": {
                "command": "npx",
                "args": [
                    "-y",
                    "@amap/amap-maps-mcp-server"
                ],
                "env": {
                    "AMAP_MAPS_API_KEY": AMAP_KEY
                }
            },
            
            # --- 工具 B: 网页抓取 (远程 SSE) ---
            # 通过 Server-Sent Events 连接 ModelScope 托管服务
            "fetch": {
                "type": "sse",
                # 注意：URL 中的 Key 需替换为 ModelScope Access Token
                "url": f"https://mcp.api-inference.modelscope.net/{MODEL_SCOPE_FETCH_API_KEY}/sse"
            },
            
            # --- 工具 C: 必应搜索 (远程 SSE) ---
            # 同样连接 ModelScope 托管的 Bing 服务
            "bing-cn-mcp-server": {
                "type": "sse",
                "url": f"https://mcp.api-inference.modelscope.net/{MODEL_SCOPE_BING_API_KEY}/sse"
            }
        }
    }]

    try:
        # [实例化助手]
        bot = Assistant(
            llm=llm_cfg,
            name='全能智能生活助手',
            description='集地图导航、实时搜索、网页阅读于一体的超级助理',
            system_message=system_prompt,
            function_list=tools_config,
        )
        print("✅ 助手服务初始化成功！已加载：高德地图 (本地), 必应搜索 (远程), 网页抓取 (远程)。")
        return bot
    except Exception as e:
        error_msg = f"❌ 助手初始化失败: {str(e)}"
        print(error_msg)
        # 针对性排查建议
        if "api_key" in str(e).lower() or not dashscope.api_key:
            print("💡 提示：请检查环境变量 DASHSCOPE_API_KEY 是否已设置。")
        if "npx" in str(e) or "node" in str(e).lower():
            print("💡 提示：请确认系统已安装 Node.js (包含 npx 命令)。")
        if "401" in str(e) or "Unauthorized" in str(e):
            print("💡 提示：API Key 无效，请检查高德 Key 或 ModelScope Token 是否正确。")
        raise

# ==========================================
# 2. 功能模式定义
# ==========================================

def test_mode(query: str = '帮我查找上海东方明珠的具体位置', file_path: Optional[str] = None):
    """
    测试模式：执行单次查询
    
    适用于验证三种工具（地图、搜索、抓取）的连通性。
    
    Args:
        query: 用户查询语句
        file_path: 可选的文件路径
    """
    try:
        print(f"🚀 进入测试模式，正在处理：'{query}'...")
        bot = init_agent_service()
        messages = []

        if not file_path:
            messages.append({'role': 'user', 'content': query})
        else:
            messages.append({'role': 'user', 'content': [{'text': query}, {'file': file_path}]})

        for response in bot.run(messages):
            if response:
                last_msg = response[-1]
                if last_msg.get('role') == 'assistant':
                    print(f"\n🤖 助手回答:\n{last_msg['content']}")
                    
    except Exception as e:
        print(f"💥 测试失败: {str(e)}")

def app_tui():
    """
    终端交互模式 (TUI)
    
    支持多轮对话，可连续测试不同领域的功能。
    """
    print("🖥️  正在启动终端交互模式... (输入 'quit' 退出)")
    try:
        bot = init_agent_service()
        conversation_history: List[Dict[str, Any]] = []

        while True:
            try:
                query = input('\n👤 用户提问: ').strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见！")
                    break
                
                if not query:
                    continue

                file_input = input('📎 附加文件/URL (直接回车跳过): ').strip()
                
                if not file_input:
                    conversation_history.append({'role': 'user', 'content': query})
                else:
                    conversation_history.append({'role': 'user', 'content': [{'text': query}, {'file': file_input}]})

                print("⏳ 正在调用工具并分析...")
                
                final_response = []
                for response_snapshot in bot.run(conversation_history):
                    final_response = response_snapshot
                
                if final_response:
                    conversation_history.extend(final_response)
                    last_msg = final_response[-1]
                    if last_msg.get('role') == 'assistant':
                        print(f"\n🤖 助手:\n{last_msg['content']}")
                        
            except KeyboardInterrupt:
                print("\n⚠️  中断退出。")
                break
            except Exception as e:
                print(f"💥 出错: {str(e)}")
                
    except Exception as e:
        print(f"❌ 启动失败: {str(e)}")

def app_gui():
    """
    图形界面模式 (GUI)
    
    提供可视化界面，预设了涵盖三大核心功能的场景建议。
    """
    print("🌐 正在启动 Web 图形界面...")
    try:
        bot = init_agent_service()
        
        # [界面配置] 预设建议覆盖：网页读取、地图导航、实时搜索
        chatbot_config = {
            'prompt.suggestions': [
                '将 https://k.sina.com.cn/article_7732457677_1cce3f0cd01901eeeq.html 网页转化为Markdown格式',
                '帮我找一下静安寺附近的停车场',
                '推荐陆家嘴附近的高档餐厅',
                '帮我搜索一下关于AI的最新新闻',
                '从浦东机场到外滩怎么走最方便？',
                '总结一下这篇技术博客的内容：https://example.com/blog',
                '今天北京的交通状况如何？',
                '帮我查找上海科技馆的营业时间'
            ]
        }
        
        print("✅ Web 服务准备就绪。")
        print("👉 浏览器未自动打开请访问: http://127.0.0.1:7860")
        
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
        
    except Exception as e:
        print(f"❌ 启动 Web 界面失败: {str(e)}")
        print("💡 请检查：Node.js 环境、高德 Key、ModelScope Token 及网络连接。")

# ==========================================
# 3. 程序入口
# ==========================================
if __name__ == '__main__':
    # [运行模式选择]
    
    # test_mode()             # 模式 1: 单次测试
    # app_tui()               # 模式 2: 终端对话
    
    app_gui()                 # 模式 3: Web 图形界面 (默认推荐)
