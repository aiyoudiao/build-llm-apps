#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本统计智能助手 (Intelligent Text Statistics Assistant)

【功能概述】
本模块是一个基于阿里云通义千问 (Qwen) 大模型和 Qwen-Agent 框架构建的智能应用入口。
它充当“用户”与“本地文本计数工具”之间的智能桥梁：
1. 理解用户的自然语言统计需求（如“统计这段代码的行数”、“计算中文字符数”）。
2. 自动调度本地的 MCP 服务 (../1.MCP_Demo/1.桌面TXT文件助手.py) 执行精确计算。
3. 利用大模型的推理能力，将原始数据整合为专业、易读的统计报告。

【系统架构】
- 交互层：提供 Web 图形界面 (GUI)、终端交互 (TUI) 和单次测试 (Test) 三种模式。
- 代理层 (Agent)：使用 qwen-max 模型作为大脑，负责意图识别、参数提取和结果润色。
- 工具层 (MCP)：通过 Model Context Protocol 动态调用本地脚本 `../1.MCP_Demo/1.桌面TXT文件助手.py` 获取精确数据。
  * 注意：大模型不直接进行数学计数，而是调用专用工具以确保 100% 准确性。

【核心流程】
1. [初始化] 读取环境变量中的 DASHSCOPE_API_KEY，配置 LLM 参数。
2. [注册工具] 定义 MCP 服务器配置，指定启动命令为 `python ../1.MCP_Demo/1.桌面TXT文件助手.py` (端口 6277)。
3. [实例化] 创建 Assistant 对象，注入系统提示词 (System Prompt) 和工具列表。
4. [交互循环] 
   - 接收用户输入 (文本内容 或 文件路径)。
   - Agent 分析意图 -> 决定调用 MCP 工具 -> 传递文本/文件。
   - 工具返回精确统计值 (字符数、单词数、行数等)。
   - Agent 结合数据生成自然语言回复 -> 返回给用户。

【依赖说明】
- 必须存在 `../1.MCP_Demo/1.桌面TXT文件助手.py` 文件（即文本计数服务端脚本）。
- 需要安装依赖：`pip install dashscope qwen-agent`。
- 需要配置环境变量：`DASHSCOPE_API_KEY`。

【运行模式】
- GUI 模式 (默认): 启动本地 Web 服务器，提供友好的聊天界面和丰富的预设统计场景。
- TUI 模式：在命令行中进行多轮对话，支持文件路径输入。
- Test 模式：执行单次查询用于调试。
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
# 若未找到环境变量，将为空字符串，导致后续调用失败（符合预期安全行为）
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
dashscope.timeout = 30  # 设置 HTTP 请求超时时间，防止长时间无响应阻塞

def init_agent_service() -> Assistant:
    """
    初始化文本统计助手服务
    
    此函数负责构建 Agent 的核心配置，包括模型参数、角色设定以及 MCP 工具桥接。
    
    Returns:
        Assistant: 配置完成并可立即使用的助手实例
        
    Raises:
        Exception: 当 API Key 缺失或助手初始化失败时抛出异常
    """
    # [LLM 配置] 使用通义千问最大模型 (qwen-max) 以获得最佳的意图理解和结果润色能力
    llm_cfg = {
        'model': 'qwen-max',
        'timeout': 30,      # 单次调用超时
        'retry_count': 3,   # 网络波动时的自动重试次数
    }
    
    # [系统角色设定] 定义 AI 的行为准则和专业领域
    # 强调其作为“文本分析专家”的身份，并引导其优先使用工具而非自行估算
    system_prompt = (
        '你扮演一个专业的文本分析与统计助手。你的核心任务是帮助用户精确统计文本的各项指标。\n'
        '支持的功能包括：字符数统计、单词数统计、行数统计、标点符号统计、空格统计等。\n'
        '重要原则：\n'
        '1. 必须优先调用“文本计数服务”工具来获取精确数据，严禁凭空估算或产生幻觉。\n'
        '2. 能够区分不同语言环境（如中文按字符计，英文按单词计）。\n'
        '3. 如果用户提供的是文件路径，应指导工具读取文件内容进行统计。\n'
        '4. 回答应结构清晰，列出详细的数据报表。'
    )
    
    # [MCP 工具配置] 定义如何启动和连接本地计数服务
    # 这里通过 command 和 args 动态启动 ../1.MCP_Demo/1.桌面TXT文件助手.py 脚本
    # port 6277 需确保未被其他程序占用（区别于其他助手的端口）
    tools_config = [{
        "mcpServers": {
            "桌面TXT文件助手": {
                "command": "python",
                "args": ["../1.MCP_Demo/1.桌面TXT文件助手.py"],  # 依赖计数服务脚本
                "port": 6277                 # 指定通信端口，避免冲突
            }
        }
    }]
    
    try:
        # [实例化助手] 组装所有配置创建 Assistant 对象
        bot = Assistant(
            llm=llm_cfg,
            name='文本统计智能助手',
            description='专注于文本数据统计与深度分析的智能工具',
            system_message=system_prompt,
            function_list=tools_config,
        )
        print("✅ 助手服务初始化成功！已加载 qwen-max 模型及 MCP 计数工具。")
        return bot
    except Exception as e:
        error_msg = f"❌ 助手初始化失败: {str(e)}"
        print(error_msg)
        # 提供具体的排查建议
        if "api_key" in str(e).lower() or not dashscope.api_key:
            print("💡 提示：请检查是否已设置环境变量 DASHSCOPE_API_KEY")
        if "../1.MCP_Demo/1.桌面TXT文件助手.py" in str(e):
            print("💡 提示：请确认当前目录下存在 ../1.MCP_Demo/1.桌面TXT文件助手.py 文件")
        raise

# ==========================================
# 2. 功能模式定义
# ==========================================

def test_mode(query: str = '帮我统计这段文本的字数：Hello World', file_path: Optional[str] = None):
    """
    测试模式：执行单次查询并打印结果
    
    适用于快速验证代码逻辑、API 连通性及工具调用是否正常。
    
    Args:
        query: 用户查询语句
        file_path: 可选的本地文件路径
    """
    try:
        print(f"🚀 进入测试模式，正在处理请求：'{query}'...")
        bot = init_agent_service()
        messages = []

        # [消息构建] 根据是否有文件附件构建不同格式的消息体
        if not file_path:
            messages.append({'role': 'user', 'content': query})
        else:
            # 支持同时发送文本指令和文件引用
            messages.append({'role': 'user', 'content': [{'text': query}, {'file': file_path}]})

        # [流式处理] 遍历生成器获取实时响应
        for response in bot.run(messages):
            # response 通常是一个列表，包含最新的消息状态
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
    特点：轻量级，无需浏览器，支持文件路径输入，适合批量处理或远程服务器。
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

                # [文件输入] 允许用户输入本地文件路径或 URL 进行统计
                file_input = input('📎 附加文件路径/URL (直接回车跳过): ').strip()
                
                # [构建消息]
                if not file_input:
                    conversation_history.append({'role': 'user', 'content': query})
                else:
                    conversation_history.append({'role': 'user', 'content': [{'text': query}, {'file': file_input}]})

                print("⏳ 正在分析并统计数据...")
                
                # [执行推理]
                # 注意：bot.run 返回的是生成器，每次 yield 最新的完整消息列表
                final_response = []
                for response_snapshot in bot.run(conversation_history):
                    final_response = response_snapshot
                
                # [更新历史] 将助手的回复加入上下文，以支持多轮对话
                if final_response:
                    conversation_history.extend(final_response)
                    # 提取并打印助手的最后一条回复
                    last_msg = final_response[-1]
                    if last_msg.get('role') == 'assistant':
                        print(f"\n🤖 助手:\n{last_msg['content']}")
                        
            except KeyboardInterrupt:
                print("\n⚠️  检测到中断信号，退出对话。")
                break
            except Exception as e:
                print(f"💥 对话处理出错: {str(e)}")
                print("💡 建议：检查网络连接、API Key 配置或 ../1.MCP_Demo/1.桌面TXT文件助手.py 文件是否存在。")
                
    except Exception as e:
        print(f"❌ 启动终端模式失败: {str(e)}")

def app_gui():
    """
    图形界面模式 (GUI)
    
    启动本地 Web 服务器，提供现代化的聊天界面。
    特点：
    - 预设丰富的统计场景建议 (Prompt Suggestions)，降低用户使用门槛。
    - 支持 Markdown 渲染，统计报表展示更美观。
    - 自动打开浏览器。
    """
    print("🌐 正在启动 Web 图形界面...")
    try:
        bot = init_agent_service()
        
        # [界面配置] 定义预设的建议问题，覆盖多种统计场景
        chatbot_config = {
            'prompt.suggestions': [
                '帮我统计这段文本的字数：Hello World',
                '计算下面这段文本的行数：\n第一行\n第二行\n第三行',
                '这段文本有多少个单词：The quick brown fox jumps over the lazy dog',
                '分析这段多语言文本：你好，世界！Hello World!',
                '统计下面文本的标点符号数量：Hello, World! How are you?',
                '帮我计算这段代码的行数：\ndef hello():\n    print("Hello")\n    return True',
                '这段文本中有多少个中文字符：你好，这是一段中文文本',
                '统计这段文本的空格数量：Hello  World   !',
                '计算这段文本的总字符数（包括空格和标点）',
                '分析这段文本的基本信息（字数、行数、单词数）'
            ]
        }
        
        print("✅ Web 服务准备就绪。")
        print("👉 如果浏览器未自动打开，请访问: http://127.0.0.1:7860 (端口可能动态变化)")
        
        # [启动服务] 运行 WebUI，此方法会阻塞主线程直到服务停止
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
        
    except Exception as e:
        print(f"❌ 启动 Web 界面失败: {str(e)}")
        print("💡 常见原因：端口被占用、缺少依赖库、API Key 无效或 ../1.MCP_Demo/1.桌面TXT文件助手.py 缺失。")

# ==========================================
# 3. 程序入口
# ==========================================
if __name__ == '__main__':
    # [运行模式选择]
    # 取消注释下方对应行即可切换模式
    
    # test_mode()             # 模式 1: 单次测试 (适合调试)
    # app_tui()               # 模式 2: 终端对话 (适合轻量使用/文件处理)
    
    app_gui()                 # 模式 3: Web 图形界面 (默认推荐，体验最佳)
