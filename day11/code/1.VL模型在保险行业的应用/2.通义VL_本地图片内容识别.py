#!/usr/bin/env python
# coding: utf-8
"""
===============================================================================
通义VL_本地图片内容识别
功能描述: 
    本脚本演示如何使用阿里云 DashScope 原生 SDK 调用通义千问视觉模型 (Qwen-VL)。
    它直接读取本地图片文件，发送给用户指定的提示词，让模型识别图片内容，
    并打印出结构化的识别结果。

适用场景:
    - 本地图片的快速内容验证
    - 单张图片的视觉问答 (VQA) 测试
    - 文档/票据/场景图的初步识别演示

执行流程:
    1. [环境检查] 验证 API Key 配置及本地图片文件是否存在。
    2. [构建请求] 构造符合多模态对话格式的消息体 (System + User(Image+Text))。
    3. [调用模型] 使用 dashscope.MultiModalConversation 接口调用 qwen-vl-plus 模型。
    4. [状态校验] 检查 API 响应状态码，确保请求成功。
    5. [解析输出] 从复杂的响应对象中提取最终的文本答案并打印。

前置条件:
    - 环境变量: 需设置 DASHSCOPE_API_KEY。
    - 依赖库: pip install dashscope
    - 输入文件: 当前目录下需存在名为 '1-Chinese-document-extraction.jpg' 的图片。
===============================================================================
"""

import os
import json
import dashscope
from dashscope import MultiModalConversation
from dashscope.api_entities.dashscope_response import Role
from dotenv import load_dotenv

load_dotenv()

# ================= 配置常量 =================
# 模型名称：通义千问视觉增强版，平衡速度与精度
MODEL_NAME = 'qwen-vl-plus'
# 本地图片路径 (使用 file:// 协议标识本地文件)
# 注意：请确保该文件在当前运行目录下存在
IMAGE_FILE_NAME = './images/1-Chinese-document-extraction.jpg'
LOCAL_IMAGE_PATH = f'file://{IMAGE_FILE_NAME}'
# 用户提示词
USER_PROMPT = '图片里有什么东西?'

# ================= 初始化配置 =================
def setup_environment():
    """
    设置 API Key 并检查基础环境。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("错误: 未找到环境变量 DASHSCOPE_API_KEY。\n请在终端执行: export DASHSCOPE_API_KEY='your_key'")
    
    dashscope.api_key = api_key
    print(f"[INFO] API Key 加载成功。")

    # 检查本地文件是否存在 (去除 file:// 前缀后检查)
    clean_path = LOCAL_IMAGE_PATH.replace('file://', '')
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"错误: 找不到本地图片文件 '{clean_path}'。\n请确保图片已放置在当前脚本运行目录下。")
    
    print(f"[INFO] 本地图片 '{IMAGE_FILE_NAME}' 检查通过。")

# ================= 主执行逻辑 =================
def main():
    try:
        # 1. 环境初始化
        setup_environment()
        
        print(f"\n=== 开始调用模型: {MODEL_NAME} ===")
        print(f"任务: 识别本地图片内容")
        print(f"提示词: {USER_PROMPT}\n")

        # 2. 构建多模态消息体
        # 格式遵循 DashScope 多模态对话标准
        messages = [
            {
                'role': 'system',
                'content': [{'text': 'You are a helpful assistant.'}]
            },
            {
                'role': 'user',
                'content': [
                    {'image': LOCAL_IMAGE_PATH},  # 加载本地图片
                    {'text': USER_PROMPT}         # 添加文本指令
                ]
            }
        ]

        # 3. 调用 API
        # MultiModalConversation.call 是原生 SDK 的核心入口
        response = MultiModalConversation.call(
            model=MODEL_NAME, 
            messages=messages
        )

        # 4. 检查响应状态
        # dashscope 响应包含 status_code, code, message 等字段
        if response.status_code != 200:
            print(f"[ERROR] API 调用失败!")
            print(f"状态码: {response.status_code}")
            print(f"错误代码: {response.code}")
            print(f"错误信息: {response.message}")
            return

        # 5. 解析并输出结果
        print("[DEBUG] 完整原始响应对象 (部分):")
        # 为了不让控制台太乱，这里只打印关键部分，如需全部可取消注释下面这行
        # print(response) 
        
        # 提取生成的文本内容
        # 结构通常为: response.output.choices[0].message.content[0]['text']
        if (response.output and 
            response.output.choices and 
            len(response.output.choices) > 0):
            
            message_content = response.output.choices[0].message.content
            
            # 兼容不同返回格式 (有时是直接文本，有时是列表)
            final_text = ""
            if isinstance(message_content, list) and len(message_content) > 0:
                final_text = message_content[0].get('text', '')
            elif isinstance(message_content, str):
                final_text = message_content
            else:
                final_text = "未能解析到文本内容，响应结构可能已变更。"

            print("\n" + "="*40)
            print("🤖 AI 识别结果:")
            print("="*40)
            print(final_text)
            print("="*40 + "\n")
            
        else:
            print("[WARNING] API 返回成功，但未找到具体的回答内容。")

    except Exception as e:
        # 捕获所有未预期的异常
        print(f"\n[CRITICAL ERROR] 程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
