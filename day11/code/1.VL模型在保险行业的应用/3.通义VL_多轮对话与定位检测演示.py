#!/usr/bin/env python
# coding: utf-8
"""
===============================================================================
文件名: 通义VL_多轮对话与定位检测演示.py
功能描述: 
    本脚本全面演示了通义千问视觉模型 (Qwen-VL) 的核心能力，包括：
    1. 云端 API 调用：使用 DashScope 接口进行图像内容识别。
    2. 多轮对话上下文：在连续对话中保持对图片的理解和记忆。
    3. 视觉定位 (Grounding)：识别物体并返回坐标，甚至在本地模式下直接绘制边界框。
    4. 场景化测试：涵盖汽车损伤检测（轮毂定位）和航空信息提取（OCR/问答）。

执行流程:
    1. [初始化] 检查环境变量，配置阿里云 DashScope 客户端。
    2. [云端演示 - 单轮识别] 调用 qwen-vl-max 识别基础图片内容。
    3. [云端演示 - 定位任务] 调用特定版本模型 (qwen-vl-max-2025-08-13) 请求物体坐标。
    4. [云端演示 - 多轮对话] 基于上一轮结果，切换模型 (qwen-vl-plus) 进行追问，验证上下文记忆。
    5. [本地演示 - (可选)] 展示如何在本地 GPU 环境加载开源模型，解析坐标标签并自动绘图保存。
       (注：本地部分默认注释，需安装 transformers 且拥有 GPU 资源方可运行)

核心特性:
    - 支持动态消息历史管理 (Message History Management)。
    - 兼容 OpenAI SDK 格式的阿里云接口。
    - 结构化输出解析，区分文本描述与坐标数据。

前置条件:
    - 环境变量: 需设置 DASHSCOPE_API_KEY。
    - 依赖库: pip install openai pandas (本地模式需: transformers, torch, torchvision, matplotlib)
    - 网络: 访问阿里云 DashScope 服务。
===============================================================================
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ================= 配置常量 =================
# 阿里云 DashScope 兼容模式 endpoint
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 默认使用的模型
DEFAULT_MODEL = "qwen-vl-max"
# 特定日期版本模型 (用于测试特定优化)
DATED_MODEL = "qwen-vl-max-2025-08-13"
# 轻量级模型 (用于多轮对话测试)
PLUS_MODEL = "qwen-vl-plus"

# 测试图片 URL
IMG_URL_DOG_GIRL = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
IMG_URL_CAR = "https://easycar.oss-cn-beijing.aliyuncs.com/car_undistorted.jpg"

# ================= 初始化客户端 =================
def init_client():
    """
    初始化 OpenAI 兼容模式的阿里云客户端。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "错误: 未找到环境变量 DASHSCOPE_API_KEY。\n"
            "请在终端执行: export DASHSCOPE_API_KEY='your_api_key'"
        )
    
    return OpenAI(
        api_key=api_key, 
        base_url=DASHSCOPE_BASE_URL
    )

# ================= 云端 API 功能模块 =================
def run_cloud_multimodal_chat(client):
    """
    演示云端多轮对话与视觉定位能力。
    包含三个步骤：
    1. 基础识别
    2. 物体定位 (获取坐标)
    3. 基于上下文的追问
    """
    print("\n=== [阶段 1] 基础图像识别 ===")
    try:
        response_1 = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "这是什么"},
                    {"type": "image_url", "image_url": {"url": IMG_URL_DOG_GIRL}}
                ]
            }]
        )
        print(f"模型回答: {response_1.choices[0].message.content}\n")
    except Exception as e:
        print(f"[阶段 1] 调用失败: {e}")

    print("=== [阶段 2] 视觉定位 (Grounding) ===")
    # 构建初始消息用于定位任务
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "框出图中轮毂的位置"},
                {"type": "image_url", "image_url": {"url": IMG_URL_CAR}}
            ]
        }
    ]
    
    try:
        response_2 = client.chat.completions.create(
            model=DATED_MODEL, # 使用特定日期版本以测试定位精度
            messages=messages
        )
        assistant_reply = response_2.choices[0].message.content
        print(f"定位结果: {assistant_reply}")
        # 注意：此时 messages 列表尚未包含助手回复，需在下一步手动添加以维持上下文
    except Exception as e:
        print(f"[阶段 2] 调用失败: {e}")
        return

    print("\n=== [阶段 3] 多轮对话上下文测试 ===")
    # 1. 将上一轮的助手回复加入历史记录
    messages.append({'role': 'assistant', 'content': assistant_reply})
    
    # 2. 添加新的用户提问 (同一张图，不同问题，测试上下文理解)
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "图中轮毂的位置在哪里？请用文字详细描述一下。"},
            {"type": "image_url", "image_url": {"url": IMG_URL_CAR}}
        ]
    })
    
    print(f"[DEBUG] 当前消息历史长度: {len(messages)}")
    
    try:
        # 切换为 plus 模型进行多轮对话测试
        response_3 = client.chat.completions.create(
            model=PLUS_MODEL,
            messages=messages
        )
        final_reply = response_3.choices[0].message.content
        print(f"多轮对话回答: {final_reply}")
        
        # 打印完整响应 JSON 供调试分析
        # print(json.dumps(response_3.model_dump(), indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"[阶段 3] 调用失败: {e}")

# ================= 本地部署功能模块 (可选) =================
def run_local_grounding_demo():
    """
    演示本地部署 Qwen-VL 并进行自动画框。
    此功能需要 GPU 环境和 transformers 库。
    默认注释，如需运行请取消注释并确保路径正确。
    """
    print("\n=== [可选阶段] 本地模型画框演示 ===")
    print("提示: 此部分代码已注释，需本地 GPU 环境支持。")
    
    """
    # 导入本地依赖
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # 配置本地模型路径 (需根据实际情况修改，例如 AutoDL 路径)
    LOCAL_MODEL_PATH = "/root/autodl-tmp/model/Qwen/Qwen-VL-Chat"
    
    print("正在加载本地模型...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, device_map="auto", trust_remote_code=True).eval()
    
    # 第一轮：识别
    query = tokenizer.from_list_format([
        {'image': IMG_URL_CAR},
        {'text': '这是什么?'},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    print(f"本地识别结果: {response}")
    
    # 第二轮：定位并画框
    query_box = "框出图中轮毂的位置"
    response_box, history = model.chat(tokenizer, query_box, history=history)
    print(f"本地定位结果: {response_box}")
    
    # 核心功能：解析坐标并在原图上画框
    # Qwen-VL 输出格式示例：<ref>轮毂</ref><box>(154,553),(310,880)</box>
    image = tokenizer.draw_bbox_on_latest_picture(response_box, history)
    
    if image:
        output_filename = 'wheel_detected.jpg'
        image.save(output_filename)
        print(f"成功绘制边界框并保存至: {output_filename}")
    else:
        print("未检测到有效的坐标框，无法绘图。")
    """

# ================= 主程序入口 =================
def main():
    print("启动通义 VL 多模态演示程序...")
    
    # 1. 初始化云端客户端
    try:
        client = init_client()
        print("[OK] 云端客户端初始化成功。")
    except Exception as e:
        print(f"[ERROR] 初始化失败: {e}")
        return

    # 2. 执行云端演示流程
    run_cloud_multimodal_chat(client)
    
    # 3. (可选) 执行本地演示流程
    # run_local_grounding_demo()
    
    print("\n=== 演示结束 ===")

if __name__ == "__main__":
    main()
