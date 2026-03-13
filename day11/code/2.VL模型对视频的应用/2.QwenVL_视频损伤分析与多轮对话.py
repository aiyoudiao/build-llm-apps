#!/usr/bin/env python
# coding: utf-8
"""
===============================================================================
DashScope_视频损伤分析与多轮对话
功能描述: 
    基于阿里百炼 DashScope API (qwen-omni-turbo-latest) 实现的视频分析脚本。
    替代了原本本地运行的 InternVL 2.5 模型，利用云端强大算力进行车损分析。

核心特性:
    1. 云端原生视频理解: 直接上传视频文件，无需本地抽帧和预处理。
    2. 极简部署: 无需 GPU，无需下载几十 GB 的模型权重。
    3. 弹性伸缩: 利用云端算力，轻松处理长视频和高并发。
    4. 多轮对话支持: 利用 DashScope 的 session 机制或手动维护历史。

执行流程:
    1. [配置] 设置 API Key 和模型名称。
    2. [上传] 将本地视频上传至 DashScope (或直接使用 OSS URL)。
    3. [交互] 
       - 构建包含视频内容的 Message。
       - 发送第一轮请求：详细描述。
       - 发送后续请求：携带历史上下文进行追问。
    4. [输出] 打印结果。

前置条件:
    - 阿里云账号及 DashScope API Key。
    - 安装依赖: pip install dashscope
    - 网络：需能访问阿里云 API 服务。
===============================================================================
"""

import os
import dashscope
from dashscope import MultiModalConversation
from dotenv import load_dotenv

load_dotenv()

# ================= 配置常量 =================
# 设置 API Key 
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# 推荐模型：qwen-omni-turbo-latest (综合能力最强)
# 备选：qwen-omni-turbo-realtime-latest (低延迟), qwen3-omni-flash-realtime (高性价比)
MODEL_NAME = 'qwen-omni-turbo-latest'

# 输入视频路径 (支持本地路径，SDK 会自动处理上传; 也支持 http/oss 链接)
VIDEO_PATH = "./files/car.mp4"

def run_video_analysis():
    print("="*70)
    print(f"启动 DashScope 视频分析任务 (模型: {MODEL_NAME})")
    print("="*70)

    if not os.path.exists(VIDEO_PATH):
        print(f"错误: 视频文件不存在: {VIDEO_PATH}")
        return

    # 初始化消息历史
    # DashScope 的多模态对话通常需要在一个 messages 列表中维护上下文
    messages = []

    # ================= 第一轮：详细视频描述 =================
    print("\n[Q1] 问题: Describe this video in detail.")
    
    # 构建第一轮消息
    # 注意：视频可以直接以文件路径形式传入，dashscope 会自动处理
    content_list = [
        {"video": VIDEO_PATH}, 
        {"text": "Describe this video in detail."}
    ]
    
    messages.append({"role": "user", "content": content_list})

    try:
        response = MultiModalConversation.call(
            model=MODEL_NAME,
            messages=messages,
            stream=False # 如果需要流式输出，可设为 True
        )
        
        if response.status_code == 200:
            answer1 = response.output.choices[0].message.content
            # 兼容不同返回格式，有时 content 是列表，有时是字符串
            if isinstance(answer1, list):
                answer1_text = "".join([item.get("text", "") for item in answer1])
            else:
                answer1_text = str(answer1)
                
            print(f"[A1] 回答:\n{answer1_text}\n")
            
            # 将助手回答加入历史，用于多轮对话
            messages.append({"role": "assistant", "content": [{"text": answer1_text}]})
        else:
            print(f"[ERROR] 请求失败: {response.code}, {response.message}")
            return

    except Exception as e:
        print(f"[ERROR] 发生异常: {e}")
        return

    # ================= 第二轮：计数任务 (多轮上下文) =================
    print("[Q2] 问题: How many people appear in the video?")
    
    # 只需要发送新的文本问题，模型会自动结合历史中的视频信息
    messages.append({"role": "user", "content": [{"text": "How many people appear in the video?"}]})

    try:
        response = MultiModalConversation.call(
            model=MODEL_NAME,
            messages=messages
        )
        
        if response.status_code == 200:
            answer2 = response.output.choices[0].message.content
            if isinstance(answer2, list):
                answer2_text = "".join([item.get("text", "") for item in answer2])
            else:
                answer2_text = str(answer2)
            print(f"[A2] 回答:\n{answer2_text}\n")
            
            messages.append({"role": "assistant", "content": [{"text": answer2_text}]})
        else:
            print(f"[ERROR] 请求失败: {response.code}, {response.message}")

    except Exception as e:
        print(f"[ERROR] 发生异常: {e}")

    # ================= 第三轮：损伤检测 (中文) =================
    # 注意：如果想开启一个新话题而不受前两轮英文影响，可以清空 messages 
    # 但保留视频引用（重新构造一个只包含视频和新问题的 messages）
    print("[Q3] 问题: 车的哪个部位损伤了？")
    
    # 策略：为了更精准的中文回答，我们可以选择重置上下文，只保留视频
    # 或者继续使用当前上下文。这里演示【重置上下文但保留视频】的策略，模拟新会话
    new_session_messages = [
        {"role": "user", "content": [
            {"video": VIDEO_PATH}, # 再次引用视频（API 允许，或者如果支持 session ID 更好，这里简化处理）
            {"text": "车的哪个部位损伤了？"}
        ]}
    ]
    
    # 注意：如果视频很大，重复上传可能会消耗流量/时间。
    # 优化方案：如果 API 支持 long context 且未超出限制，直接沿用上面的 messages 并追加中文问题也是可以的。
    # 这里为了演示“新话题”，我们假设直接追加到原对话流中通常也是有效的，
    # 但为了模拟原代码的“重置 history”，我们构造一个新的起始消息。
    # *实际使用中，建议直接沿用 messages 列表，只需追加 text 即可，因为视频已经在历史里了*
    
    # 修正策略：直接沿用 messages 列表追加中文问题，这样最省资源且符合多轮逻辑
    # 如果必须“重置”，则需要重新上传视频。这里演示直接追加（连续对话）。
    # 如果用户希望完全独立，请取消注释下面的 new_session_messages 逻辑并替换 response 调用。
    
    # 这里采用【直接追加】方式，因为视频已在上下文中
    messages.append({"role": "user", "content": [{"text": "车的哪个部位损伤了？"}]})

    try:
        response = MultiModalConversation.call(
            model=MODEL_NAME,
            messages=messages
        )
        
        if response.status_code == 200:
            answer3 = response.output.choices[0].message.content
            if isinstance(answer3, list):
                answer3_text = "".join([item.get("text", "") for item in answer3])
            else:
                answer3_text = str(answer3)
            print(f"[A3] 回答:\n{answer3_text}\n")
            messages.append({"role": "assistant", "content": [{"text": answer3_text}]})
        else:
            print(f"[ERROR] 请求失败: {response.code}, {response.message}")

    except Exception as e:
        print(f"[ERROR] 发生异常: {e}")

    # ================= 第四轮：碰撞位置追问 =================
    print("[Q4] 问题: 车撞到哪里了？ (基于上一轮上下文)")
    
    messages.append({"role": "user", "content": [{"text": "车撞到哪里了？"}]})

    try:
        response = MultiModalConversation.call(
            model=MODEL_NAME,
            messages=messages
        )
        
        if response.status_code == 200:
            answer4 = response.output.choices[0].message.content
            if isinstance(answer4, list):
                answer4_text = "".join([item.get("text", "") for item in answer4])
            else:
                answer4_text = str(answer4)
            print(f"[A4] 回答:\n{answer4_text}\n")
        else:
            print(f"[ERROR] 请求失败: {response.code}, {response.message}")

    except Exception as e:
        print(f"[ERROR] 发生异常: {e}")

    print("="*70)
    print("所有任务执行完毕。")
    print("="*70)

if __name__ == "__main__":
    run_video_analysis()
