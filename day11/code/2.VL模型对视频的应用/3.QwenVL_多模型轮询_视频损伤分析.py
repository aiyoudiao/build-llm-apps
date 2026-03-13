#!/usr/bin/env python
# coding: utf-8
"""
===============================================================================
DashScope_多模型轮询_视频损伤分析
功能描述: 
    基于阿里百炼 DashScope API，实现多模型自动轮询机制。
    当当前模型 Token 额度用尽或调用失败时，自动切换到列表中的下一个模型，
    最大化利用免费试用额度。

核心策略:
    1. 模型优先级池: 按推荐程度排序 (Turbo最新 -> Flash最新 -> 其他版本)。
    2. 自动故障转移: 捕获 QuotaExhausted 等错误，无缝切换下一模型。
    3. 上下文重建: 切换模型时，将历史问答摘要注入新 Prompt，保持对话连贯。
    4. 视频复用: 视频文件路径不变，每次请求重新上传/引用。

执行流程:
    1. 初始化模型列表。
    2. 遍历预设的问答任务列表。
    3. 对每个任务：
       - 尝试使用当前活跃模型调用 API。
       - 若成功：打印结果，更新对话历史，继续下一题。
       - 若失败(额度耗尽)：标记当前模型不可用，切换到下一个可用模型，重试当前问题。
       - 若所有模型均耗尽：终止程序。

前置条件:
    - 阿里云账号及 DashScope API Key (环境变量 DASHSCOPE_API_KEY)。
    - 安装依赖: pip install dashscope
    - 本地存在视频文件 './files/car.mp4'。
===============================================================================
"""

import os
import time
import dashscope
from dashscope import MultiModalConversation
from dotenv import load_dotenv

load_dotenv()


# 设置 API Key 
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# ================= 配置常量 =================

# 模型轮询池 (按推荐优先级排序)
# 策略：优先使用不同架构的模型 (Turbo vs Flash)，以确保证额度独立
MODEL_POOL = [
    # --- 第一梯队：最强理解力 (Turbo 系列) ---
    'qwen-omni-turbo-latest',             # 最新 Turbo
    'qwen-omni-turbo-realtime-latest',    # 实时 Turbo
    
    # --- 第二梯队：高性价比/高速 (Qwen3 Flash 系列) ---
    'qwen3-omni-flash-realtime',          # Qwen3 实时 Flash
    'qwen3-omni-flash',                   # Qwen3 标准 Flash
    
    # --- 第三梯队：特定日期版本 (作为备用，防止 latest 额度共享) ---
    'qwen-omni-turbo-2025-03-26',
    'qwen3-omni-flash-2025-09-15',
    'qwen-omni-turbo-realtime-2025-05-08',
    
    # --- 第四梯队：其他备选 ---
    'qwen2.5-omni-7b',                    # 经典开源版 API
    'qwen-omni-turbo',                    # 不带后缀的标准版
]

VIDEO_PATH = "./files/car.mp4"

# 定义任务队列 (模拟多轮对话)
# 每个任务包含：问题文本，以及是否需要重置上下文（模拟新话题）
TASKS = [
    {"question": "详细描述一下这个视频。", "reset_context": False, "lang": "cn"},
    {"question": "视频中出现了多少人？", "reset_context": False, "lang": "cn"},
    {"question": "车的哪个部位损伤了？", "reset_context": True, "lang": "cn"}, # 新话题
    {"question": "车撞到哪里了？", "reset_context": False, "lang": "cn"}      # 追问
]

class ModelRotator:
    def __init__(self, model_list):
        self.models = model_list
        self.disabled_models = set() # 记录已耗尽额度的模型
        self.current_model_index = 0

    def get_next_available_model(self):
        """获取下一个可用的模型"""
        while self.current_model_index < len(self.models):
            model_name = self.models[self.current_model_index]
            if model_name not in self.disabled_models:
                return model_name
            self.current_model_index += 1
        return None

    def disable_current_model(self):
        """标记当前模型为不可用，并移动索引"""
        current = self.models[self.current_model_index]
        print(f"\n[!] 模型 '{current}' 额度耗尽或不可用，已禁用。")
        self.disabled_models.add(current)
        self.current_model_index += 1

def run_analysis_with_rotation():
    print("="*70)
    print("启动多模型轮询视频分析任务")
    print(f"初始模型池: {len(MODEL_POOL)} 个模型")
    print("="*70)

    if not os.path.exists(VIDEO_PATH):
        print(f"错误: 视频文件不存在: {VIDEO_PATH}")
        return

    rotator = ModelRotator(MODEL_POOL)
    
    # 维护一个简单的文本历史用于上下文重建 (因为切换模型后无法继承云端 session)
    conversation_summary = [] 

    for task_idx, task in enumerate(TASKS):
        question = task["question"]
        reset_ctx = task["reset_context"]
        
        print(f"\n>>> 开始任务 {task_idx + 1}/{len(TASKS)}: {question}")
        
        if reset_ctx:
            print("    [系统] 检测到新话题，重置上下文历史。")
            conversation_summary = []
        
        # 构建当前请求的上下文提示
        # 如果切换了模型，我们将之前的问答摘要作为 System Prompt 或前缀传入
        context_prefix = ""
        if conversation_summary and not reset_ctx:
            context_prefix = "Previous conversation summary:\n" + "\n".join(conversation_summary) + "\n\nCurrent Question: "
        
        full_prompt = context_prefix + question

        success = False
        attempts = 0
        
        while not success:
            current_model = rotator.get_next_available_model()
            
            if not current_model:
                print("\n[ERROR] 所有模型额度均已耗尽，任务终止。")
                return

            print(f"    [尝试] 使用模型: {current_model} (尝试次数: {attempts + 1})")
            
            try:
                # 构建消息体
                # 注意：每次切换模型，都需要重新发送视频，因为新模型没有之前的记忆
                content_payload = [
                    {"video": VIDEO_PATH},
                    {"text": full_prompt}
                ]
                
                messages = [{"role": "user", "content": content_payload}]

                response = MultiModalConversation.call(
                    model=current_model,
                    messages=messages,
                    stream=False
                )
                
                if response.status_code == 200:
                    # 成功获取回答
                    answer_raw = response.output.choices[0].message.content
                    answer_text = ""
                    if isinstance(answer_raw, list):
                        answer_text = "".join([item.get("text", "") for item in answer_raw])
                    else:
                        answer_text = str(answer_raw)
                    
                    print(f"    [OK] 模型 {current_model} 响应成功:")
                    print(f"    >>> {answer_text[:200]}..." if len(answer_text) > 200 else f"    >>> {answer_text}")
                    
                    # 更新本地历史摘要 (只保留关键信息，节省下一轮 Token)
                    qa_pair = f"Q: {question}\nA: {answer_text}"
                    conversation_summary.append(qa_pair)
                    
                    # 限制摘要长度，防止超出新模型的上下文限制 (简单截断策略)
                    if len(conversation_summary) > 3:
                        conversation_summary = conversation_summary[-3:]
                    
                    success = True
                    break # 跳出重试循环，进入下一个任务
                    
                else:
                    # 处理 API 错误
                    error_code = response.code
                    error_msg = response.message
                    
                    print(f"    [失败] 模型 {current_model} 返回错误: {error_code} - {error_msg}")
                    
                    # 判断是否为额度耗尽或不可用
                    # 常见错误码: QuotaExhausted, InvalidApiKey, Arrearage (欠费), ResourceNotFound
                    if error_code in ['QuotaExhausted', 'Arrearage', 'ResourceNotFound'] or 'quota' in str(error_msg).lower():
                        rotator.disable_current_model()
                        attempts += 1
                        time.sleep(1) # 短暂等待后切换
                        continue
                    else:
                        # 其他错误（如参数错误），不切换模型，直接报错退出
                        print(f"    [严重错误] 非额度问题，停止切换。")
                        return

            except Exception as e:
                print(f"    [异常] 调用发生网络或本地异常: {e}")
                # 假设网络波动或临时不可用，也尝试切换模型
                rotator.disable_current_model()
                attempts += 1
                time.sleep(1)
                continue

    print("\n" + "="*70)
    print("所有任务执行完毕。")
    print("="*70)

if __name__ == "__main__":
    # 请确保环境变量已设置: export DASHSCOPE_API_KEY="sk-..."
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("警告: 未检测到 DASHSCOPE_API_KEY 环境变量，请设置后再运行。")
        # 可以在这里手动输入测试
        # key = input("请输入 API Key: ")
        # os.environ["DASHSCOPE_API_KEY"] = key
        
    run_analysis_with_rotation()
