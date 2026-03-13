#!/usr/bin/env python
# coding: utf-8
"""
===============================================================================
保险_VLM_批量图片分析
功能描述: 
    本脚本用于保险业务场景下的自动化视觉分析。它读取 Excel 文件中的理赔/审核指令
    和图片标识符，调用阿里云通义千问视觉大模型 (Qwen-VL) 进行多模态推理，
    并将分析结果自动回写到新的 Excel 文件中。

适用场景:
    - 车险定损识别 (受损部位、程度判断)
    - 医疗/财产单据信息提取与核对
    - 理赔材料真实性初审
    - 其他基于图片的保险业务自动化审核

执行流程:
    1. [初始化] 加载阿里云 DashScope API 客户端 (兼容 OpenAI 协议)。
    2. [读取数据] 从 'prompt_template_cn.xlsx' 读取提示词和图片标识符。
    3. [循环处理] 遍历每一行数据:
       a. 解析图片地址 (支持单图或多图列表，自动拼接 OSS 域名和后缀)。
       b. 构建多模态消息体 (Text + Image)。
       c. 调用 Qwen-VL-Max 模型获取推理结果。
       d. 捕获异常并记录，避免单点失败影响整体任务。
    4. [保存结果] 将包含 AI 回复的完整数据保存为 'prompt_template_cn_result.xlsx'。

前置条件:
    - 环境变量: 需设置 DASHSCOPE_API_KEY。
    - 依赖库: pip install openai pandas openpyxl
    - 输入文件: 当前目录下需存在 'prompt_template_cn.xlsx'，且包含 'prompt' 和 'image' 列。
    - 图片存储: 图片需存储在指定的阿里云 OSS Bucket 中，Excel 中仅需填写文件名(不含后缀)。
===============================================================================
"""

import os
import time
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ================= 配置常量 =================
# 模型名称：使用通义千问视觉最大版，适合复杂场景分析
MODEL_NAME = "qwen-vl-max" 
# 图片存储前缀：阿里云 OSS 地址 (上海区域)
OSS_BASE_URL = "https://vl-image.oss-cn-shanghai.aliyuncs.com/"
# 图片默认后缀
IMAGE_SUFFIX = ".jpg"
# 输入/输出文件名
INPUT_FILE = './prompt_template_cn.xlsx'
OUTPUT_FILE = './prompt_template_cn_result.xlsx'

# ================= 初始化客户端 =================
def init_client():
    """
    初始化 OpenAI 兼容模式的阿里云 DashScope 客户端。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("错误: 未找到环境变量 DASHSCOPE_API_KEY，请检查配置。")
    
    return OpenAI(
        api_key=api_key, 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

client = init_client()

# ================= 核心功能函数 =================
def parse_image_urls(image_input):
    """
    解析图片输入字符串，处理单张或多张图片的情况，并拼接完整的 OSS URL。
    
    参数:
        image_input (str): Excel 中的原始字符串。
                           格式示例 1: "claim_001" (单张)
                           格式示例 2: "[claim_001, claim_002]" (多张列表)
                           
    返回:
        list: 完整的图片 URL 列表。
    """
    image_url_list = []
    
    # 判断是否为列表格式 (以 '[' 开头且包含 ',')
    if isinstance(image_input, str) and image_input.strip().startswith('[') and ',' in image_input:
        # 去除方括号并分割
        clean_str = image_input.strip()[1:-1]
        raw_urls = clean_str.split(',')
        image_url_list = [url.strip() for url in raw_urls if url.strip()]
    else:
        # 单张图片情况，直接放入列表
        if image_input:
            image_url_list = [str(image_input).strip()]
    
    # 拼接完整的 OSS URL
    full_urls = []
    for name in image_url_list:
        # ✅ 正确移除后缀的方式
        clean_name = name
        if clean_name.endswith('.jpg'):
            clean_name = clean_name[:-4]
        elif clean_name.endswith('.png'):
            clean_name = clean_name[:-4]
        
        full_url = f"{OSS_BASE_URL}{clean_name}{IMAGE_SUFFIX}"
        full_urls.append(full_url)
        
    return full_urls

def get_vlm_response(user_prompt, image_input):
    """
    构建请求并调用 VLM 模型获取推理结果。
    
    参数:
        user_prompt (str): 用户提出的分析问题或指令。
        image_input (str): 图片标识符字符串。
        
    返回:
        str: 模型生成的文本回复。若出错则返回错误信息。
    """
    # 1. 解析并获取完整的图片 URL 列表
    image_url_list = parse_image_urls(image_input)
    print(f"图片地址列表: {image_url_list}")
    
    if not image_url_list:
        return "错误: 未解析到有效的图片地址。"

    # 2. 构建多模态消息内容 (Content Payload)
    # 格式遵循 OpenAI Chat Completion 多模态标准
    content_payload = [{"type": "text", "text": user_prompt}]
    
    for img_url in image_url_list:
        content_payload.append({
            "type": "image_url",
            "image_url": {"url": img_url}
        })

    messages = [{
        "role": "user",
        "content": content_payload
    }]

    # 调试信息：打印构建的消息结构 (生产环境可酌情关闭)
    # print(f"[DEBUG] Messages payload: {messages}")

    try:
        # 3. 调用 API
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            timeout=60  # 设置超时时间，防止大图长时间无响应
        )
        
        # 4. 提取返回结果
        if completion.choices and len(completion.choices) > 0:
            return completion.choices[0].message.content
        else:
            return "警告: API 返回为空，未生成内容。"
            
    except Exception as e:
        # 捕获网络错误、API 限流或鉴权失败等异常
        error_msg = f"API 调用失败: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return error_msg

# ================= 主执行流程 =================
def main():
    print(f"=== 开始执行保险 VLM 批量分析任务 ===")
    print(f"输入文件: {INPUT_FILE}")
    print(f"使用模型: {MODEL_NAME}")
    
    # 1. 读取 Excel 数据
    try:
        df = pd.read_excel(INPUT_FILE)
        # 确保必要的列存在
        if 'prompt' not in df.columns or 'image' not in df.columns:
            raise ValueError("Excel 文件中缺少 'prompt' 或 'image' 列。")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {INPUT_FILE}")
        return
    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        return

    # 初始化结果列
    df['response'] = ''
    total_rows = len(df)
    
    print(f"共检测到 {total_rows} 条数据，开始处理...\n")

    # 2. 逐行处理
    for index, row in df.iterrows():
        user_prompt = row['prompt']
        image_input = row['image']
        
        # 显示进度
        print(f"[{index + 1}/{total_rows}] 正在处理: {str(user_prompt)[:30]}... | 图片: {image_input}")
        
        # 调用核心函数获取结果
        response_text = get_vlm_response(user_prompt, image_input)
        
        # 更新 DataFrame
        df.loc[index, 'response'] = response_text
        
        # 简单延时，避免触发过于频繁的 API 限流 (可选)
        # time.sleep(0.5) 

    # 3. 保存结果
    try:
        df.to_excel(OUTPUT_FILE, index=False)
        print(f"\n=== 处理完成 ===")
        print(f"结果已保存至: {OUTPUT_FILE}")
    except Exception as e:
        print(f"保存文件失败: {e}")

if __name__ == "__main__":
    main()
