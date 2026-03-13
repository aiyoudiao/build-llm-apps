#!/usr/bin/env python
# coding: utf-8
"""
===============================================================================
InternVL2.5_视频损伤分析与多轮对话
功能描述: 
    本脚本基于 InternVL 2.5 (InternVideo2_5_Chat_8B) 多模态大模型，
    专为汽车保险定损、交通事故分析及安防监控场景设计。
    它能够对输入视频进行深度语义理解，不仅生成详细的视频描述，
    还能精准定位车辆损伤部位、分析碰撞过程，并支持中英文混合的多轮交互式问答。

核心特性:
    1. 智能模型生命周期管理 (Smart Model Lifecycle Management):
       - 自动检测本地缓存，若模型缺失则通过 ModelScope 自动触发下载。
       - 支持断点续传逻辑（依赖 modelscope 库），避免重复下载大文件。
       - 灵活配置缓存根目录，适应不同的存储挂载点（如 AutoDL /root/autodl-tmp）。
    
    2. 动态视觉分块技术 (Dynamic Visual Preprocessing):
       - 摒弃传统的强制缩放，采用 InternVL 特有的动态分块策略。
       - 根据每一帧视频的原始宽高比，自适应计算最佳切割网格 (Grid)。
       - 将高分辨率视频帧切割为多个 448x448 的图像块 (Patches)，并可选附加全局缩略图。
       - 显著提升对细小划痕、裂纹、车牌文字等小目标的识别精度。
    
    3. 自适应时空采样策略 (Adaptive Spatio-Temporal Sampling):
       - 基于 Decord 后端的高效视频解码，支持多线程读取。
       - 采用均匀分段采样算法，将视频时长等分为 N 段，提取每段中间帧。
       - 有效覆盖视频全貌，避免关键动作遗漏，同时控制显存占用在可控范围。
    
    4. 深度多轮上下文记忆 (Deep Multi-turn Context Memory):
       - 维护完整的对话历史状态 (Chat History)，支持跨轮次指代消解。
       - 支持“中英混合”提问：可用英文询问整体概况，随即用中文追问细节。
       - 针对特定场景（如车损）优化了 Prompt 构建方式，区分“新话题”与“追问”。
    
    5. 高精度低显存推理 (High-Precision Low-Memory Inference):
       - 默认启用 BFloat16 (BF16) 精度加载模型，相比 Float32 节省 50% 显存。
       - 针对 NVIDIA Ampere (A10/A100) 及 Hopper (H100) 架构显卡优化。
       - 推理结束后自动执行 `torch.cuda.empty_cache()`，防止显存泄漏。

执行流程:
    1. [环境初始化]
       - 检查 CUDA 设备可用性。
       - 验证输入视频文件路径是否存在。
       - 调用 `ensure_model_downloaded` 检测模型完整性，缺失则自动下载。
    
    2. [模型加载与量化]
       - 使用 `AutoTokenizer` 和 `AutoModel` 加载模型权重。
       - 将模型转换为 BF16 精度并迁移至 GPU，设置为 eval 模式。
    
    3. [视频流预处理]
       - 使用 Decord 读取视频元数据 (FPS, 总帧数)。
       - 计算采样索引，提取关键帧数组。
       - 对每一帧执行：RGB 转换 -> 动态分块切割 -> 双三次插值缩放 -> Tensor 归一化。
       - 将所有帧的图像块拼接为单一的大 Tensor，并记录每帧的块数列表 (`num_patches_list`)。
    
    4. [多轮推理交互]
       - 构建 Prompt 前缀：将每一帧标记为 `Frame1: <image>\n`... 注入上下文。
       - Round 1 (全局理解): 发送“详细描述视频”指令，获取初始回答和历史状态。
       - Round 2 (细节计数): 基于 Round 1 的历史，追问“视频中有多少人”，测试上下文保持。
       - Round 3 (领域任务): 重置历史或新建会话，发送中文指令“车的哪个部位损伤了？”。
       - Round 4 (逻辑推理): 基于 Round 3 的历史，追问“车撞到哪里了？”，测试因果推断。
    
    5. [结果输出与清理]
       - 格式化打印每一轮的问答结果。
       - 显式释放 GPU 显存，结束进程。

前置条件:
    - 硬件环境:
      * GPU: NVIDIA 显卡，推荐显存 >= 24GB (RTX 3090/4090, A10, A100 等)。
             8B 模型加载约需 16-18GB 显存，长视频预处理需额外显存。
      * CPU: 建议 4 核以上，用于视频解码和数据预处理。
      * 内存: 系统内存 >= 32GB，防止视频加载时 OOM。
    
    - 软件依赖:
      * Python >= 3.8
      * PyTorch >= 2.0 (支持 BFloat16)
      * 必备库: pip install torch torchvision modelscope decord pillow numpy transformers
      * 系统库: ffmpeg (Decord 依赖，通常 conda 环境会自动安装)
    
    - 模型数据:
      * 首次运行需联网，自动下载 'OpenGVLab/InternVideo2_5_Chat_8B' (约 15-20GB)。
      * 也可手动下载后放置于配置的 `CACHE_DIR` 目录下。
    
    - 输入数据:
      * 当前工作目录下需存在名为 './files/car.mp4' 的视频文件。
      * 或在代码配置区修改 `VIDEO_PATH` 指向绝对路径。
      * 支持常见视频格式：.mp4, .avi, .mov, .mkv 等 (取决于 Decord 支持)。
===============================================================================
"""

import os
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from modelscope import snapshot_download, AutoModel, AutoTokenizer

# ================= 配置常量 =================
# 模型标识符 (ModelScope ID)
MODEL_ID = 'OpenGVLab/InternVideo2_5_Chat_8B'

# 模型缓存根目录 (可根据实际机器路径修改，例如 '/root/autodl-tmp/models')
CACHE_DIR = '/root/autodl-tmp/models'

# 计算出的完整模型路径
MODEL_PATH = os.path.join(CACHE_DIR, MODEL_ID)

# 输入视频路径 (请确保该文件存在，或修改为绝对路径)
VIDEO_PATH = "./files/car.mp4"

# 视频处理参数
NUM_SEGMENTS = 128          # 采样帧数：将视频分为多少段取帧 (越多越细，显存占用越大)
MAX_NUM_PATCHES = 1         # 每帧最大分块数：1表示不分块，>1表示根据宽高比动态分块
INPUT_SIZE = 448            # 模型输入分辨率 (InternVL 标准尺寸)
USE_BF16 = True             # 是否使用 BFloat16 精度 (True 可显著节省显存)

# 生成配置 (Greedy Search 以保证结果稳定性)
GENERATION_CONFIG = dict(
    do_sample=False,
    temperature=0.0,
    max_new_tokens=1024,
    top_p=0.1,
    num_beams=1
)

# ImageNet 标准化参数 (模型训练时的统计值，不可更改)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ================= 模型管理函数 =================

def ensure_model_downloaded(model_id, cache_dir):
    """
    智能模型检查与下载器。
    如果模型已存在则跳过，否则调用 ModelScope 下载。
    """
    target_path = os.path.join(cache_dir, model_id)
    config_file = os.path.join(target_path, "config.json")
    
    if os.path.exists(config_file):
        print(f"[OK] 模型已存在: {target_path}")
        print("     检测到完整配置文件，跳过下载步骤。")
        return target_path
    else:
        print(f"[INFO] 未在本地找到模型: {target_path}")
        print(f"     正在从 ModelScope 云端下载 '{model_id}' ...")
        print("     (提示：模型较大，首次下载可能需要数分钟，请保持网络畅通)\n")
        
        try:
            downloaded_path = snapshot_download(
                model_id, 
                cache_dir=cache_dir
            )
            print(f"\n[OK] 模型下载成功: {downloaded_path}")
            return downloaded_path
        except Exception as e:
            print(f"[ERROR] 模型下载失败: {e}")
            print("     可能原因：网络中断、磁盘空间不足或 ModelScope 账号权限问题。")
            raise e

# ================= 工具函数 =================

def build_transform(input_size):
    """
    构建图像预处理流水线。
    包含：RGB 强制转换 -> 双三次插值缩放 -> 转 Tensor -> 标准化。
    """
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    在候选比例集中寻找最接近原始图像宽高比的比例。
    算法目标：最小化切割时的形变和信息损失。
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        
        # 优先选择差异最小的
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # 若差异相同，选择面积利用率更高的（保留更多像素）
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """
    动态预处理核心逻辑 (InternVL 特色)。
    根据图像宽高比，将图像切割成 N 个小块 (Patches)，并可选添加全局缩略图。
    这使得模型既能看清全局，又能通过高分辨率小块识别细微特征（如车漆划痕）。
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # 生成候选比例集 (例如：1x1, 1x2, 2x1, 2x2, ..., 直到 max_num)
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) 
        for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # 寻找最佳匹配比例
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # 计算切割后的总块数
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]

    # 调整图像大小到目标尺寸，以便切割
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    
    # 按网格顺序切割图像
    for i in range(blocks):
        # 计算当前块的坐标 (left, upper, right, lower)
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    assert len(processed_images) == blocks
    
    # 可选：添加缩略图以保留全局上下文信息，防止因切割丢失整体结构感
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
        
    return processed_images

def get_frame_indices(bound, fps, max_frame, first_idx=0, num_segments=32):
    """
    计算需要采样的帧索引列表。
    策略：将视频时间段均匀分为 num_segments 段，取每段的中间帧。
    这比固定间隔采样更能保证覆盖视频的起止内容。
    """
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
        
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    
    seg_size = float(end_idx - start_idx) / num_segments
    # 计算每段中间点的索引
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) 
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video_pipeline(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """
    完整的视频加载与预处理流程。
    返回: 
        pixel_values: 所有帧的所有图像块拼接后的 Tensor (Total_Patches, C, H, W)
        num_patches_list: 列表，记录每一帧包含多少个图像块 (用于模型解析边界)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    
    print(f"视频信息: 总帧数={max_frame+1}, FPS={fps:.2f}, 计划采样帧数={num_segments}")
    
    frame_indices = get_frame_indices(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    pixel_values_list = []
    num_patches_list = []
    transform = build_transform(input_size=input_size)
    
    for idx, frame_index in enumerate(frame_indices):
        # 读取帧并转为 PIL Image
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        
        # 动态分块处理
        img_blocks = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        
        # 转换每个块为 Tensor
        pixel_values = [transform(tile) for tile in img_blocks]
        pixel_values = torch.stack(pixel_values)
        
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
        
        # 进度提示
        if (idx + 1) % 20 == 0:
            print(f"  ...已处理 {idx+1}/{len(frame_indices)} 帧")
            
    # 拼接所有帧的数据为一个大的 Tensor
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

# ================= 主执行流程 =================

def main():
    print("="*70)
    print("启动 InternVL 2.5 视频分析任务")
    print("="*70)
    
    # 1. 检查设备
    if not torch.cuda.is_available():
        print("错误: 未检测到 CUDA 设备，此模型需要 GPU 运行。")
        return
    
    # 2. 确保模型已下载
    try:
        final_model_path = ensure_model_downloaded(MODEL_ID, CACHE_DIR)
    except Exception:
        print("终止执行：模型准备失败。")
        return

    # 3. 加载模型
    print(f"\n正在加载模型: {final_model_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(final_model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            final_model_path, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16
        ).cuda().eval()
        print("[OK] 模型加载成功。")
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")
        return

    # 4. 加载并预处理视频
    print(f"\n正在处理视频: {VIDEO_PATH} ...")
    try:
        pixel_values, num_patches_list = load_video_pipeline(
            VIDEO_PATH, 
            num_segments=NUM_SEGMENTS, 
            max_num=MAX_NUM_PATCHES,
            input_size=INPUT_SIZE
        )
        # 转移数据到 GPU 并转换精度
        pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
        print(f"[OK] 视频处理完成。总 Tensor 形状: {pixel_values.shape}")
        print(f"     采样帧数: {len(num_patches_list)}, 每帧块数分布示例: {num_patches_list[:5]}...")
    except Exception as e:
        print(f"[ERROR] 视频处理失败: {e}")
        return

    # 5. 构建视频 Prompt 前缀
    # 格式: Frame1: <image>\nFrame2: <image>\n... (告知模型每一帧对应的位置)
    video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
    
    chat_history = None

    # ================= 推理阶段 =================
    print("\n" + "="*70)
    print("开始多轮对话推理...")
    print("="*70)
    
    with torch.no_grad():
        # --- 第一轮：详细视频描述 (英文) ---
        print("\n[Q1] 问题: Describe this video in detail.")
        question1 = video_prefix + "Describe this video in detail."
        
        try:
            output1, chat_history = model.chat(
                tokenizer, 
                pixel_values, 
                question1, 
                GENERATION_CONFIG, 
                num_patches_list=num_patches_list, 
                history=None, 
                return_history=True
            )
            print(f"[A1] 回答:\n{output1}\n")
        except Exception as e:
            print(f"[ERROR] 第一轮推理失败: {e}")
            return

        # --- 第二轮：计数任务 (英文，基于上下文) ---
        print("[Q2] 问题: How many people appear in the video?")
        question2 = "How many people appear in the video?"
        
        try:
            output2, chat_history = model.chat(
                tokenizer, 
                pixel_values, 
                question2, 
                GENERATION_CONFIG, 
                num_patches_list=num_patches_list, 
                history=chat_history, # 继承上一轮历史
                return_history=True
            )
            print(f"[A2] 回答:\n{output2}\n")
        except Exception as e:
            print(f"[ERROR] 第二轮推理失败: {e}")

        # --- 第三轮：损伤检测 (中文，新话题) ---
        print("[Q3] 问题: 车的哪个部位损伤了？")
        # 注意：这里重新构建了 prefix + question，并重置 history=None
        # 目的是让模型专注于视频内容本身，不受前两轮英文对话的干扰
        question3 = video_prefix + "车的哪个部位损伤了？"
        
        try:
            output3, chat_history_cn = model.chat(
                tokenizer, 
                pixel_values, 
                question3, 
                GENERATION_CONFIG, 
                num_patches_list=num_patches_list, 
                history=None, # 重置历史，开启新的中文对话线程
                return_history=True
            )
            print(f"[A3] 回答:\n{output3}\n")
        except Exception as e:
            print(f"[ERROR] 第三轮推理失败: {e}")

        # --- 第四轮：碰撞位置追问 (中文，基于上一轮中文上下文) ---
        print("[Q4] 问题: 车撞到哪里了？ (基于上一轮上下文)")
        question4 = "车撞到哪里了？"
        
        try:
            output4, _ = model.chat(
                tokenizer, 
                pixel_values, 
                question4, 
                GENERATION_CONFIG, 
                num_patches_list=num_patches_list, 
                history=chat_history_cn, # 使用上一轮中文对话的历史
                return_history=True
            )
            print(f"[A4] 回答:\n{output4}\n")
        except Exception as e:
            print(f"[ERROR] 第四轮推理失败: {e}")

    print("="*70)
    print("所有任务执行完毕。")
    print("="*70)
    
    # 清理显存
    torch.cuda.empty_cache()
    print("已清理 GPU 显存。")

if __name__ == "__main__":
    main()
