# -*- coding: utf-8 -*-
"""
===============================================================================
SVD图像压缩与还原（模拟大模型微调）

功能描述: 
    利用奇异值分解 (SVD, Singular Value Decomposition) 技术对图像进行
    有损压缩与还原演示。通过保留前 k 个最大的奇异值，丢弃较小的奇异值，
    在大幅减少数据存储量的同时，尽可能保留图像的主要视觉特征。

核心数学原理:
    1. 奇异值分解 (SVD):
       对于任意实数矩阵 A (m x n)，可以分解为: A = U * Σ * V^T
       - U (左奇异向量): m x m 正交矩阵，代表行空间的特征。
       - Σ (奇异值矩阵): m x n 对角矩阵，对角线元素为奇异值 (σ₁ ≥ σ₂ ≥ ... ≥ 0)。
       - V^T (右奇异向量转置): n x n 正交矩阵，代表列空间的特征。
    
    2. 低秩近似 (Low-Rank Approximation):
       图像的主要能量集中在前几个较大的奇异值上。
       若只保留前 k 个奇异值，构造近似矩阵 A_k = U_k * Σ_k * V_k^T，
       则 A_k 是在秩为 k 的矩阵中，与原矩阵 A 误差最小 (Frobenius 范数意义下) 的最优近似。
    
    3. 压缩比:
       原始存储量: m * n
       压缩后存储量: k * (m + n + 1)
       当 k << min(m, n) 时，实现显著压缩。

执行流程:
    1. [加载图像] 读取本地图片文件，转换为 NumPy 数组。
    2. [预处理] 若为彩色图片 (RGB)，拆分为 R, G, B 三个通道矩阵分别处理。
    3. [SVD 分解] 对每个通道矩阵进行奇异值分解，得到 U, S, V^T。
    4. [压缩还原] 
       - 设定不同的 k 值 (如 5, 50, 500)。
       - 截断奇异值数组，仅保留前 k 个。
       - 重构矩阵: A_reconstructed = U * S_truncated * V^T。
    5. [结果展示] 
       - 合并通道 (如果是彩色图)。
       - 使用 Matplotlib 对比显示原图与不同 k 值下的还原图。
       - 计算并打印均方误差 (MSE) 以量化还原质量。

依赖说明:
    - numpy: 矩阵运算
    - scipy.linalg: 高效 SVD 算法
    - PIL: 图像加载
    - matplotlib: 图像可视化

注意:
    - 请确保当前目录下存在名为 '256.bmp' 的图片，或修改代码中的路径。
    - k 值越大，图像越清晰，但压缩率越低；k 值越小，图像越模糊，压缩率越高。
===============================================================================
"""

import numpy as np
from scipy.linalg import svd
from PIL import Image
import matplotlib.pyplot as plt

def reconstruct_image_from_svd(U, s, Vt, k, original_shape):
    """
    利用前 k 个奇异值重构图像矩阵
    
    参数:
        U: 左奇异向量矩阵 (m x m 或 m x k)
        s: 奇异值数组 (长度为 min(m, n))
        Vt: 右奇异向量矩阵的转置 (n x n 或 k x n)
        k: 保留的奇异值个数
        original_shape: 原始矩阵形状，用于调试参考
    
    返回:
        reconstructed_matrix: 重构后的矩阵
    """
    # 1. 截断奇异值：创建一个全零数组，只填入前 k 个最大的奇异值
    # 这一步相当于将对角矩阵 Σ 中 k 之后的元素置为 0
    s_truncated = np.zeros_like(s)
    s_truncated[:k] = s[:k]
    
    # 2. 构建对角矩阵 Σ_k
    # np.diag(s_truncated) 会生成一个方阵，对角线为 s_truncated
    Sigma_k = np.diag(s_truncated)
    
    # 3. 矩阵乘法重构: A_k = U * Σ_k * V^T
    # 注意：由于 s_truncated 后面是 0，实际上只需要计算 U 的前 k 列和 V^T 的前 k 行
    # 但为了逻辑清晰，这里直接进行完整矩阵乘法 (scipy 的 svd 返回的是紧凑模式或全模式)
    reconstructed_matrix = np.dot(U, np.dot(Sigma_k, Vt))
    
    return reconstructed_matrix

def process_channel_and_show(channel_data, k_values, title_prefix):
    """
    对单个通道 (或灰度图) 进行 SVD 处理并展示不同 k 值的效果
    """
    # 进行奇异值分解
    # full_matrices=False 返回紧凑模式，节省内存且计算更快
    U, s, Vt = svd(channel_data, full_matrices=False)
    
    print(f"\n{title_prefix} 通道 SVD 分解完成:")
    print(f"  - 原始形状: {channel_data.shape}")
    print(f"  - 奇异值个数: {len(s)}")
    print(f"  - 前 5 个奇异值: {s[:5]}")

    # 创建画布用于对比展示
    n_plots = len(k_values) + 1 # 原图 + k 个还原图
    fig_height = 6
    fig, axes = plt.subplots(1, n_plots, figsize=(15, fig_height))
    
    # 显示原图
    if n_plots == 1:
        ax = axes
    else:
        ax = axes[0]
    ax.imshow(channel_data, cmap='gray', interpolation='nearest')
    ax.set_title("Original")
    ax.axis('off')

    # 循环处理不同的 k 值
    for i, k in enumerate(k_values):
        # 限制 k 不超过最大奇异值数量
        actual_k = min(k, len(s))
        
        # 重构图像
        reconstructed = reconstruct_image_from_svd(U, s, Vt, actual_k, channel_data.shape)
        
        # 确保数据范围在 0-255 之间，并转换为 uint8 格式以便显示
        # 截断操作可能会引入极小的浮点误差导致超出范围
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        
        # 计算误差 (均方误差 MSE)
        mse = np.mean((channel_data - reconstructed) ** 2)
        
        # 绘图
        ax_idx = i + 1
        if n_plots == 1:
            ax = axes # 这种情况理论上不会发生，因为 k_values 至少有 1 个
        else:
            ax = axes[ax_idx]
            
        ax.imshow(reconstructed, cmap='gray', interpolation='nearest')
        ax.set_title(f"k={actual_k}\nMSE: {mse:.2f}")
        ax.axis('off')
        
        print(f"  - k={actual_k}: 重构完成，MSE = {mse:.2f}")

    plt.tight_layout()
    plt.show()

def main():
    # 配置
    image_path = './256.bmp'
    k_values = [5, 50, 500]  # 设定要测试的奇异值保留个数
    
    print("="*60)
    print("开始 SVD 图像压缩与还原演示")
    print("="*60)

    try:
        # 1. 加载图像
        image = Image.open(image_path)
        print(f"成功加载图像: {image_path}, 模式: {image.mode}, 尺寸: {image.size}")
    except FileNotFoundError:
        print(f"错误: 未找到文件 '{image_path}'")
        print("提示: 请确保当前目录下存在该图片，或修改代码中的路径。")
        # 创建一个简单的测试图像 (灰度渐变) 以防没有图片
        print("正在生成测试用渐变图像...")
        x = np.linspace(0, 255, 200)
        y = np.linspace(0, 255, 200)
        X, Y = np.meshgrid(x, y)
        A = (X + Y) / 2
        image = Image.fromarray(A.astype(np.uint8))
        image.save(image_path)
        print(f"测试图像已保存为 {image_path}")

    # 转换为 NumPy 数组
    img_array = np.array(image)

    # 2. 判断是灰度图还是彩色图
    if len(img_array.shape) == 2:
        # 灰度图: 直接处理
        print("检测到灰度图像，直接进行 SVD 处理...")
        process_channel_and_show(img_array, k_values, "Gray")
        
    elif len(img_array.shape) == 3:
        # 彩色图 (RGB): 拆分通道分别处理，然后合并显示
        print("检测到彩色图像 (RGB)，将分通道进行 SVD 处理...")
        h, w, c = img_array.shape
        
        # 准备存放各通道还原结果的列表
        reconstructed_channels = []
        
        # 为了演示，我们只针对最后一个 k 值 (如 50) 展示合并后的彩色效果
        # 因为分别展示 R,G,B 的对比图太多，这里简化流程：
        # 1. 先展示 R 通道的详细对比 (代表整体效果)
        # 2. 再展示最终合成的彩色对比图
        
        # --- 步骤 A: 展示红色通道的详细分解过程 (作为示例) ---
        r_channel = img_array[:, :, 0]
        process_channel_and_show(r_channel, k_values, "Red Channel (Example)")
        
        # --- 步骤 B: 对所有通道进行处理并合成彩色图进行对比 ---
        fig, axes = plt.subplots(1, len(k_values) + 1, figsize=(20, 5))
        axes[0].imshow(img_array)
        axes[0].set_title("Original Color")
        axes[0].axis('off')
        
        for i, k in enumerate(k_values):
            temp_channels = []
            for ch_idx in range(c):
                channel_data = img_array[:, :, ch_idx]
                U, s, Vt = svd(channel_data, full_matrices=False)
                recon_ch = reconstruct_image_from_svd(U, s, Vt, min(k, len(s)), channel_data.shape)
                temp_channels.append(np.clip(recon_ch, 0, 255).astype(np.uint8))
            
            # 合并通道
            merged_img = np.stack(temp_channels, axis=2)
            
            ax_idx = i + 1
            axes[ax_idx].imshow(merged_img)
            axes[ax_idx].set_title(f"Reconstructed (k={min(k, len(s))})")
            axes[ax_idx].axis('off')
            
        plt.tight_layout()
        plt.show()
        
    else:
        print("不支持的图像格式。")

if __name__ == "__main__":
    main()
