import os
import torch
from modelscope import snapshot_download
import transformers.dynamic_module_utils

# 从 transformers 内部模块中导入 dynamic_module_utils
# 这个模块负责在 trust_remote_code=True 时，
# 对远程模型文件进行“依赖检查”
import transformers.dynamic_module_utils


# 保存原始的 check_imports 函数
# 后面我们会调用它，以保持原有逻辑
original_check_imports = transformers.dynamic_module_utils.check_imports


# 定义一个“打补丁”的新函数
# 用来替换 transformers 内部的 check_imports
def patched_check_imports(filename):
    """
    自定义依赖检查函数

    参数:
        filename: 当前正在被 transformers 检查的 modeling 文件路径

    作用:
        1. 调用原始的依赖检查逻辑
        2. 如果发现缺少 flash_attn 依赖，则忽略该错误
        3. 其它依赖缺失仍然正常抛出异常
    """
    try:
        # 先调用原始检查逻辑
        return original_check_imports(filename)

    except ImportError as e:
        # 如果报错信息中包含 flash_attn
        # 说明是因为没有安装 flash_attn 导致的 ImportError
        # flash_attn 是为了 GPU 加速，该依赖在 macOS/CPU 环境无法安装
        # 必须绕过 transformers 在 trust_remote_code=True 时对 flash_attn 的强制依赖检查
        if "flash_attn" in str(e):
            print("忽略 flash_attn 依赖检查")

            # 返回空列表，表示“依赖检查通过”
            # 让 transformers 继续加载模型
            return []

        # 如果是其他依赖缺失，不处理，继续抛出异常
        raise e


# 用我们自定义的函数
# 替换 transformers 内部的 check_imports 方法
# 这就是 monkey patch（hack：在程序运行时，动态修改已有代码的行为）
transformers.dynamic_module_utils.check_imports = patched_check_imports

# gte_Qwen2-1.5B-instruct 模型的 modelscope 地址
model_name = "iic/gte_Qwen2-1.5B-instruct"
# 模型下载保存的目录
cache_dir = "../../../../../models"
# 模型的本地路径
model_dir = os.path.join(cache_dir, model_name)

# 判断目录下是否包含该模型，如果不包含则下载
if not os.path.exists(model_dir):
    print(f"开始下载模型 {model_name} 到 {cache_dir}")
    # 使用 modelscope 下载模型
    snapshot_download(model_name, cache_dir=cache_dir)
    print(f"{model_name} 下载完成")
else:
    print(f"模型 {model_name} 已存在于 {model_dir}")
    print("请勿重复下载")

# 选择设备：优先使用 CUDA，其次是 MPS (Mac)，最后是 CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"使用设备 {device}")

# 加载 gte_Qwen2 模型
from sentence_transformers import SentenceTransformer

print("正在加载模型...")
# trust_remote_code=True 是必须的，因为它是基于 Qwen2 的自定义模型
# 如果显存不足，可以尝试添加 model_kwargs={"torch_dtype": torch.float16}
model = SentenceTransformer(model_dir, trust_remote_code=True, device=device)
# 设置最大序列长度
model.max_seq_length = 8192
print("模型加载完成")

# 定义待编码的句子列表
sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

# 生成句子的 dense embedding (稠密向量)
print("正在生成向量...")
# prompt_name="query" 用于查询/问题，文档不需要 prompt_name
# normalize_embeddings=True 确保向量归一化，这样点积就等于余弦相似度
embeddings_1 = model.encode(sentences_1, prompt_name="query", normalize_embeddings=True)
embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)

print(f"句子1的向量形状: {embeddings_1.shape}")
print(f"句子2的向量形状: {embeddings_2.shape}")

# 计算相似度
# 使用 @ 运算符进行矩阵乘法计算点积
similarity = embeddings_1 @ embeddings_2.T
print("相似度矩阵:")
print(similarity)

# 打印具体的相似度分数
print(f"'{sentences_1[0]}' 和 '{sentences_2[0]}' 的相似度: {similarity[0][0]:.4f}")
print(f"'{sentences_1[0]}' 和 '{sentences_2[1]}' 的相似度: {similarity[0][1]:.4f}")
print(f"'{sentences_1[1]}' 和 '{sentences_2[0]}' 的相似度: {similarity[1][0]:.4f}")
print(f"'{sentences_1[1]}' 和 '{sentences_2[1]}' 的相似度: {similarity[1][1]:.4f}")
