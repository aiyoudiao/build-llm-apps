"""
## BGE Reranker

计算「问题」和「候选答案」之间的相关性分数
分数越大 → 越相关
"""

import os
import torch
from modelscope import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ==============================
# 1. 模型配置
# ==============================

# ModelScope 上的模型名称
model_name = "BAAI/bge-reranker-base"

# 本地缓存目录
cache_dir = "../../../../../models"

# 本地模型完整路径
model_dir = os.path.join(cache_dir, model_name)


# ==============================
# 2. 模型下载（如果不存在）
# ==============================

if not os.path.exists(model_dir):
    print(f"开始下载模型 {model_name}")
    snapshot_download(model_name, cache_dir=cache_dir)
    print("模型下载完成")
else:
    print("本地已有模型，跳过下载")


# ==============================
# 3. 加载 tokenizer & model
# ==============================

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 加载序列分类模型（用于输出相关性分数）
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# 切换到评估模式（关闭 dropout）
model.eval()


# ==============================
# 4. 单条测试
# ==============================

pairs = [
    ["what is panda?", "The giant panda is a bear species endemic to China."]
]

# tokenizer 会把 (query, doc) 拼接为一个输入序列
inputs = tokenizer(
    pairs,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# 关闭梯度计算（推理模式，更高效）
with torch.no_grad():
    logits = model(**inputs).logits

# 展平成一维张量
scores = logits.view(-1).float()

print("单条相关性分数：", scores)


# ==============================
# 5. 多条对比测试
# ==============================

pairs = [
    ["what is panda?", "The giant panda is a bear species endemic to China."],  # 高相关
    ["what is panda?", "Pandas are cute."],                                     # 中相关
    ["what is panda?", "The Eiffel Tower is in Paris."]                         # 不相关
]

inputs = tokenizer(
    pairs,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

with torch.no_grad():
    logits = model(**inputs).logits

scores = logits.view(-1).float()

print("多条相关性分数：", scores)
