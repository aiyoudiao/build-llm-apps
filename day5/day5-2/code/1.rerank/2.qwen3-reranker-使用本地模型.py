"""
## Qwen3 Reranker

基于 Qwen3-Reranker-0.6B 模型计算 Query 和 Document 的相关性分数。
该模型基于 CausalLM 架构，通过生成 "yes"/"no" 的概率来判断相关性。
"""

import os
import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================
# 1. 模型配置与下载
# ==============================

# ModelScope 上的模型名称
model_name = "Qwen/Qwen3-Reranker-0.6B"
# 本地缓存目录
cache_dir = "../../../../../models"
# 本地模型完整路径
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

# ==============================
# 2. 核心辅助函数
# ==============================

def format_instruction(instruction, query, doc):
    """格式化输入指令"""
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return output

def process_inputs(pairs, tokenizer, max_length, prefix_tokens, suffix_tokens, device):
    """处理输入数据：Tokenize -> 添加前缀后缀 -> Padding"""
    # 1. Tokenize (不填充，先截断)
    inputs = tokenizer(
        pairs, 
        padding=False, 
        truncation='longest_first', 
        return_attention_mask=False, 
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    
    # 2. 手动添加 prefix 和 suffix
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    
    # 3. Padding (使用 tokenizer 的 pad 方法)
    inputs = tokenizer.pad(inputs,  padding=True, return_tensors="pt", max_length=max_length)
    
    # 4. 移动到指定设备
    for key in inputs:
        inputs[key] = inputs[key].to(device)
        
    return inputs

@torch.no_grad()
def compute_logits(model, inputs, token_true_id, token_false_id):
    """计算相关性分数"""
    # 获取最后一个 token 的 logits
    batch_scores = model(**inputs).logits[:, -1, :]
    
    # 提取 "yes" 和 "no" 对应的 logits
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    
    # 堆叠 logits: [false_score, true_score]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    
    # 计算 log_softmax
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    
    # 取 "yes" 的概率 (索引为1) 并 exp 还原
    scores = batch_scores[:, 1].exp().tolist()
    return scores

# ==============================
# 3. 加载 Tokenizer 和 Model
# ==============================

# 选择设备
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"使用设备: {device}")

print("正在加载 Tokenizer...")
# 注意：padding_side='left' 是必须的，因为是 Decoder-only 模型
tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side='left', trust_remote_code=True)

print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to(device).eval()

# 获取 "yes" 和 "no" 的 token ID
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")

# 最大序列长度
max_length = 8192

# 定义 Prompt 模板 (包含 System Prompt 和 思维链占位符)
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# 预先编码前缀和后缀
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

# ==============================
# 4. 运行示例
# ==============================

task = 'Given a web search query, retrieve relevant passages that answer the query'

# 构造测试数据 (Query, Document) 对
queries = [
    "What is the capital of China?", 
    "Explain gravity",
    "What is panda?" 
]

documents = [
    "The capital of China is Beijing.", 
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    "The Eiffel Tower is in Paris." # 这是一个不相关的例子
]

print("\n--- 开始评估 ---")

# 格式化输入对
pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, documents)]

# 处理输入
inputs = process_inputs(pairs, tokenizer, max_length, prefix_tokens, suffix_tokens, device)

# 计算分数
scores = compute_logits(model, inputs, token_true_id, token_false_id)

# 打印结果
print("Scores:", scores)
print("\n详细结果:")
for q, d, s in zip(queries, documents, scores):
    print(f"Query: {q}")
    print(f"Doc: {d}")
    print(f"Score: {s:.4f}")
    print("-" * 30)
