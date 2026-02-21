import os
import torch # PyTorch深度学习库
import torch.nn.functional as F # PyTorch函数式接口，包含各种神经网络函数
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModel # 从modelscope导入自动分词器和模型加载器
import transformers.dynamic_module_utils

# 这一步是为了解决在 macOS 上安装不了 flash_attn 的问题
original_check_imports = transformers.dynamic_module_utils.check_imports

def patched_check_imports(filename):
    try:
        return original_check_imports(filename)
    except ImportError as e:
        if "flash_attn" in str(e):
            return []
        raise e

transformers.dynamic_module_utils.check_imports = patched_check_imports

# 配置模型路径
model_name = "iic/gte_Qwen2-1.5B-instruct"
cache_dir = "../../../../../models"
model_dir = os.path.join(cache_dir, model_name)

# 下载模型（如果不存在）
if not os.path.exists(model_dir):
    print(f"开始下载模型 {model_name} 到 {cache_dir}")
    snapshot_download(model_name, cache_dir=cache_dir)
else:
    print(f"模型 {model_name} 已存在于 {model_dir}")

# 选择设备
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"使用设备 {device}")

# --- 核心函数定义 ---

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    手写 pooling 逻辑：提取最后一个有效 token 的隐藏状态
    """
    # 检查是否为左侧填充
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        # 计算每个序列的长度（注意索引从0开始，所以要减1）
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        # 提取最后一个有效 token
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    """
    为查询添加任务描述指令
    """
    return f'Instruct: {task_description}\nQuery: {query}'

# --- 加载模型 ---
print("正在加载 tokenizer 和 model...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device)
print("模型加载完成")

# --- 准备数据 ---
# 定义任务描述
task = 'Given a web search query, retrieve relevant passages that answer the query'

# 原始查询和文档
queries = ["What is BGE M3?", "Defination of BM25"]
documents = [
    "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
    "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"
]

# 格式化输入：查询需要添加指令，文档不需要
formatted_queries = [get_detailed_instruct(task, q) for q in queries]
input_texts = formatted_queries + documents

# Tokenize
max_length = 8192
batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)

# 推理 (获取 hidden states)
print("正在执行推理...")
with torch.no_grad():
    outputs = model(**batch_dict)

# --- 1. 手写 pooling ---
# 原理：gte-Qwen2 使用最后一个 token 的输出作为句子的向量表示
print("\n--- 1. 手写 pooling (从 token 变成句子向量) ---")
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
print(f"原始 Embedding 形状: {embeddings.shape}")
print(f"前5个数值 (未归一化): {embeddings[0][:5].cpu().tolist()}")

# --- 2. 手写归一化 ---
# 原理：将向量长度缩放为 1，方便计算余弦相似度
print("\n--- 2. 手写归一化 (把向量长度变成 1) ---")
embeddings = F.normalize(embeddings, p=2, dim=1)
print(f"归一化后 Embedding 形状: {embeddings.shape}")
print(f"前5个数值 (已归一化): {embeddings[0][:5].cpu().tolist()}")

# 分离查询向量和文档向量
query_embeddings = embeddings[:len(queries)]
doc_embeddings = embeddings[len(queries):]

# --- 3. 手写相似度计算 ---
# 原理：矩阵乘法 (Query @ Doc.T) 等价于计算余弦相似度
print("\n--- 3. 手写相似度计算 (算余弦相似度) ---")
similarity = query_embeddings @ doc_embeddings.T
print("相似度矩阵:")
print(similarity)

# 打印具体结果
print("\n详细结果:")
for i, q in enumerate(queries):
    for j, d in enumerate(documents):
        score = similarity[i][j].item()
        print(f"Query: '{q}' \n  vs Doc: '{d[:30]}...' \n  -> Score: {score:.4f}")
# 对比 2.gte-qwen2-只用本地模型.py 中直接使用 SentenceTransformer encode 方法的结果
"""
[[0.6934099 0.3438392]
 [0.4896578 0.6371974]]
"""
