"""
多模态 RAG (检索增强生成) 助手。
它能够处理 Word 文档和图片，提取其中的信息，并结合文本和图片的内容来回答用户的问题。

主要功能：
1. 文档解析：从 .docx 文件中提取文本和表格。
2. 图片理解：使用 OCR (光学字符识别) 提取图片文字，使用 CLIP 模型提取图片特征。
3. 向量存储：使用 FAISS 向量数据库存储文本和图片的特征向量，支持本地持久化存储。
4. 增量更新：自动检测源文件变化（基于 MD5 哈希），仅在文件变更时重建索引，避免重复计算。
5. 检索增强：根据用户问题，在知识库中检索相关的文本和图片。
6. 智能问答：利用大语言模型 (LLM) 结合检索到的信息生成最终回答。

运行本脚本需要配置环境变量 DASHSCOPE_API_KEY (阿里云百炼 API Key)。
"""

import os
# 设置环境变量以避免可能的冲突导致的 Segmentation Fault
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import re
import numpy as np
import faiss
import pickle
import json
import hashlib
from openai import OpenAI
from docx import Document as DocxDocument
import fitz  # PyMuPDF，虽然这里主要用 docx 库处理 Word，但在处理 PDF 时会用到 fitz
from PIL import Image
import pytesseract
from transformers import CLIPProcessor, CLIPModel
import torch
from dotenv import load_dotenv  # 用于加载 .env 文件中的环境变量

# 加载 .env 文件中的环境变量，方便本地开发
load_dotenv()

# Step0. 全局配置与模型加载

# 检查环境变量 DASHSCOPE_API_KEY 是否存在
# 这是调用阿里云百炼大模型服务的必要凭证
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    # 如果没有找到 API Key，抛出错误并提示用户设置
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。可以在当前目录下创建一个 .env 文件并写入 DASHSCOPE_API_KEY=您的Key")

# 初始化 OpenAI 客户端，配置为兼容阿里云百炼的接口
# base_url 指向阿里云的服务地址
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 加载 CLIP 模型，用于图像和文本的特征提取
# CLIP (Contrastive Language-Image Pre-Training) 可以将图片和文本映射到同一个向量空间
print("正在加载 CLIP 模型...")
try:
    # 加载预训练的 CLIP 模型
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to('cpu')  # 强制使用 CPU，避免 macOS 上的潜在兼容性问题
    # 加载对应的处理器，用于预处理图片和文本
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP 模型加载成功。")
except Exception as e:
    # 如果加载失败（通常是网络问题导致无法下载模型），打印错误并退出
    print(f"加载 CLIP 模型失败，请检查网络连接或 Hugging Face Token。错误: {e}")
    exit()

# 定义全局变量
# 文档存放的目录
DOCS_DIR = "disney_knowledge_base"
# 图片存放的子目录
IMG_DIR = os.path.join(DOCS_DIR, "images")
# 使用的文本 Embedding 模型名称 (阿里云百炼提供的模型)
TEXT_EMBEDDING_MODEL = "text-embedding-v4"
# 文本 Embedding 的维度，需要与模型输出一致
TEXT_EMBEDDING_DIM = 1024
# 图像 Embedding 的维度，取决于 CLIP 模型的配置 (vit-base-patch32 输出 512 维)
IMAGE_EMBEDDING_DIM = 512
# 向量库保存目录
VECTOR_STORE_DIR = "vector_store"

# Step1. 文档解析与内容提取

def get_file_hash(file_path):
    """计算文件的 MD5 哈希值，用于检测文件变化。"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_current_files_state(docs_dir, img_dir):
    """获取当前所有文件的状态（文件名: 哈希值）。"""
    files_state = {}
    
    # 扫描文档目录
    for filename in os.listdir(docs_dir):
        if filename.startswith('.') or os.path.isdir(os.path.join(docs_dir, filename)):
            continue
        if filename.endswith(".docx"):
            file_path = os.path.join(docs_dir, filename)
            files_state[f"doc:{filename}"] = get_file_hash(file_path)
            
    # 扫描图片目录
    if os.path.exists(img_dir):
        for img_filename in os.listdir(img_dir):
            if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img_path = os.path.join(img_dir, img_filename)
                files_state[f"img:{img_filename}"] = get_file_hash(img_path)
                
    return files_state

def parse_docx(file_path):
    """
    解析 DOCX 文件，提取文本和表格内容。
    
    参数:
        file_path: DOCX 文件的路径
        
    返回:
        content_chunks: 包含文本和表格内容的字典列表
    """
    # 使用 python-docx 库打开 Word 文档
    doc = DocxDocument(file_path)
    content_chunks = []
    
    # 遍历文档体中的每一个元素（段落或表格）
    for element in doc.element.body:
        if element.tag.endswith('p'):
            # 处理段落元素
            paragraph_text = ""
            # 提取段落中的所有文本运行 (run)
            for run in element.findall('.//w:t', {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
                paragraph_text += run.text if run.text else ""
            
            # 如果段落有内容，将其添加到结果列表中
            if paragraph_text.strip():
                content_chunks.append({"type": "text", "content": paragraph_text.strip()})
                
        elif element.tag.endswith('tbl'):
            # 处理表格元素
            md_table = []
            # 找到对应的表格对象
            table = [t for t in doc.tables if t._element is element][0]
            
            if table.rows:
                # 处理表头：获取第一行的所有单元格文本
                header = [cell.text.strip() for cell in table.rows[0].cells]
                # 转换为 Markdown 表格格式
                md_table.append("| " + " | ".join(header) + " |")
                md_table.append("|" + "---|"*len(header))
                
                # 处理数据行：遍历剩余的行
                for row in table.rows[1:]:
                    row_data = [cell.text.strip() for cell in row.cells]
                    md_table.append("| " + " | ".join(row_data) + " |")
                
                # 将表格行合并为一个字符串
                table_content = "\n".join(md_table)
                if table_content.strip():
                    content_chunks.append({"type": "table", "content": table_content})
    
    return content_chunks

def image_to_text(image_path):
    """
    对图片进行处理，提取文字信息。
    这里使用 Tesseract OCR 引擎识别图片中的文字。
    
    参数:
        image_path: 图片文件的路径
        
    返回:
        字典，包含 OCR 识别出的文本
    """
    try:
        # 打开图片文件
        image = Image.open(image_path)
        # 使用 pytesseract 调用 Tesseract OCR 引擎进行文字识别
        # lang='chi_sim+eng' 表示同时识别简体中文和英文
        ocr_text = pytesseract.image_to_string(image, lang='chi_sim+eng').strip()
        return {"ocr": ocr_text}
    except Exception as e:
        # 如果 OCR 失败（例如未安装 Tesseract），打印错误但程序继续运行
        print(f"处理图片失败 {image_path}: {e}")
        return {"ocr": ""}

# Step2. Embedding 与索引构建

def get_text_embedding(text):
    """
    调用阿里云百炼 API 获取文本的 Embedding 向量。
    
    参数:
        text: 输入文本
        
    返回:
        embedding: 文本对应的向量 (列表形式)
    """
    # 调用 embeddings.create 接口
    response = client.embeddings.create(
        model=TEXT_EMBEDDING_MODEL,
        input=text,
        dimensions=TEXT_EMBEDDING_DIM
    )
    # 返回第一个结果的 embedding 数据
    return response.data[0].embedding

def get_image_embedding(image_path):
    """
    使用本地 CLIP 模型获取图片的 Embedding 向量。
    
    参数:
        image_path: 图片文件路径
        
    返回:
        image_features: 图片的特征向量 (numpy 数组)
    """
    # 打开图片
    image = Image.open(image_path)
    # 使用 processor 预处理图片，转换为模型输入格式 (tensor)
    inputs = clip_processor(images=image, return_tensors="pt")
    # 使用模型进行推理，获取图像特征
    with torch.no_grad():  # 禁用梯度计算，节省内存并加速
        image_features = clip_model.get_image_features(**inputs)
    # 返回第一个结果，并转换为 numpy 数组
    return image_features[0].numpy()

def get_clip_text_embedding(text):
    """
    使用本地 CLIP 模型获取文本的 Embedding 向量。
    这用于将用户的文本查询映射到图像向量空间，以便进行以文搜图。
    
    参数:
        text: 输入文本
        
    返回:
        text_features: 文本的特征向量 (numpy 数组)
    """
    # 使用 processor 预处理文本
    inputs = clip_processor(text=text, return_tensors="pt")
    # 使用模型进行推理，获取文本特征
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    return text_features[0].numpy()

def build_knowledge_base(docs_dir, img_dir, save_dir=VECTOR_STORE_DIR):
    """
    构建或加载知识库：
    1. 检查本地是否有保存的索引且文件未发生变化。
    2. 如果满足条件，直接加载。
    3. 否则，重新解析文档、生成 Embedding 并构建 FAISS 索引，然后保存。
    
    参数:
        docs_dir: 文档目录路径
        img_dir: 图片目录路径
        save_dir: 向量库保存目录
        
    返回:
        metadata_store: 元数据列表
        text_index_map: 文本向量索引 (FAISS)
        image_index_map: 图片向量索引 (FAISS)
    """
    
    # 定义保存文件的路径
    metadata_path = os.path.join(save_dir, "metadata.pkl")
    text_index_path = os.path.join(save_dir, "text.index")
    image_index_path = os.path.join(save_dir, "image.index")
    file_state_path = os.path.join(save_dir, "file_state.json")
    
    # 获取当前文件状态
    current_files_state = get_current_files_state(docs_dir, img_dir)
    
    # 检查是否可以直接加载
    need_rebuild = True
    if os.path.exists(save_dir) and \
       os.path.exists(metadata_path) and \
       os.path.exists(text_index_path) and \
       os.path.exists(image_index_path) and \
       os.path.exists(file_state_path):
        
        try:
            with open(file_state_path, 'r') as f:
                saved_files_state = json.load(f)
            
            # 比较文件状态
            if current_files_state == saved_files_state:
                print("\n--- 检测到本地向量库且源文件未变化，正在加载... ---")
                
                # 加载 metadata
                with open(metadata_path, 'rb') as f:
                    metadata_store = pickle.load(f)
                
                # 加载 FAISS 索引
                text_index_map = faiss.read_index(text_index_path)
                image_index_map = faiss.read_index(image_index_path)
                
                print(f"加载完成。共包含 {len([m for m in metadata_store if m['type']=='text'])} 个文本片段和 {len([m for m in metadata_store if m['type']=='image'])} 张图片。")
                need_rebuild = False
            else:
                print("\n--- 源文件发生变化，准备重建向量库... ---")
        except Exception as e:
            print(f"\n--- 加载本地向量库失败 ({e})，准备重建... ---")
    else:
        print("\n--- 本地向量库不存在，准备构建... ---")

    if not need_rebuild:
        return metadata_store, text_index_map, image_index_map

    # --- 以下是重建逻辑 ---
    print("\n--- 步骤 1 & 2: 正在解析、Embedding并索引知识库 ---")
    
    metadata_store = []
    text_vectors = []
    image_vectors = []
    
    doc_id_counter = 0

    # 1. 处理 Word 文档
    # 遍历文档目录下的所有文件
    for filename in os.listdir(docs_dir):
        # 跳过隐藏文件和子目录
        if filename.startswith('.') or os.path.isdir(os.path.join(docs_dir, filename)):
            continue
            
        file_path = os.path.join(docs_dir, filename)
        # 只处理 .docx 文件
        if filename.endswith(".docx"):
            print(f"  - 正在处理: {filename}")
            # 调用解析函数提取内容
            chunks = parse_docx(file_path)
            
            # 遍历提取出的每个片段
            for chunk in chunks:
                metadata = {
                    "id": doc_id_counter,
                    "source": filename,
                    "page": 1  # docx 通常不分页，这里简单标记为 1
                }
                
                # 处理文本和表格片段
                if chunk["type"] == "text" or chunk["type"] == "table":
                    text = chunk["content"]
                    if not text.strip(): 
                        continue
                    
                    metadata["type"] = "text"
                    metadata["content"] = text
                    
                    # 获取文本向量
                    vector = get_text_embedding(text)
                    text_vectors.append(vector)
                    metadata_store.append(metadata)
                    doc_id_counter += 1

    # 2. 处理 images 目录中的独立图片文件
    print("  - 正在处理独立图片文件...")
    # 确保图片目录存在
    if os.path.exists(img_dir):
        for img_filename in os.listdir(img_dir):
            # 只处理常见的图片格式
            if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img_path = os.path.join(img_dir, img_filename)
                print(f"    - 处理图片: {img_filename}")
                
                # 对图片进行 OCR 识别
                print(f"      - [OCR] 正在识别图片文字: {img_filename} ...")
                img_text_info = image_to_text(img_path)
                
                metadata = {
                    "id": doc_id_counter,
                    "source": f"独立图片: {img_filename}",
                    "type": "image",
                    "path": img_path,
                    "ocr": img_text_info["ocr"],
                    "page": 1
                }
                
                # 获取图片向量
                print(f"      - [CLIP] 正在提取图片特征: {img_filename} ...")
                vector = get_image_embedding(img_path)
                image_vectors.append(vector)
                metadata_store.append(metadata)
                doc_id_counter += 1
    else:
        print(f"    - 警告: 图片目录 {img_dir} 不存在，跳过图片处理。")

    # 3. 创建 FAISS 索引
    
    # 创建文本向量索引
    # IndexFlatL2 使用欧氏距离进行精确搜索
    text_index = faiss.IndexFlatL2(TEXT_EMBEDDING_DIM)
    # IndexIDMap 用于将向量 ID 映射到我们自定义的 doc_id
    text_index_map = faiss.IndexIDMap(text_index)
    
    # 提取所有文本类型的 ID
    text_ids = [m["id"] for m in metadata_store if m["type"] == "text"]
    if text_vectors:  # 只有当有文本向量时才添加到索引
        # FAISS 需要 float32 类型的 numpy 数组
        text_index_map.add_with_ids(np.array(text_vectors).astype('float32'), np.array(text_ids))
    
    # 创建图像向量索引
    image_index = faiss.IndexFlatL2(IMAGE_EMBEDDING_DIM)
    image_index_map = faiss.IndexIDMap(image_index)
    
    # 提取所有图片类型的 ID
    image_ids = [m["id"] for m in metadata_store if m["type"] == "image"]
    if image_vectors:  # 只有当有图像向量时才添加到索引
        image_index_map.add_with_ids(np.array(image_vectors).astype('float32'), np.array(image_ids))
    
    print(f"索引构建完成。共索引 {len(text_vectors)} 个文本片段和 {len(image_vectors)} 张图片。")
    
    # 4. 保存到本地
    print(f"正在保存知识库到本地: {save_dir} ...")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存 metadata
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata_store, f)
        
    # 保存 FAISS 索引
    faiss.write_index(text_index_map, text_index_path)
    faiss.write_index(image_index_map, image_index_path)
    
    # 保存文件状态
    with open(file_state_path, 'w') as f:
        json.dump(current_files_state, f)
        
    print("保存完成。")
    
    return metadata_store, text_index_map, image_index_map

# Step3. RAG 问答流程

def rag_ask(query, metadata_store, text_index, image_index, k=3):
    """
    执行完整的 RAG 流程：
    1. 检索：根据用户查询，在向量库中检索相关的文本和图片。
    2. 构建 Prompt：将检索到的上下文组装成 Prompt。
    3. 生成：调用 LLM 生成最终答案。
    
    参数:
        query: 用户的问题
        metadata_store: 元数据存储
        text_index: 文本向量索引
        image_index: 图片向量索引
        k: 检索的最相关文档数量
    """
    print(f"\n--- 收到用户提问: '{query}' ---")
    
    # 步骤 1: 检索
    print("  - 步骤 1: 向量化查询并进行检索...")
    retrieved_context = []
    
    # 1.1 文本检索
    # 将用户查询转换为文本向量
    query_text_vec = np.array([get_text_embedding(query)]).astype('float32')
    # 在文本索引中搜索最近的 k 个邻居
    distances, text_ids = text_index.search(query_text_vec, k)
    
    # 遍历搜索结果
    for i, doc_id in enumerate(text_ids[0]):
        if doc_id != -1:  # -1 表示未找到
            # 通过 ID 在元数据中查找对应的文档片段
            match = next((item for item in metadata_store if item["id"] == doc_id), None)
            if match:
                retrieved_context.append(match)
                print(f"    - 文本检索命中 (ID: {doc_id}, 距离: {distances[0][i]:.4f})")

    # 1.2 图像检索 (使用 CLIP 文本编码器进行以文搜图)
    # 简单判断是否需要检索图片 (包含特定关键词)
    # 这里的关键词可以根据实际需求扩展
    image_keywords = ["海报", "图片", "长什么样", "看看", "万圣节", "聚在一起"]
    if any(keyword in query.lower() for keyword in image_keywords):
        print("  - 检测到图像查询关键词，执行图像检索...")
        # 将用户查询转换为 CLIP 文本向量
        query_clip_vec = np.array([get_clip_text_embedding(query)]).astype('float32')
        # 在图片索引中搜索最相关的 1 张图
        distances, image_ids = image_index.search(query_clip_vec, 1)
        
        for i, doc_id in enumerate(image_ids[0]):
            if doc_id != -1:
                match = next((item for item in metadata_store if item["id"] == doc_id), None)
                if match:
                    # 将图片信息和 OCR 内容加入上下文
                    context_text = f"找到一张相关图片，图片路径: {match['path']}。图片上的文字是: '{match['ocr']}'"
                    # 添加特殊的 image_context 类型，方便后续处理
                    retrieved_context.append({"type": "image_context", "content": context_text, "metadata": match})
                    print(f"    - 图像检索命中 (ID: {doc_id}, 距离: {distances[0][i]:.4f})")
    
    # 步骤 2: 构建 Prompt 并生成答案
    print("  - 步骤 2: 构建 Prompt...")
    context_str = ""
    # 拼接所有检索到的上下文
    for i, item in enumerate(retrieved_context):
        content = item.get('content', '')
        source = item.get('metadata', {}).get('source', item.get('source', '未知来源'))
        context_str += f"背景知识 {i+1} (来源: {source}):\n{content}\n\n"
        
    # 构建最终的 Prompt
    prompt = f"""你是一个迪士尼客服助手。请根据以下背景知识，用友好和专业的语气回答用户的问题。请只使用背景知识中的信息，不要自行发挥。

[背景知识]
{context_str}
[用户问题]
{query}
"""
    print("--- Prompt Start ---")
    print(prompt)
    print("--- Prompt End ---")
    
    # 步骤 3: 调用 LLM 生成最终答案
    print("\n  - 步骤 3: 调用 LLM 生成最终答案...")
    try:
        # 调用大模型生成回答
        completion = client.chat.completions.create(
            model="qwen-plus", # 使用通义千问 Plus 模型，性能较强
            messages=[
                {"role": "system", "content": "你是一个迪士尼客服助手。"},
                {"role": "user", "content": prompt}
            ]
        )
        final_answer = completion.choices[0].message.content
        
        # 答案后处理：如果上下文中包含图片，在回答最后提示用户
        image_path_found = None
        for item in retrieved_context:
            if item.get("type") == "image_context":
                image_path_found = item.get("metadata", {}).get("path")
                break
        
        if image_path_found:
            final_answer += f"\n\n(同时，我为您找到了相关图片，路径为: {image_path_found})"

    except Exception as e:
        final_answer = f"调用LLM时出错: {e}"

    print("\n--- 最终答案 ---")
    print(final_answer)
    return final_answer

# --- 主函数 ---
if __name__ == "__main__":
    # 1. 构建知识库 (这是一个一次性的离线过程，但在本脚本中每次运行都会重新构建)
    # 传入文档目录和图片目录
    metadata_store, text_index, image_index = build_knowledge_base(DOCS_DIR, IMG_DIR)
    
    # 2. 开始问答演示
    print("\n=============================================")
    print("迪士尼客服RAG助手已准备就绪，开始模拟提问。")
    print("=============================================")
    
    # 案例1: 文本问答 - 测试对 Word 文档内容的检索
    rag_ask(
        query="我想了解一下迪士尼门票的退款流程",
        metadata_store=metadata_store,
        text_index=text_index,
        image_index=image_index
    )
    
    print("\n---------------------------------------------\n")
    
    # 案例2: 多模态问答 - 测试对图片内容的检索 (基于 CLIP 的语义匹配)
    rag_ask(
        query="最近万圣节的活动海报是什么",
        metadata_store=metadata_store,
        text_index=text_index,
        image_index=image_index
    )
    
    print("\n---------------------------------------------\n")
    # 案例3: 年卡相关问答 - 测试对表格或特定规则的检索
    rag_ask(
        query="迪士尼年卡有什么优惠",
        metadata_store=metadata_store,
        text_index=text_index,
        image_index=image_index
    )
    
    print("\n---------------------------------------------\n")
    print("")
