# =========================
# 目标：把 PDF 做成一个“可语义搜索 + 可问答 + 可溯源”的知识库系统
# 流程：
#   PDF
#    ↓
#   切块（Chunk）
#    ↓
#   向量化（Embedding）
#    ↓
#   FAISS 向量数据库
#    ↓
#   用户提问
#    ↓
#   相似度检索（TopK）
#    ↓
#   大模型生成答案
#    ↓
#   输出答案 + 来源页码
# =========================

from dotenv import load_dotenv
load_dotenv()

# 导入 PDF 读取器：用于逐页读取 PDF 文本
from PyPDF2 import PdfReader


# 导入 LangChain 的问答链（新版推荐使用 create_stuff_documents_chain）
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 导入文本切分器：用于把长文本切成适合向量化与检索的小块
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 导入 DashScope 向量模型封装：把文本转成向量（Embedding）
from langchain_community.embeddings import DashScopeEmbeddings

# 导入 FAISS 向量库：本地向量数据库（适合做相似度检索）
from langchain_community.vectorstores import FAISS

# 导入通义千问（DashScope）LLM：用于把“检索到的片段 + 问题”变成自然语言答案
from langchain_community.llms import Tongyi

# 导入类型注解：让函数输入输出更清晰
from typing import List, Dict, Any

# 导入 OS：用于读取环境变量、处理文件路径
import os


# 读取 DashScope API Key（向量化与大模型调用都需要）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 如果没有设置 API Key，就直接报错（避免后面运行到一半才出错）
if not DASHSCOPE_API_KEY:
    raise ValueError("请先设置环境变量 DASHSCOPE_API_KEY，例如：export DASHSCOPE_API_KEY=你的Key")


def extract_pages_text(pdf_path: str) -> List[Dict[str, Any]]:
    """
    逐页提取 PDF 文本，并保留页码信息，便于“可溯源”

    参数:
        pdf_path: PDF 文件路径

    返回:
        pages: 一个列表，每个元素包含 page_number 与 page_text
    """
    # 创建 PDF Reader（PyPDF2 会解析 PDF 的页面结构）
    pdf_reader = PdfReader(pdf_path)

    # 用列表保存每一页的文本（以及页码）
    pages: List[Dict[str, Any]] = []

    # enumerate 从 1 开始计数，让页码与真实 PDF 页码一致
    for page_number, page in enumerate(pdf_reader.pages, start=1):
        # 提取当前页文本（某些 PDF 可能是扫描件，提取不到文字）
        page_text = page.extract_text() or ""

        # 如果这一页提取不到文本，打印提示（不影响继续处理）
        if not page_text.strip():
            print(f"提示：第 {page_number} 页未提取到文本（可能是图片或无文本层）。")

        # 把页码与文本保存起来
        pages.append(
            {
                "page_number": page_number,
                "page_text": page_text,
            }
        )

    # 返回所有页面的文本与页码
    return pages


def split_pages_to_chunks(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    把每页文本切分成多个 chunk，并给每个 chunk 绑定来源页码，实现“可溯源”
    
    参数:
        pages: extract_pages_text 返回的 pages 列表
    
    返回:
        chunks_with_meta: 一个列表，每个元素包含 chunk_text 与 metadata（含页码）
    """
    # 创建递归字符切分器：优先按段落/换行/句号/空格等分隔，再退化到按字符切
    # 调整参数以减少碎片化
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", ".", " ", ""],  # 分隔符优先级，增加中文句号
        chunk_size=1000,  # 减小每个 chunk 的目标长度（字符数），提高相关性
        chunk_overlap=100,  # 适当的重叠
        length_function=len,  # 用 len 统计长度
    )

    # 用列表保存切分后的 chunk（附带 metadata）
    chunks_with_meta: List[Dict[str, Any]] = []

    # 逐页切分：这样每个 chunk 天然知道“来自第几页”
    for page in pages:
        # 取出页码
        page_number = page["page_number"]

        # 取出该页文本
        page_text = page["page_text"]

        # 跳过空页（避免产生空 chunk）
        if not page_text.strip():
            continue

        # 对单页文本做切分，得到多个 chunk
        page_chunks = text_splitter.split_text(page_text)

        # 把每个 chunk 记录下来，并写入 metadata（页码）
        for chunk_text in page_chunks:
            chunks_with_meta.append(
                {
                    "text": chunk_text,
                    "metadata": {"page": page_number},  # 关键：用于溯源输出
                }
            )

    # 返回包含 chunk 与 metadata 的列表
    return chunks_with_meta


def build_or_load_faiss(
    chunks_with_meta: List[Dict[str, Any]],
    save_dir: str,
    dashscope_api_key: str,
) -> FAISS:
    """
    构建或加载 FAISS 向量数据库（用于语义检索）

    参数:
        chunks_with_meta: 切分后的 chunk + metadata 列表
        save_dir: 向量库保存目录（本地持久化）
        dashscope_api_key: DashScope API Key

    返回:
        knowledge_base: FAISS 向量数据库对象
    """
    # 初始化 Embedding 模型（把文本 -> 向量）
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",  # DashScope 的文本向量模型
        dashscope_api_key=dashscope_api_key,  # API Key
    )

    # 如果本地已经存在向量库目录，则直接加载（避免重复向量化，节省时间/费用）
    if os.path.exists(save_dir) and os.path.isdir(save_dir) and os.listdir(save_dir):
        # 从磁盘加载 FAISS 向量库
        knowledge_base = FAISS.load_local(
            save_dir,
            embeddings,
            allow_dangerous_deserialization=True,  # 允许反序列化（示例环境常用）
        )

        # 打印提示：说明走的是“加载”路径
        print(f"已从磁盘加载向量数据库：{save_dir}")

        # 返回已加载的向量库
        return knowledge_base

    # 如果本地不存在向量库，则走“构建”路径
    print("未发现本地向量库，开始构建（切块 -> 向量化 -> 写入 FAISS）...")

    # 提取所有 chunk 文本
    texts = [item["text"] for item in chunks_with_meta]

    # 提取所有 metadata（用于溯源）
    metadatas = [item["metadata"] for item in chunks_with_meta]

    # 通过 FAISS.from_texts 构建向量库（内部会对每个 chunk 调 embedding）
    knowledge_base = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存向量库到本地（下次可以直接 load_local）
    knowledge_base.save_local(save_dir)

    # 打印提示：构建完成
    print(f"向量数据库构建并保存完成：{save_dir}")

    # 返回向量库
    return knowledge_base


def answer_with_rag(
    knowledge_base: FAISS,
    query: str,
    dashscope_api_key: str,
    top_k: int = 10,
) -> None:
    """
    使用 RAG 流程回答问题：相似度检索 + LLM 生成 + 输出溯源页码

    参数:
        knowledge_base: FAISS 向量库
        query: 用户问题
        dashscope_api_key: DashScope API Key
        top_k: 检索返回的相关 chunk 数量
    """
    # 初始化大模型（Tongyi 是 DashScope 的 LLM 封装）
    llm = Tongyi(
        model_name="deepseek-v3",  # 与示例保持一致（也可换成 qwen-turbo 等）
        dashscope_api_key=dashscope_api_key,
    )

    # 1) 相似度检索：从向量库中找出与问题最相关的文本块（Docs）
    docs = knowledge_base.similarity_search(query, k=top_k)

    """ 
        旧版本是以下这样的：
        
        # 导入 LangChain 的问答链：用于把检索到的内容“塞给大模型”生成答案
        # from langchain.chains.question_answering import load_qa_chain 「已废弃」

        # 2) 加载问答链：把 docs 作为上下文，让大模型生成最终答案
        chain = load_qa_chain(llm, chain_type="stuff")

        # 3) 调用链执行问答：传入“检索文档 + 用户问题”
        response = chain.invoke({"input_documents": docs, "question": query})

        # 4) 输出答案
        print("\n==================== 答案 ====================")
        print(response.get("output_text", "（未返回答案）"))
    """

    # 2) 构建问答链（使用新版 create_stuff_documents_chain）
    # 定义 Prompt 模板
    prompt = ChatPromptTemplate.from_template("""请根据以下检索到的上下文（Context）回答问题。
如果上下文中没有答案，请直接说“未找到相关信息”，不要编造。

<context>
{context}
</context>

问题: {input}
""")

    # 创建文档合并链（Stuff 模式：把所有检索到的 chunk 拼接到 prompt 中）
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 3) 调用链执行问答：传入“检索文档 + 用户问题”
    # 这里的 "context" 对应 docs，"input" 对应 query
    response_text = document_chain.invoke({"context": docs, "input": query})

    # 4) 输出答案
    print("\n==================== 答案 ====================")
    print(response_text)

    # 5) 输出溯源信息：收集 docs 的来源页码（metadata 里的 page）
    print("\n==================== 来源页码 ====================")

    # 逐个文档块读取 metadata 的页码
    for i, doc in enumerate(docs):
        # LangChain Document 的 metadata 是一个 dict
        page = (doc.metadata or {}).get("page", "未知")
        
        # 打印详细的文档来源信息，包括页码和部分内容预览
        print(f"\n[相关文档 {i+1}] 来源页码: {page}")
        content_preview = doc.page_content.strip()[:100].replace('\n', ' ')
        print(f"内容预览: {content_preview}...")

    print("\n==================== 唯一来源页码 ====================")
    # 用集合去重：避免重复打印相同页码
    unique_pages = sorted(list(set((doc.metadata or {}).get("page", 0) for doc in docs if (doc.metadata or {}).get("page") is not None)))
    
    for page in unique_pages:
        print(f"页码: {page}")


def main() -> None:
    """
    主程序入口：串起完整流程
    """
    # 1) 指定 PDF 文件路径（请替换成你自己的 PDF）
    pdf_path = "./浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf"

    # 2) 指定向量库保存目录（用于本地持久化）
    save_dir = "./vector_db"

    # 3) 读取 PDF -> 按页提取文本（保留页码，为溯源做准备）
    pages = extract_pages_text(pdf_path)

    # 4) 切块：把每页文本切成多个 chunk，并给每个 chunk 绑定页码 metadata
    chunks_with_meta = split_pages_to_chunks(pages)

    # 5) 向量化 + 建库：把 chunk 做 embedding，并写入/加载 FAISS 向量库
    knowledge_base = build_or_load_faiss(
        chunks_with_meta=chunks_with_meta,
        save_dir=save_dir,
        dashscope_api_key=DASHSCOPE_API_KEY,
    )

    # 6) 用户提问（你可以改成自己的问题）
    # query = "客户经理多久升到总经理？"
    query = "被投诉了，投诉一次扣多少分"

    # 7) RAG：相似度检索 -> 大模型生成 -> 输出答案 + 来源页码
    answer_with_rag(
        knowledge_base=knowledge_base,
        query=query,
        dashscope_api_key=DASHSCOPE_API_KEY,
        top_k=3,
    )


# Python 脚本入口：直接运行该文件时，从 main() 开始执行
if __name__ == "__main__":
    main()
