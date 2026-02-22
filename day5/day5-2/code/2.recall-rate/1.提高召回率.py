# =========================
# 目标：把 PDF 做成一个"可语义搜索 + 可问答 + 可溯源 + 高召回率"的知识库系统
# 流程：
#   PDF -> 切块（Chunk）-> 向量化（Embedding）-> FAISS 向量数据库
#   -> 用户提问 -> MultiQuery 检索（LLM 改写问题提高召回率）-> 大模型生成答案 -> 输出答案 + 来源页码
#  本质是相似语义改写生成多个 Query
# =========================

from dotenv import load_dotenv
load_dotenv()

# PDF 读取器
from PyPDF2 import PdfReader

# LangChain 新版问答链（推荐）
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 文本切分器
from langchain.text_splitter import RecursiveCharacterTextSplitter

# DashScope 向量模型 & 通义千问 LLM
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi

# FAISS 向量库
from langchain_community.vectorstores import FAISS

# MultiQueryRetriever（核心：提高检索召回率）
from langchain.retrievers import MultiQueryRetriever

# 类型注解 & 工具
from typing import List, Dict, Any, Set, Tuple
import os

# 成本追踪（可选）
from langchain_community.callbacks.manager import get_openai_callback


# =========================
# 配置部分
# =========================

# 读取 DashScope API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请先设置环境变量 DASHSCOPE_API_KEY")

# 配置参数
CONFIG = {
    "pdf_path": "./浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf",
    "vector_db_path": "./vector_db",
    "embedding_model": "text-embedding-v1",
    "llm_model": "deepseek-v3",  # 也可用 qwen-turbo / qwen-max 等
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "top_k": 3,  # 检索返回的文档块数量
    "multi_query_enabled": True,  # 是否启用 MultiQuery 检索
}


# =========================
# 核心功能函数
# =========================

def extract_pages_text(pdf_path: str) -> List[Dict[str, Any]]:
    """
    逐页提取 PDF 文本，并保留页码信息（用于溯源）
    
    参数:
        pdf_path: PDF 文件路径
    
    返回:
        pages: 列表，每个元素包含 page_number 和 page_text
    """
    pdf_reader = PdfReader(pdf_path)
    pages: List[Dict[str, Any]] = []
    
    for page_number, page in enumerate(pdf_reader.pages, start=1):
        page_text = page.extract_text() or ""
        
        if not page_text.strip():
            print(f"[警告] 第 {page_number} 页未提取到文本（可能是图片或无文本层）")
        
        pages.append({
            "page_number": page_number,
            "page_text": page_text,
        })
    
    print(f"[完成] 成功提取 {len(pages)} 页文本")
    return pages


def split_pages_to_chunks(pages: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    把每页文本切分成多个 chunk，并给每个 chunk 绑定来源页码 metadata
    
    参数:
        pages: extract_pages_text 返回的 pages 列表
    
    返回:
        texts: 所有 chunk 文本列表
        metadatas: 对应的 metadata 列表（含页码）
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ".", " ", ""],  # 优先按段落/换行/句号切分
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
        length_function=len,
    )
    
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    
    for page in pages:
        page_number = page["page_number"]
        page_text = page["page_text"]
        
        if not page_text.strip():
            continue
        
        # 对单页文本切分
        page_chunks = text_splitter.split_text(page_text)
        
        for chunk_text in page_chunks:
            texts.append(chunk_text)
            # 关键：metadata 中绑定页码，用于后续溯源
            metadatas.append({"page": page_number, "source": os.path.basename(CONFIG["pdf_path"])})
    
    print(f"[完成] 文本被分割成 {len(texts)} 个块")
    return texts, metadatas


def build_or_load_faiss(
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    save_dir: str,
    dashscope_api_key: str,
) -> FAISS:
    """
    构建或加载 FAISS 向量数据库
    
    参数:
        texts: chunk 文本列表
        metadatas: 对应的 metadata 列表
        save_dir: 向量库保存目录
        dashscope_api_key: DashScope API Key
    
    返回:
        knowledge_base: FAISS 向量数据库对象
    """
    embeddings = DashScopeEmbeddings(
        model=CONFIG["embedding_model"],
        dashscope_api_key=dashscope_api_key,
    )
    
    # 检查是否已有保存的向量库
    if os.path.exists(save_dir) and os.path.isdir(save_dir) and os.listdir(save_dir):
        print(f"[目录] 发现现有向量数据库：{save_dir}")
        knowledge_base = FAISS.load_local(
            save_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"[完成] 已加载向量数据库")
        return knowledge_base
    
    # 创建新的向量库
    print(f"[构建] 开始构建向量数据库（切块 -> 向量化 -> 写入 FAISS）...")
    knowledge_base = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    
    # 保存到本地
    os.makedirs(save_dir, exist_ok=True)
    knowledge_base.save_local(save_dir)
    print(f"[完成] 向量数据库构建并保存完成：{save_dir}")
    
    return knowledge_base


def create_retriever(
    vectorstore: FAISS,
    llm: Tongyi,
    multi_query_enabled: bool = True,
    search_kwargs: Dict[str, Any] = None,
):
    """
    创建检索器（支持 MultiQuery 模式提高召回率）
    
    参数:
        vectorstore: FAISS 向量库
        llm: 大语言模型（用于查询改写）
        multi_query_enabled: 是否启用 MultiQuery 检索
        search_kwargs: 检索参数（如 k 值）
    
    返回:
        retriever: 检索器对象
    """
    if search_kwargs is None:
        search_kwargs = {"k": CONFIG["top_k"]}
    
    # 创建基础检索器
    base_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    if multi_query_enabled:
        print(f"[启动] 启用 MultiQuery 检索模式（LLM 自动改写问题，提高召回率）")
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
        )
    else:
        print(f"[提示] 使用标准相似度检索模式")
        retriever = base_retriever
    
    return retriever


def create_qa_chain(llm: Tongyi) -> Any:
    """
    创建问答链（使用新版 LangChain API）
    
    参数:
        llm: 大语言模型
    
    返回:
        document_chain: 问答链对象
    """
    # 自定义 Prompt 模板（可调整）
    prompt = ChatPromptTemplate.from_template("""你是一个专业的文档问答助手。请根据以下检索到的上下文（Context）回答问题。

要求：
1. 如果上下文中没有答案，请直接说"未找到相关信息"，不要编造
2. 回答要准确、简洁，引用具体数据时请注明来源
3. 如果上下文中有多个相关信息，请综合整理后回答

<context>
{context}
</context>

问题：{input}

回答：""")
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    return document_chain


def answer_with_rag(
    retriever,
    query: str,
    document_chain,
    enable_cost_tracking: bool = True,
) -> Tuple[str, Set[int]]:
    """
    使用 RAG 流程回答问题：检索 -> LLM 生成 -> 输出答案 + 来源页码
    
    参数:
        retriever: 检索器（MultiQueryRetriever 或基础检索器）
        query: 用户问题
        document_chain: 问答链
        enable_cost_tracking: 是否启用成本追踪
    
    返回:
        response_text: 生成的答案
        unique_pages: 来源页码集合
    """
    # 执行检索
    print(f"\n[检索] 正在检索相关文档...")
    docs = retriever.invoke(query)
    print(f"[完成] 找到 {len(docs)} 个相关文档块")
    
    # 执行问答
    print(f"[AI] 正在生成答案...")
    if enable_cost_tracking:
        with get_openai_callback() as cost:
            response_text = document_chain.invoke({"context": docs, "input": query})
            print(f"[成本] API 调用成本：{cost}")
    else:
        response_text = document_chain.invoke({"context": docs, "input": query})
    
    # 收集来源页码
    unique_pages: Set[int] = set()
    for doc in docs:
        page = (doc.metadata or {}).get("page")
        if page is not None:
            unique_pages.add(page)
    
    return response_text, unique_pages


def print_results(
    query: str,
    response_text: str,
    unique_pages: Set[int],
    docs: List[Any] = None,
):
    """
    打印问答结果和溯源信息
    """
    print("\n" + "=" * 60)
    print(f"[查询] {query}")
    print("=" * 60)
    
    print("\n[答案]")
    print("-" * 60)
    print(response_text)
    print("-" * 60)
    
    print(f"\n[来源] 页码（共 {len(unique_pages)} 页）:")
    for page in sorted(unique_pages):
        print(f"   - 第 {page} 页")
    
    # 可选：打印文档预览
    if docs:
        print(f"\n[预览] 检索到的文档块（共 {len(docs)} 个）:")
        for i, doc in enumerate(docs[:5], 1):  # 只显示前 5 个
            page = (doc.metadata or {}).get("page", "未知")
            preview = doc.page_content.strip()[:80].replace('\n', ' ')
            print(f"   [{i}] 第{page}页：{preview}...")
        if len(docs) > 5:
            print(f"   ... 还有 {len(docs) - 5} 个文档块")
    
    print("=" * 60)


# =========================
# 主程序入口
# =========================

def main():
    """
    主程序：串起完整 RAG 流程
    """
    print("\n" + "=" * 60)
    print("   PDF 智能问答系统（MultiQuery + 高召回率 + 可溯源）")
    print("=" * 60 + "\n")
    
    # 1. 提取 PDF 文本
    print("[步骤 1/5] 提取 PDF 文本...")
    pages = extract_pages_text(CONFIG["pdf_path"])
    
    # 2. 文本切块
    print("\n[步骤 2/5] 文本切块...")
    texts, metadatas = split_pages_to_chunks(pages)
    
    # 3. 构建/加载向量库
    print("\n[步骤 3/5] 构建/加载向量数据库...")
    knowledge_base = build_or_load_faiss(
        texts=texts,
        metadatas=metadatas,
        save_dir=CONFIG["vector_db_path"],
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
    
    # 4. 初始化 LLM 和检索器
    print("\n[步骤 4/5] 初始化 LLM 和检索器...")
    llm = Tongyi(
        model_name=CONFIG["llm_model"],
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
    
    retriever = create_retriever(
        vectorstore=knowledge_base,
        llm=llm,
        multi_query_enabled=CONFIG["multi_query_enabled"],
    )
    
    # 5. 创建问答链
    print("\n[步骤 5/5] 创建问答链...")
    document_chain = create_qa_chain(llm)
    
    # 6. 执行查询
    print("\n[提示] 准备就绪，开始问答...\n")
    
    # 示例问题列表
    queries = [
        "客户经理被投诉了，投诉一次扣多少分？",
        "客户经理每年评聘申报时间是怎样的？",
        "客户经理的考核标准是什么？",
        # 你可以添加更多问题
    ]
    
    for query in queries:
        response_text, unique_pages = answer_with_rag(
            retriever=retriever,
            query=query,
            document_chain=document_chain,
            enable_cost_tracking=True,
        )
        
        # 获取检索到的文档（用于预览）
        docs = retriever.invoke(query)
        
        print_results(
            query=query,
            response_text=response_text,
            unique_pages=unique_pages,
            docs=docs,
        )
        
        print("\n")
    
    print("[完成] 所有查询完成！")


if __name__ == "__main__":
    main()
