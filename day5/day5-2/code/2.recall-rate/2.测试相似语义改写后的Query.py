# =========================
# 文件名：test_multi_query_rewrite.py
# 目标：测试并输出 MultiQueryRetriever 改写后的 Query
# =========================

from dotenv import load_dotenv
load_dotenv()

from langchain_community.llms import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from typing import List


# =========================
# 配置部分
# =========================

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请先设置环境变量 DASHSCOPE_API_KEY")

CONFIG = {
    "llm_model": "deepseek-v3",
    "embedding_model": "text-embedding-v1",
    "vector_db_path": "./vector_db",
}


# =========================
# 方法一：使用 MultiQueryRetriever 内部机制
# =========================

def test_multi_query_with_retriever(original_query: str) -> List[str]:
    """
    使用 MultiQueryRetriever 获取改写后的 Query
    
    参数:
        original_query: 原始用户问题
    
    返回:
        改写后的 Query 列表
    """
    print("=" * 60)
    print("方法一：通过 MultiQueryRetriever 获取改写 Query")
    print("=" * 60)
    
    # 加载向量库（需要预先存在）
    embeddings = DashScopeEmbeddings(
        model=CONFIG["embedding_model"],
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
    
    vectorstore = FAISS.load_local(
        CONFIG["vector_db_path"],
        embeddings,
        allow_dangerous_deserialization=True,
    )
    
    # 创建 LLM
    llm = Tongyi(
        model_name=CONFIG["llm_model"],
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
    
    # 创建 MultiQueryRetriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        llm=llm,
    )
    
    # 获取改写后的 Query（通过访问内部属性）
    generated_queries = []
    
    # 方法：手动执行 LLM 改写
    from langchain.retrievers.multi_query import LineListOutputParser
    
    # 默认 Prompt 模板
    DEFAULT_QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is 
to generate 3 different versions of the given user question to retrieve 
relevant documents from a vector database. By generating multiple perspectives 
on the user question, your goal is to help the user overcome some of the 
limitations of the distance-based similarity search. Provide these alternative 
questions separated by newlines. Original question: {question}""",
    )
    
    # 执行改写
    query_generator = DEFAULT_QUERY_PROMPT | llm | StrOutputParser()
    generated_text = query_generator.invoke({"question": original_query})
    
    # 解析生成的 Query（按行分割）
    generated_queries = [
        q.strip() for q in generated_text.split("\n") 
        if q.strip() and not q.strip().startswith(tuple("0123456789."))
    ]
    
    # 清理编号
    cleaned_queries = []
    for q in generated_queries:
        # 移除行首的编号（如 "1. " 或 "1、"）
        import re
        cleaned = re.sub(r"^[1-9][.、]\s*", "", q.strip())
        if cleaned:
            cleaned_queries.append(cleaned)
    
    return cleaned_queries


# =========================
# 方法二：直接使用自定义 Prompt 调用 LLM
# =========================

def test_multi_query_with_custom_prompt(original_query: str, num_queries: int = 3) -> List[str]:
    """
    使用自定义 Prompt 直接调用 LLM 生成改写 Query
    
    参数:
        original_query: 原始用户问题
        num_queries: 生成 Query 的数量
    
    返回:
        改写后的 Query 列表
    """
    print("\n" + "=" * 60)
    print("方法二：使用自定义 Prompt 直接调用 LLM")
    print("=" * 60)
    
    # 创建 LLM
    llm = Tongyi(
        model_name=CONFIG["llm_model"],
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
    
    # 自定义 Prompt 模板（中文版）
    QUERY_REWRITE_PROMPT = PromptTemplate(
        input_variables=["question", "num"],
        template="""你是一位专业的文档检索专家。你的任务是根据用户问题，生成多个不同版本的查询。

要求：
1. 生成 {num} 个不同版本的查询
2. 每个版本应该从不同角度表达相同的意思
3. 使用不同的关键词、句式和表述方式
4. 保持原问题的核心意图不变
5. 适合向量相似度检索

原始问题：{question}

请生成 {num} 个不同版本的查询（每行一个，不要编号）：
""",
    )
    
    # 执行改写
    query_generator = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
    generated_text = query_generator.invoke({
        "question": original_query,
        "num": num_queries,
    })
    
    # 解析生成的 Query（按行分割）
    import re
    generated_queries = []
    for line in generated_text.split("\n"):
        line = line.strip()
        if line:
            # 移除行首的编号
            cleaned = re.sub(r"^[1-9][.、]\s*", "", line)
            # 移除项目符号
            cleaned = re.sub(r"^[-*•]\s*", "", cleaned)
            if cleaned and len(cleaned) > 5:  # 过滤太短的行
                generated_queries.append(cleaned)
    
    return generated_queries


# =========================
# 方法三：带详细输出的完整测试
# =========================

def test_multi_query_detailed(original_query: str, num_queries: int = 3):
    """
    完整测试：展示改写过程、输出所有改写 Query、并执行检索
    
    参数:
        original_query: 原始用户问题
        num_queries: 生成 Query 的数量
    """
    print("\n" + "#" * 60)
    print("#  MultiQuery Query 改写测试")
    print("#" * 60)
    
    print(f"\n[原始问题]")
    print(f"    {original_query}")
    
    # 生成改写 Query
    print(f"\n[生成改写 Query]")
    print(f"    目标数量：{num_queries} 个")
    
    rewritten_queries = test_multi_query_with_custom_prompt(original_query, num_queries)
    
    print(f"\n[改写结果]")
    print(f"    实际生成：{len(rewritten_queries)} 个")
    print()
    for i, q in enumerate(rewritten_queries, 1):
        print(f"    Query {i}: {q}")
    
    # 尝试执行检索（如果向量库存在）
    print(f"\n[执行检索测试]")
    try:
        embeddings = DashScopeEmbeddings(
            model=CONFIG["embedding_model"],
            dashscope_api_key=DASHSCOPE_API_KEY,
        )
        
        vectorstore = FAISS.load_local(
            CONFIG["vector_db_path"],
            embeddings,
            allow_dangerous_deserialization=True,
        )
        
        print(f"    向量库：已加载 ({CONFIG['vector_db_path']})")
        print()
        
        # 对每个改写 Query 执行检索
        all_docs = []
        for i, q in enumerate(rewritten_queries, 1):
            docs = vectorstore.similarity_search(q, k=2)
            all_docs.extend(docs)
            print(f"    Query {i} 检索到 {len(docs)} 个文档")
        
        # 去重
        unique_docs = []
        seen = set()
        for doc in all_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
        
        print(f"\n    合并后唯一文档数：{len(unique_docs)} 个")
        
    except FileNotFoundError:
        print(f"    向量库未找到：{CONFIG['vector_db_path']}")
        print(f"    跳过检索测试")
    except Exception as e:
        print(f"    检索测试出错：{e}")
    
    print("\n" + "#" * 60)
    
    return rewritten_queries


# =========================
# 主程序
# =========================

def main():
    """
    主程序：测试多个问题的 Query 改写
    """
    print("\n")
    print("*" * 60)
    print("*  MultiQuery Query 改写测试工具")
    print("*" * 60)
    
    # 测试问题列表
    test_queries = [
        "客户经理被投诉了，投诉一次扣多少分？",
        "客户经理每年评聘申报时间是怎样的？",
        "客户经理的考核标准是什么？",
    ]
    
    # 逐个测试
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'=' * 60}")
        print(f"测试 {i}/{len(test_queries)}")
        print("=" * 60)
        
        rewritten_queries = test_multi_query_detailed(query, num_queries=3)
        
        # 保存结果到文件
        output_file = f"2.query_rewrite_results_{i}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"原始问题：{query}\n\n")
            f.write("改写后的 Query:\n")
            for j, q in enumerate(rewritten_queries, 1):
                f.write(f"  {j}. {q}\n")
        print(f"\n[保存] 结果已保存到：{output_file}")
    
    print("\n\n")
    print("*" * 60)
    print("*  所有测试完成！")
    print("*" * 60)
    print("\n")


if __name__ == "__main__":
    main()
