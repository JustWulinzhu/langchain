from app.rag.vector_store import get_vector_store

# 定义 从向量数据库检索 方法
# top_k: int = 5：控制返回最相似的文档片段数量，默认返回前 5 条
# 注意这里检索出来的内容不是AI的答案，而是AI回答用户问题的依据，AI会依据这个组织自然语言回答用户的问题
def search_document(question: str, top_k: int = 5):
    """
    根据用户问题检索文档
    :param question:
    :param top_k:
    :return:
    """
    vector_store = get_vector_store()
    # similarity_search_with_score：这是 LangChain 向量库的核心检索方法，执行以下两步操作：
    # 问题向量化：自动调用初始化 vector_store 时绑定的 Embedding 模型（智谱 Embedding），把 question 字符串转换成高维向量。
    # 相似度计算：在 Chroma 数据库中计算该问题向量与所有已存文档向量的距离，按相关性从高到低排序，取出前 k（即 top_k）条结果。
    # results 的数据结构：返回一个包含元组的列表，格式为 [(Document对象1, 距离分数1), (Document对象2, 距离分数2), ...]
    # 数值越小，代表越相似（0 表示完全一致）
    results = vector_store.similarity_search_with_score(question, k = top_k)
    documents = []

    for doc, score in results:
        documents.append({
            "content": doc.page_content, # 切分出来的具体文本内容本身
            "source": doc.metadata.get("source", ""), # 这篇内容是从哪个文件里提取出来的（文件路径或文件名）
            "page": doc.metadata.get("page"), # 这段文字在原 PDF 文件中的第几页。
            "score": float(score)
        })

    return documents