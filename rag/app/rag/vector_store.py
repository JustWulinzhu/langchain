from langchain_chroma import Chroma
# 从项目配置文件 app/config.py 中导入的本地目录路径，用于指定向量数据在磁盘上的持久化存储位置
from app.config import CHROMA_DIR
from app.rag.embedding import get_embedding_model

# 作用：创建一个 Chroma 向量数据库实例。
def get_vector_store():
    """
    获取持久化的 Chroma 向量数据库
    """
    embedding = get_embedding_model()
    vector_store = Chroma(
        # collection_name = "api_knowledge"：指定集合名称（类似关系型数据库中的“表名”），用于隔离不同业务的知识库数据。
        collection_name = "api_knowledge",
        # embedding_function = embedding：传入向量化模型。Chroma 在存入文本或执行检索时，会自动调用该模型将文本转换为高维向量
        embedding_function = embedding,
        # 指定本地持久化路径。这表示 Chroma 会把数据持久化到本地目录，
        # 如果该目录下已有 "api_knowledge" 的数据，Chroma 会直接加载；如果不存在，则会自动创建
        persist_directory = CHROMA_DIR
    )
    return vector_store


def save_chunks(chunks, batch_size: int = 50):
    """
    将文本切片分批保存到向量数据库（避免触发智谱 Embedding 单次最多64条的限制）
    """
    vector_store = get_vector_store()
    all_ids = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        ids = vector_store.add_documents(batch)
        all_ids.extend(ids)

    print(f"成功保存了{len(all_ids)}个文本块")
    return all_ids