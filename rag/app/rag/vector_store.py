# 本地向量数据库的操作工具
from langchain_chroma import Chroma
from langchain_core.documents import Document
# 从项目配置文件 app/config.py 中导入的本地目录路径，用于指定向量数据在磁盘上的持久化存储位置
from app.config import CHROMA_DIR
from app.rag.embedding import get_embedding_model
from typing import List
import hashlib

# 计算文件hash值
def calculate_file_sha256(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    # 打开这个文件。
    # "rb" 的意思是以纯二进制字节模式只读打开（因为 PDF 不是纯文本，必须按底层二进制读取），并给打开的文件起个临时外号叫 f。
    # with 的好处是用完会自动安全关闭文件，自动是否文件资源
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# 根据文件hash值判断文件内容是否重复
# 作用：去 Chroma 数据库里查一下，当前这个知识库里是不是已经存过一模一样的文件了。
# 组合条件 $and：必须同时满足 file_hash 相同 且 knowledge_base_id 相同。因为用户 A 上传了一份《微信支付文档》，用户 B 在自己的独立知识库里也上传同一份文档时，应该允许用户 B 正常入库，不能因为库里有别人的数据就拦截用户 B
# limit=1：只要能查到 1 条切片，就足以证明整个文件已经入库了，没必要把几百条切片全查出来，查询速度极快。
def is_duplicate_file(vector_store: Chroma, file_hash: str, knowledge_id: str) -> bool:
    # 执行一次数据库查询（.get）
    is_exist = vector_store.get(
        where = {
            "$and": [
                {"file_hash": {"$eq": file_hash}}, # 数据库中的file_hash必须等于传进来的file_hash
                {"knowledge_base_id": {"$eq": knowledge_id}} # 数据库中的知识库ID必须等于传进来的知识库ID
            ]
        },
        limit = 1
    )
    return len(is_exist['ids']) > 0

# 在 knowledge_id 这个知识库里面，删除文档 doc_id 的向量数据
# 文档内容改了之后切片内容发生变化，必须全量删除重新切片才可以，要不然废弃的数据会影响检索效果
def delete_document_vectors(vector_store: Chroma, knowledge_id: str, doc_id: str):
    # 在数据库执行删除动作（.delete）
    vector_store.delete(
        where = {
            "$and": [
                {"knowledge_base_id": {"$eq": knowledge_id}},
                {"doc_id": {"$eq": doc_id}}
            ]
        }
    )

# 作用：创建一个 Chroma 向量数据库实例。
def get_vector_store() -> Chroma:
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
    # 把这个连接好的数据库交出去（返回结果），供后面的代码使用。
    return vector_store

def save_chunks(
        chunks,
        file_path: str,
        knowledge_id: str,
        doc_id: str = "",
        batch_size: int = 50) -> List[str]:
    """
    将文本切片保存到向量数据库
    :param chunks: 切分后的 Document 文本块列表
    :param file_path: 原始的pdf路径（用于计算文件hash值）
    :param knowledge_id: 知识库ID，实现多业务隔离
    :param doc_id: 文档唯一ID 默认使用文件名
    :param batch_size: 写入向量数据库的批次大小，智谱默认上限64
    """

    if not chunks:
        return []

    # 连上数据库，获取数据库实例
    vector_store = get_vector_store()

    # 计算文件hash值
    file_hash = calculate_file_sha256(file_path)

    # 检查文档内容是否完全一致（相同的知识库，相同的文件hash）
    is_duplicate = is_duplicate_file(vector_store, file_hash, knowledge_id)
    if is_duplicate:
        print(f"当前知识库（{knowledge_id}）中有完全相同的文件，文件hash（{file_hash}）完全相同")
        return []

    # 代码走到这里，说明文件内容不重复（要不是全新的文件，要不是文件内容更新了）
    # 清理旧向量
    # 如果未指定 doc_id，默认用文件名作为文档逻辑标识
    doc_id = doc_id or file_path.split("/")[-1]
    delete_document_vectors(vector_store, knowledge_id, doc_id)

    # 含义：挨个遍历所有的文本块（同时拿到序号 index 和文本块 chunk）。
    # 在每个文本块自带的附加信息字典（metadata）里，写入 4 个新标签：知识库 ID、文件指纹、文档 ID、以及它是第几块。
    stable_ids = []
    for index, chunk in enumerate(chunks):
        # 补充元数据，供后续删除和检索过滤使用
        chunk.metadata["knowledge_base_id"] = knowledge_id
        chunk.metadata["file_hash"] = file_hash
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["chunk_index"] = index

        # 3. 生成稳定 ID：知识库ID + 文档ID + 切片序号
        chunk_id = f"{knowledge_id}::{doc_id}::{index}"
        stable_ids.append(chunk_id)

    # 步骤 D：分批写入 Chroma（显式传入 ids 参数）
    all_saved_ids = []
    for i in range(0, len(chunks), batch_size):
        batch_docs = chunks[i: i + batch_size]
        batch_ids = stable_ids[i: i + batch_size]

        ids = vector_store.add_documents(batch_docs, ids = batch_ids)
        all_saved_ids.extend(ids)

    print(f"成功保存了{len(all_saved_ids)}个文本块到知识库{knowledge_id}中")
    return all_saved_ids