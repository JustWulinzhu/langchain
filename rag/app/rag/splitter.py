# 引入类型注解，用于标明参数和返回值是列表类型
from typing import List
# 引入 LangChain 的标准文档对象，每个 Document 包含 page_content（正文）和 metadata（元数据字典）
from langchain_core.documents import Document

# 导入切分器
from langchain_text_splitters import (
    # 专门按 Markdown 标题（#、## 等）进行结构化切分的切分器
    MarkdownHeaderTextSplitter,
    # 按字符长度递归切分的通用切分器，防止单段文本过长
    RecursiveCharacterTextSplitter,
)

# 定义文本拆分的函数
# 参数documents: 通常是一个包含多个 langchain_core.documents.base.Document 对象的列表（例如通过 PyPDFLoader、TextLoader 或 DirectoryLoader 加载出来的文档）
def document_splitter(
        documents,
        chunk_size: int = 800, # 设定每个切分块（Chunk）的最大字符数上限，默认 800 字
        chunk_overlap: int = 100, # 设定相邻两个块之间的重叠字符数，默认 100 字，防止一句话在边界被切断导致语义丢失
):
    """
    将文档拆分成多个 chunk
    """
    text_splitter = RecursiveCharacterTextSplitter(
        # 每个 chunk 的目标最大长度（默认按字符数计算，即 len(text)）。这里设为 256 字符，属于比较紧凑的粒度，适合做精确检索，但上下文略
        chunk_size = chunk_size,
        # 相邻两个 chunk 之间的重叠字符数为
        chunk_overlap = chunk_overlap,
        # 分隔符优先级列表
        # 切分器会按列表从前往后尝试用分隔符拆分文本：
        # 1、"\n\n"：优先按空行（段落）切分；
        # 2、"\n"：段落超长则按单换行切分；
        # 3、"。" / "！" / "？"：仍超长则按中文句末标点切分；
        # 4、" "：再按空格切分；
        # 5、""：如果连空格都没有且仍超长，兜底按单个字符硬切分。
        # 拆完之后，切分器还会把相邻且未超过 chunk_size 的小片段合并起来，尽量凑到接近 chunk_size 的大小
        separators = [
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            " ",
            ""
        ]
    )
    # 执行文档切分
    chunks = text_splitter.split_documents(documents)
    # for index, chunk in enumerate(chunks):
    #     print(f"\n========== Chunk {index + 1} ==========")
    #     print("文本：")
    #     print(chunk.page_content)
    #     print("元数据：")
    #     print(chunk.metadata)

    return chunks

def markdown_splitter(
    docs: List[Document],
    chunk_size: int = 800, # 设定每个切分块（Chunk）的最大字符数上限，默认 800 字
    chunk_overlap: int = 100, # 设定相邻两个块之间的重叠字符数，默认 100 字，防止一句话在边界被切断导致语义丢失
) -> List[Document]:
    """
        “先按结构、再按长度” 的两阶段智能切分Markdown文档
    """
    # 按 Markdown 标题切分（保留结构元数据）
    header_splitter = MarkdownHeaderTextSplitter(
        # 定义切分依据的标题级别。遇到一级、二级、三级标题时就切开，并把对应的标题文字分别存入 metadata 的 "Header_1"、"Header_2"、"Header_3" 键中
        headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ],
        # 是否在切分后的正文中删除标题本身。设为 False 表示保留标题在正文里，让语义更完整（设为 True 则标题只存在 metadata 里）
        strip_headers = False,
    )

    # 初始化一个空列表，用来存放最终切好的所有小块chunk
    final_chunks = []
    # 执行两阶段切分循环，为什么要两阶段？
    # 只按标题切：章节可能很长，超过 embedding 模型或检索粒度限制。
    # 只按长度切：会破坏 Markdown 结构，chunk 可能跨章节，语义混乱。
    # 两阶段结合：
    # 先按标题切，保证 chunk 不跨章节，保留结构信息；
    # 再按长度切，保证每个 chunk 不超过 chunk_size；
    # 最终 chunk 既有结构元数据，又长度可控。
    for doc in docs:
        # 第一刀（按标题切）：提取当前文档的 Markdown 正文 doc.page_content，按标题层级切分成若干个大章节
        # 注意：split_text() 返回的是全新的 Document 列表（header_docs），此时这些新 Document 的 metadata 里只有刚提取出来的标题，原本的 source（文件路径）和 file_type（文件类型）丢失了
        header_docs = header_splitter.split_text(doc.page_content)
        # 继承原文档的 metadata（如 source, file_type）
        # 补回原元数据： 遍历刚切出来的每个章节块 h_doc，用 .update(doc.metadata) 把原始文档的 source、file_type 等信息合并回 h_doc.metadata 中，确保每个子块都能溯源到原文件
        for h_doc in header_docs:
            h_doc.metadata.update(doc.metadata)
        # 第二刀
        # 再按长度拆成小块（直接接收 Document 列表，自动保留所有 metadata）
        # final_chunks.extend(...) 把当前文档切出的所有 chunk 追加到总列表
        final_chunks.extend(document_splitter(header_docs, chunk_size, chunk_overlap))

    return final_chunks
