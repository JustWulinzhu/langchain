# 导入切分器
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 定义文本拆分的函数
# 参数documents: 通常是一个包含多个 langchain_core.documents.base.Document 对象的列表（例如通过 PyPDFLoader、TextLoader 或 DirectoryLoader 加载出来的文档）
def document_splitter(documents):
    """
    将文档拆分成多个 chunk
    """
    text_splitter = RecursiveCharacterTextSplitter(
        # 每个 chunk 的目标最大长度（默认按字符数计算，即 len(text)）。这里设为 256 字符，属于比较紧凑的粒度，适合做精确检索，但上下文略
        chunk_size = 256,
        # 相邻两个 chunk 之间的重叠字符数为 20
        chunk_overlap = 20,
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
    for index, chunk in enumerate(chunks):
        print(f"\n========== Chunk {index + 1} ==========")
        print("文本：")
        print(chunk.page_content)
        print("元数据：")
        print(chunk.metadata)

    return chunks
