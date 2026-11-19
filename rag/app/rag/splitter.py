from langchain_text_splitters import RecursiveCharacterTextSplitter

def document_splitter(documents):
    """
    将文档拆分成多个 chunk
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 256,
        chunk_overlap = 20,
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
    chunks = text_splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        print(f"\n========== Chunk {index + 1} ==========")
        print("文本：")
        print(chunk.page_content)
        print("元数据：")
        print(chunk.metadata)

    return chunks
