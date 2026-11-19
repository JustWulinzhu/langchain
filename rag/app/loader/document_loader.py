from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: str):
    """读取 PDF 文件，返回Document列表"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    print(f"文件名称{path.name}")
    print(f"总页数{len(documents)}")
    print(f"documents: {documents}")

    return documents
