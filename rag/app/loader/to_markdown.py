# 从 Python 标准库 pathlib 导入 Path
# Path 用于以面向对象方式处理文件路径，例如获取文件名、后缀、读取文本等
from pathlib import Path
from typing import List

# 从 LangChain 核心包导入 Document 类
# Document 通常包含两个核心字段：1、page_content：文档正文内容；2、metadata：元数据，例如来源文件、文件类型等
# 在 RAG、向量数据库、文本切分等流程中，Document 是常用的数据载体
from langchain_core.documents import Document

# 生产环境推荐安装：
# pip install markitdown pymupdf4llm pandas tabulate
import pymupdf4llm

# 微软开源的文档转 Markdown 工具
from markitdown import MarkItDown


def parse_to_markdown(file_path: str) -> List[Document]:
    """
    生产级多格式统一解析入口：将 PDF / Word / Excel / HTML / MD 统一高保真转换为 Markdown
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    # 1. PDF 文件：使用 pymupdf4llm（轻量高速）或 MinerU/Docling（复杂扫描/双栏PDF）
    #    它会自动识别标题字号和表格框线，直接输出带 |...| 表格的 Markdown
    if suffix == ".pdf":
        md_text = pymupdf4llm.to_markdown(str(path))
        return [
            Document(
                page_content=md_text,
                metadata={"source": str(path), "file_type": "pdf"},
            )
        ]

    # 2. Word (.docx) / Excel (.xlsx) / HTML / PPTX：
    #    使用微软开源的 MarkItDown，原生保留标题层级 (#) 和完整 Markdown 表格 (|...|)
    elif suffix in {".docx", ".xlsx", ".xls", ".html", ".htm", ".pptx"}:
        md_converter = MarkItDown()
        result = md_converter.convert(str(path))
        return [
            Document(
                page_content=result.text_content,
                metadata={"source": str(path), "file_type": suffix.lstrip(".")},
            )
        ]

    # 3. 原生 Markdown / 纯文本
    elif suffix in {".md", ".markdown", ".txt"}:
        content = path.read_text(encoding="utf-8")
        return [
            Document(
                page_content=content,
                metadata={"source": str(path), "file_type": "markdown"},
            )
        ]

    else:
        raise ValueError(f"不支持的文件格式: {suffix}")