from pathlib import Path
from typing import List
from langchain_core.documents import Document

# 生产环境推荐安装：
# pip install markitdown pymupdf4llm pandas tabulate
import pymupdf4llm
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