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

# 微软开源的文档转 Markdown 的工具
# 对 Word (.docx)、Excel (.xlsx)、PowerPoint (.pptx)、HTML 等office格式的转换质量很高
# 但它的 PDF 处理能力被公认为最薄弱的环节，例如表格结构严重损坏等等，所以PDF还需要引入pymupdf4llm专业的工具
from markitdown import MarkItDown

# pymupdf4llm 的定位：专为 LLM/RAG 打造的高质量 PDF 解析器
import pymupdf4llm



### 一个典型的 RAG 流程是：加载文件 → 解析成文本 → 切分 → 向量化 → 存入向量库 → 检索 → 交给 LLM
### parse_to_markdown 的核心作用是：把各种格式的文件，统一转换成 LangChain 能直接消费的 Document 列表，并且内容都尽量转成 Markdown。
def parse_to_markdown(file_path: str) -> List[Document]:
    """
    生产级多格式统一解析入口：将 PDF / Word / Excel / HTML / MD 统一高保真转换为 Markdown
    """
    # 把传入的字符串路径 file_path 包装成 Path 对象
    path = Path(file_path)
    suffix = path.suffix.lower()

    # 1. PDF 文件：使用 pymupdf4llm（轻量高速）或 MinerU/Docling（复杂扫描/双栏PDF）
    #    它会自动识别标题字号和表格框线，直接输出带 |...| 表格的 Markdown
    if suffix == ".pdf":
        # str(path) 把 Path 对象再转回字符串路径，因为该函数通常接收字符串路径
        md_text = pymupdf4llm.to_markdown(str(path))
        return [
            Document(
                # page_content 是文档正文，这里就是 PDF 转出来的 Markdown 文本
                page_content = md_text,
                # metadata 是元数据字典："source"：原始文件路径；"file_type"：文件类型，这里固定为 "pdf"。元数据后续可用于溯源、过滤、展示来源等
                metadata = {"source": str(path), "file_type": "pdf"},
            )
        ]

    # 2. Word (.docx) / Excel (.xlsx) / HTML / PPTX：
    #    使用微软开源的 MarkItDown，原生保留标题层级 (#) 和完整 Markdown 表格 (|...|)
    elif suffix in {".docx", ".xlsx", ".xls", ".html", ".htm", ".pptx"}:
        # 创建 MarkItDown 转换器实例
        md_converter = MarkItDown()
        # 调用 convert() 进行转换
        result = md_converter.convert(str(path))
        return [
            Document(
                # 对MarkItDown来说，text_content 通常就是转换后的 Markdown 文本
                page_content = result.text_content,
                metadata = {"source": str(path), "file_type": suffix.lstrip(".")},
            )
        ]

    # 3. 原生 Markdown / 纯文本
    elif suffix in {".md", ".markdown", ".txt"}:
        # 使用 Path.read_text() 读取整个文件内容
        # 指定 encoding="utf-8"，避免默认编码不一致导致乱码
        content = path.read_text(encoding = "utf-8")
        return [
            Document(
                page_content = content,
                metadata = {"source": str(path), "file_type": "markdown"},
            )
        ]

    else:
        raise ValueError(f"不支持的文件格式: {suffix}")