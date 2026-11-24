import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    BSHTMLLoader,
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


# ==============================================================================
# 模块一：全局正则常量
# 含义：匹配 Markdown 中由三反引号（```）包裹的完整代码块（如 ```json、```bash 等），
#       非贪婪匹配（[\s\S]*?）确保遇到第一个闭合 ``` 即停止，避免跨多个代码块误匹配。
# ==============================================================================
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)


# ==============================================================================
# 模块二：OCR 图像文字识别模块
# 含义：当输入为图片（如架构图、接口截图、扫描件）时，将图像内容转换为文本，
#       从而纳入后续的文本切分与向量检索流程。实际生产中需替换为真实 OCR 引擎。
# ==============================================================================
def run_ocr_on_image(image_path: str) -> str:
    """
    识别图片中的文本内容。

    参数:
        image_path: 图片文件路径

    返回:
        识别出的文本字符串
    """
    # 生产环境接入示例（按需安装对应依赖并取消注释）：
    # from rapidocr_onnxruntime import RapidOCR
    # ocr_engine = RapidOCR()
    # result, _ = ocr_engine(image_path)
    # if not result:
    #     return ""
    # return "\n".join([item[1] for item in result if item and len(item) > 1])

    return f"[待接入OCR引擎识别图片: {image_path}]"


# ==============================================================================
# 模块三：多格式文档加载模块
# 含义：针对 PDF、Word、Markdown、HTML、图片等不同文档形态调用对应加载器，
#       统一转换为 LangChain 标准 Document 对象（包含 page_content 正文与 metadata 元数据），
#       并尽量保留 Word 中的标题层级与表格信息。
# ==============================================================================
def load_api_document(file_path: str) -> List[Document]:
    """
    根据文件类型加载文档并返回 Document 列表。

    参数:
        file_path: 本地文件路径

    返回:
        List[Document]: 解析后的文档对象列表
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        # 1. 普通文本型 PDF：按页解析，每页生成一个 Document（metadata 中含 page 页码）
        #    若遇到复杂排版、多栏或扫描 PDF，可替换为 PyMuPDFLoader 或 UnstructuredPDFLoader
        loader = PyPDFLoader(str(path))
        return loader.load()

    elif suffix == ".docx":
        # 2. Word 文档：优先使用 UnstructuredWordDocumentLoader(mode="elements")
        #    按元素提取 Title（标题）、Table（表格）、正文等，尽量保留结构信息
        try:
            loader = UnstructuredWordDocumentLoader(str(path), mode="elements")
            elements = loader.load()

            formatted_parts: List[str] = []
            for el in elements:
                category = el.metadata.get("category", "")
                text = el.page_content.strip()
                if not text:
                    continue

                if category == "Title":
                    # 将 Word 标题转换为 Markdown 二级标题格式，方便后续按标题层级切分
                    formatted_parts.append(f"\n## {text}\n")
                elif category == "Table":
                    # 若解析出 HTML 表格，优先保留 text_as_html 以维持行列结构；否则回退纯文本
                    table_html = el.metadata.get("text_as_html")
                    formatted_parts.append(f"\n{table_html or text}\n")
                else:
                    formatted_parts.append(text)

            return [
                Document(
                    page_content="\n".join(formatted_parts),
                    metadata={"source": str(path), "file_type": "docx"},
                )
            ]
        except Exception:
            # 3. 兜底方案：若 Unstructured 依赖未安装或解析异常，使用轻量级 Docx2txtLoader
            return Docx2txtLoader(str(path)).load()

    elif suffix in {".md", ".markdown"}:
        # 4. Markdown 文档：直接按 UTF-8 读取，完整保留标题（#）、表格（|...|）和代码块
        content = path.read_text(encoding="utf-8")
        return [
            Document(
                page_content=content,
                metadata={"source": str(path), "file_type": "markdown"},
            )
        ]

    elif suffix in {".html", ".htm"}:
        # 5. HTML 文档：提取网页正文内容
        #    建议在正式上线前先用 BeautifulSoup 剔除侧边栏、导航栏、页脚等无关噪声
        loader = BSHTMLLoader(str(path), open_encoding="utf-8")
        return loader.load()

    elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        # 6. 图片文件：先调用 OCR 转成文本，再封装为标准 Document
        ocr_text = run_ocr_on_image(str(path))
        return [
            Document(
                page_content=ocr_text,
                metadata={"source": str(path), "file_type": "image_ocr"},
            )
        ]

    else:
        raise ValueError(f"暂不支持的文件后缀类型: {suffix}")


# ==============================================================================
# 模块四：代码块防截断保护模块
# 含义：普通字符切分器按长度硬切时，容易把完整的 JSON 请求/返回示例从中间断开，
#       导致检索出的片段缺少闭合括号或字段上下文。
#       本模块通过“占位符替换 -> 主体切分 -> 占位符还原”保证代码块完整保留。
# ==============================================================================
def split_preserving_code_blocks(
    text: str,
    base_splitter: RecursiveCharacterTextSplitter,
) -> List[str]:
    """
    在切分文本时保护完整代码块不被拆碎。

    参数:
        text: 待切分的文本内容
        base_splitter: 用于切分非代码块主体的递归字符切分器

    返回:
        切分并还原代码块后的文本片段列表
    """
    code_blocks: List[str] = []

    # 第一步：扫描所有代码块，存入 code_blocks，并用唯一占位符替换原位置
    def _replace_with_placeholder(match: re.Match) -> str:
        block_index = len(code_blocks)
        code_blocks.append(match.group(0))
        return f"\n[[CODE_BLOCK_{block_index}]]\n"

    masked_text = CODE_BLOCK_PATTERN.sub(_replace_with_placeholder, text)

    # 第二步：对替换后的文本执行常规递归切分
    raw_chunks = base_splitter.split_text(masked_text)

    # 第三步：遍历切分结果，把占位符还原为完整的代码块内容
    restored_chunks: List[str] = []
    for raw_chunk in raw_chunks:
        restored_text = raw_chunk
        for idx, block_content in enumerate(code_blocks):
            placeholder = f"[[CODE_BLOCK_{idx}]]"
            if placeholder in restored_text:
                restored_text = restored_text.replace(placeholder, block_content)

        cleaned = restored_text.strip()
        if cleaned:
            restored_chunks.append(cleaned)

    return restored_chunks


# ==============================================================================
# 模块五：API 文档结构化切分核心模块
# 含义：先按 Markdown 标题（# ~ ####）拆分出接口/章节逻辑单元，并把各级标题路径
#       写入 metadata；随后根据小节长度决定“整块保留”还是“防截断二次切分”，
#       兼顾检索精准度与参数表/JSON 示例的完整性。
# ==============================================================================
def split_api_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 80,
) -> List[Document]:
    """
    对 API 文档执行结构化切分。

    参数:
        documents: 原始 Document 列表
        chunk_size: 二次切分时的目标最大字符数（API 文档建议略大于普通短文本）
        chunk_overlap: 二次切分时相邻块之间的重叠字符数

    返回:
        切分完成且带有结构化元数据的 Document 列表
    """
    # 定义需要识别的标题层级及对应的元数据字段名
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]

    # strip_headers=False：保留标题行在正文中，避免片段丢失所属接口/模块名称
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    # 当单个标题小节过长时，按优先级从高到低（空行 -> 换行 -> 句号 -> 分号 -> 空格 -> 逐字）切分
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "。",
            "；",
            " ",
            "",
        ],
    )

    final_documents: List[Document] = []

    for doc in documents:
        # 保留原始文档元数据（如 source、page、file_type）
        base_metadata = dict(doc.metadata)

        # 第一层切分：按标题层级划分逻辑小节
        section_docs = header_splitter.split_text(doc.page_content)

        for section_doc in section_docs:
            # 合并原文档元数据与当前小节的标题元数据（h1 ~ h4）
            merged_metadata = {**base_metadata, **section_doc.metadata}

            # 拼接完整层级路径（例如："订单查询接口 > 返回示例"），便于后续定位与过滤
            section_path = " > ".join(
                merged_metadata[level]
                for level in ("h1", "h2", "h3", "h4")
                if merged_metadata.get(level)
            )
            if section_path:
                merged_metadata["section_path"] = section_path

            section_text = section_doc.page_content.strip()
            if not section_text:
                continue

            # 第二层处理：若当前小节总长度不超过 chunk_size 的 1.5 倍，优先整块保留，
            # 防止把“字段说明/参数表 + 请求/返回 JSON 示例”拆散到不同 chunk 中
            if len(section_text) <= int(chunk_size * 1.5):
                final_documents.append(
                    Document(page_content=section_text, metadata=merged_metadata)
                )
            else:
                # 若小节确实超长，则调用带代码块保护的切分函数进行二次拆分
                sub_texts = split_preserving_code_blocks(
                    text=section_text,
                    base_splitter=fallback_splitter,
                )
                for sub_idx, sub_text in enumerate(sub_texts):
                    sub_metadata = {**merged_metadata, "chunk_index": sub_idx}
                    final_documents.append(
                        Document(page_content=sub_text, metadata=sub_metadata)
                    )

    return final_documents


# ==============================================================================
# 模块六：本地运行示例入口
# 含义：模拟一份包含标题、参数表格与 JSON 请求/返回示例的 API 文档，
#       验证切分结果是否保留了结构信息与完整 JSON。
#       这里使用列表逐行拼接代替三引号嵌套，避免复制粘贴时出现引号冲突。
# ==============================================================================
if __name__ == "__main__":
    sample_api_markdown = "\n".join([
        "# 订单查询接口（GET /v1/orders/detail）",
        "",
        "## 请求参数",
        "| 字段名 | 类型 | 必填 | 说明 |",
        "| :--- | :--- | :--- | :--- |",
        "| order_id | string | 是 | 订单唯一编号 |",
        "",
        "## 请求示例",
        "```json",
        "{",
        '  "order_id": "ORD_20261124_001"',
        "}",
        "```",
        "",
        "## 返回示例",
        "```json",
        "{",
        '  "code": 0,',
        '  "message": "success",',
        '  "data": {',
        '    "order_id": "ORD_20261124_001",',
        '    "status": "PAID",',
        '    "amount": 199.00',
        "  }",
        "}",
        "```",
    ])

    # 构造初始 Document 对象
    input_docs = [
        Document(
            page_content=sample_api_markdown,
            metadata={"source": "order_detail_api.md", "file_type": "markdown"},
        )
    ]

    # 执行切分
    split_result = split_api_documents(
        documents=input_docs,
        chunk_size=800,
        chunk_overlap=80,
    )

    # 输出每个 chunk 的详细信息以便检查
    for index, item in enumerate(split_result, start=1):
        print(f"\n==================== Chunk {index} ====================")
        print("元数据 (metadata):", item.metadata)
        print("正文内容 (page_content):")
        print(item.page_content)