import shutil
import uuid
from pathlib import Path
from app.config import DOCUMENT_DIR
from app.schemas import ChatRequest
from app.rag.splitter import document_splitter
from app.rag.vector_store import save_chunks
from app.loader.document_loader import load_pdf
from app.services.rag_service import ask_question
from fastapi import FastAPI, UploadFile, File, HTTPException

# 创建一个 FastAPI 后端服务实例 app
# 这里传入的 title、description 和 version 会自动渲染在 FastAPI 自带的交互式 API 文档页面（启动服务后访问 [http://127.0.0.1:8000/docs] 即可看到）
app = FastAPI(
    title = "AI API 文档知识库",
    description = "基于 RAG 的 API 文档智能问答系统",
    version = "1.0.0"
)

@app.get("/")
def index():
    return {
        "message": "AI API Knowledge Base is running"
    }

@app.post("/api/documents/upload")
def upload_document(file: UploadFile = File(...)):
    """
    上传 PDF 文档并写入知识库
    """

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="目前只支持 PDF 文件"
        )

    document_dir = Path(DOCUMENT_DIR)
    document_dir.mkdir(parents=True, exist_ok=True)

    # 使用 UUID 生成服务器文件名，避免文件名冲突
    document_id = str(uuid.uuid4())
    file_path = document_dir / f"{document_id}.pdf"

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. 读取 PDF
        docs = load_pdf(str(file_path))

        if not docs or not any(
            doc.page_content.strip() for doc in docs
        ):
            raise HTTPException(
                status_code=400,
                detail="PDF 没有提取到可用文本，可能是扫描件"
            )

        # 2. 记录原始文件名
        for doc in docs:
            doc.metadata["source"] = file.filename
            doc.metadata["document_id"] = document_id

        # 3. 文本切片
        chunks = split_documents(docs)

        # 4. 保存向量
        save_chunks(chunks)

        return {
            "message": "上传成功，知识库构建完成",
            "document_id": document_id,
            "filename": file.filename,
            "pages": len(docs),
            "chunks": len(chunks)
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"文档处理失败：{str(e)}"
        )

    finally:
        file.file.close()


@app.post("/api/chat", tags = ["RAG相关"])
def chat(request: ChatRequest):
    """
    RAG 智能问答
    """

    try:
        result = ask_question(request.question)
        return result
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = f"问答失败：{str(e)}"
        )
# 这个是啥意思 uvicorn app.main:app --reload --port 8000
# 一句话总结：这行命令的意思就是：“用 Uvicorn 服务器在本地 8000 端口启动 app/main.py 里的 app 应用，并且在代码改动时自动重启。”

# 详细解释：
# uvicorn Web 服务器程序。FastAPI 本身只负责写接口逻辑，不能直接对外接收网络请求，必须靠 uvicorn 这个高性能 ASGI 服务器把它运行起来，监听网络端口。

# app.main:app
# 告诉服务器去哪里找你的程序（格式为 文件夹.文件名:FastAPI实例名）：
# • app.main：指 app 文件夹下的 main.py 文件。
# • :app：指 main.py 文件里面创建的 app = FastAPI(...) 变量名。

# --reload 开启热重载（开发模式专用）。只要你修改并保存了代码文件，服务器会自动检测并瞬间重启生效，不需要你每次手动按 Ctrl+C 停止再重新运行命令。

# --port 8000 指定服务监听的端口号。启动后，你的接口服务就会运行在本地的 8000 端口上（即访问地址为 [http://127.0.0.1:8000](http://127.0.0.1:8000)）。如果不写这个参数，默认也是 8000 端口；如果 8000 被其他软件占用了，可以改成 --port 8001 等其他数字。